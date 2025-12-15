from pathlib import Path
import argparse
import platform
import torch

def pick_backend(arg_backend: str) -> str:
    if arg_backend != "auto":
        return arg_backend
    mach = platform.machine().lower()
    if "arm" in mach or "aarch64" in mach:
        return "qnnpack"
    return "fbgemm"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_ckpt", required=True, help="FP32 state_dict checkpoint (.pt)")
    ap.add_argument("--out_ts", required=True, help="Output TorchScript model (.pt)")
    ap.add_argument("--backend", default="auto", choices=["auto", "fbgemm", "qnnpack"])
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=512)

    # How to handle ConvTranspose2d (upsampling in U-Net)
    ap.add_argument("--convtranspose", default="skip", choices=["skip", "per_tensor"],
                    help="skip keeps ConvTranspose2d in FP32; per_tensor tries per-tensor quant (may still fail)")

    args = ap.parse_args()

    # Import model from src/model
    THIS_DIR = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(THIS_DIR))
    from small_unet import SmallUNet

    backend = pick_backend(args.backend)
    torch.backends.quantized.engine = backend
    print("Quant backend:", backend)

    # Load FP32 model on CPU (quantization is CPU oriented)
    model = SmallUNet().cpu()
    sd = torch.load(args.fp32_ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # FX PTQ
    import torch.ao.quantization as aq
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
    import torch.nn as nn

    qconfig_mapping = get_default_qconfig_mapping(backend)

    if args.convtranspose == "skip":
        # Safest: keep ConvTranspose2d float
        qconfig_mapping = qconfig_mapping.set_object_type(nn.ConvTranspose2d, None)
        print("ConvTranspose2d: skipped (kept FP32)")
    else:
        # Try per-tensor weights for ConvTranspose2d (less likely to hit per-channel assertion)
        from torch.ao.quantization.observer import default_observer, default_weight_observer
        q_per_tensor = aq.QConfig(activation=default_observer, weight=default_weight_observer)
        qconfig_mapping = qconfig_mapping.set_object_type(nn.ConvTranspose2d, q_per_tensor)
        print("ConvTranspose2d: per-tensor weight observer")

    example = torch.rand(1, 3, args.image_size, args.image_size)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=(example,))

    # Calibration
    with torch.no_grad():
        for _ in range(args.calib_batches):
            prepared(torch.rand(1, 3, args.image_size, args.image_size))

    quantized = convert_fx(prepared)
    quantized.eval()

    # Save TorchScript
    ts = torch.jit.trace(quantized, example)
    outp = Path(args.out_ts)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(outp))
    print("Wrote quantized TorchScript:", outp)

if __name__ == "__main__":
    main()
