from pathlib import Path
import argparse
import sys
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_ckpt", required=True)
    ap.add_argument("--out_ts", default="model/real_water_unet_int8_ts.pt")
    ap.add_argument("--calib_images", default="", help="optional: folder of images for calibration (not required for this basic flow)")
    args = ap.parse_args()

    # pick backend
    arch = (torch.backends.quantized.engine or "").lower()
    # set based on platform
    if arch not in ("fbgemm", "qnnpack"):
        # heuristic: ARM -> qnnpack, else fbgemm
        import platform
        if "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower():
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = "fbgemm"

    device = torch.device("cpu")  # quantized runs on CPU
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(args.fp32_ckpt, map_location=device))
    model.eval()

    # FX graph mode PTQ (Conv-friendly)
    import torch.ao.quantization as aq
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    qconfig = aq.get_default_qconfig(torch.backends.quantized.engine)
    qconfig_dict = {"": qconfig}

    example = torch.randn(1, 3, 256, 256)
    prepared = prepare_fx(model, qconfig_dict, example_inputs=(example,))

    # calibration: run a few random tensors (good enough for a demo)
    with torch.no_grad():
        for _ in range(50):
            prepared(torch.rand(1, 3, 256, 256))

    quantized = convert_fx(prepared)
    quantized.eval()

    ts = torch.jit.trace(quantized, example)
    outp = Path(args.out_ts)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(outp))
    print(f"[quantized] engine={torch.backends.quantized.engine} -> {outp}")

if __name__ == "__main__":
    main()
