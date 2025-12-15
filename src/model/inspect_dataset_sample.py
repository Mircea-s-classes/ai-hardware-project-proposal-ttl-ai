from pathlib import Path
import sys
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from cnn_dataset import BubbleSegDataset
from small_unet import SmallUNet

ds = BubbleSegDataset("data/cnn/real_water", split="val")
print("val samples:", len(ds))

x, y = ds[0]
print("x shape:", tuple(x.shape), "min/max/mean:", float(x.min()), float(x.max()), float(x.mean()))
print("y shape:", tuple(y.shape), "min/max/mean:", float(y.min()), float(y.max()), float(y.mean()))
print("y>0.5 coverage:", float((y > 0.5).float().mean()))

m = SmallUNet().cpu()
m.load_state_dict(torch.load("model/real_water_unet.pt", map_location="cpu"))
m.eval()

with torch.no_grad():
    logits = m(x.unsqueeze(0))
    probs = torch.sigmoid(logits)
print("logits min/max/mean:", float(logits.min()), float(logits.max()), float(logits.mean()))
print("probs  min/max/mean:", float(probs.min()), float(probs.max()), float(probs.mean()))
