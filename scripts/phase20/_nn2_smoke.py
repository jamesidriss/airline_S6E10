import sys, torch
sys.path.insert(0, '.')
from src.phase20 import neural_arch

F = 22  # route view features
cfgs = {
    "saint": {}, "autoint": {}, "deepfm": {}, "xdeepfm": {}, "node": {},
    "grande": {}, "tabnet": {}, "excelformer": {}, "amformer": {},
    "t2gformer": {}, "trompt": {},
}
x = torch.randn(128, F)
for name, cfg in cfgs.items():
    try:
        m = neural_arch.build(name, F, **cfg)
        out = m(x)
        assert out.shape == (128,), f"shape {out.shape}"
        n = sum(p.numel() for p in m.parameters())
        # backward smoke too
        out.sum().backward()
        print(f"{name:14s} OK params={n:>10,} out_mean={out.mean().item():+.4f}")
    except Exception as e:
        print(f"{name:14s} FAIL {type(e).__name__}: {e}")
