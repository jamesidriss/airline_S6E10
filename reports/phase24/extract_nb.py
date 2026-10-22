import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for tag, p in [
    ("najiama", ROOT / "reports/phase24/public_notebooks/najiama/high-scoring-oof-cv-0-96221-lb-0-96187.ipynb"),
    ("kirilltym", ROOT / "reports/phase24/public_notebooks/kirilltym/0-96206-disagreement-gated-teachers.ipynb"),
    ("mayank", ROOT / "reports/phase24/public_notebooks/mayank/s6e10-lb-surface-ascent-v2-0-96206.ipynb"),
]:
    nb = json.load(p.open(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        out.append("--- cell %d %s ---" % (i, c["cell_type"]))
        out.append("".join(c["source"]))
    dst = ROOT / "reports/phase24/public_notebooks" / (tag + ".txt")
    dst.write_text("\n".join(out), encoding="utf-8")
    print(tag, "cells", len(nb["cells"]), "chars", sum(len("".join(c["source"])) for c in nb["cells"]))
