import json
import pathlib
import sys

import pandas as pd

which = sys.argv[1] if len(sys.argv) > 1 else "survey"
d = json.loads(pathlib.Path("reports/forensics/forensics.json").read_text(encoding="utf-8"))

if which == "survey":
    for k, v in d["survey_value_counts"].items():
        print("---", k)
        for r in v:
            vk = [x for x in r if x not in ("train", "test", "train_frac", "test_frac", "diff")]
            vals = "/".join(f"{r[x]}" for x in vk)
            print(f"   {vals:>10}  train={r['train']:>7} ({r['train_frac']:.5f})  test={r['test']:>7} ({r['test_frac']:.5f})  diff={r['diff']:+.5f}")
elif which == "ks":
    print(pd.DataFrame(d["marginal_ks"]).sort_values("ks", ascending=False).to_string(index=False))
elif which == "ints":
    print(pd.DataFrame(d["integer_columns"]).T.to_string())
elif which == "surfaces":
    for k, v in d["target_rate_surfaces"].items():
        df = pd.DataFrame(v)
        sp = df["pos_rate"].max() - df["pos_rate"].min()
        print(f"--- {k}  spread={sp:.4f}")
        print(df.to_string(index=False))
else:
    print(json.dumps(d[which], indent=2)[:6000])