import json
import pathlib
import sys

import pandas as pd

d = json.loads(pathlib.Path("reports/original_analysis.json").read_text(encoding="utf-8"))
what = sys.argv[1] if len(sys.argv) > 1 else "support"
key_filter = sys.argv[2] if len(sys.argv) > 2 else ""

rows = []
for k, v in d["candidates"].items():
    if key_filter and key_filter not in k:
        continue
    print("=" * 115)
    print(k, v.get("shape"), "common_cols:", v.get("n_common_cols"), "missing:", v.get("missing_vs_syn"))
    print("labels:", v.get("label_counts"))
    for key, label in [
        ("exact_full_row_match_frac_synth", "full-row"),
        ("exact_match_frac[survey14]", "survey14"),
        ("exact_match_frac[all_but_delay]", "all_but_delay"),
        ("exact_match_frac[all_but_dist_age]", "all_but_dist_age"),
    ]:
        if key in v:
            print(f"   rowmatch {label:<18} {v[key]:.6f}")
    if what == "support":
        for c, s in (v.get("support") or {}).items():
            extra = ""
            if s["orig_vals_not_in_syn"]:
                extra += f"  ORIG_ONLY={s['orig_vals_not_in_syn']}"
            if s["syn_vals_not_in_orig"]:
                extra += f"  SYN_ONLY={s['syn_vals_not_in_orig']}"
            print(f"   {c:<38} n_orig={s['n_orig_vals']:<6} n_syn={s['n_syn_vals']:<6} inter={s['n_inter']:<6} synrowcov={s['frac_syn_rows_covered_by_orig_support']:.4f}{extra}")
    elif what == "marg":
        for c, s in (v.get("marginals") or {}).items():
            print(f"   {c:<38} {s}")