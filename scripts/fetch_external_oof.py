"""Download every public S6E10 OOF library into one staging directory.

Staging lives outside the repo on purpose: these files are bulk data and must never be
committed. The script is idempotent -- an existing non-empty directory is skipped.
"""
from __future__ import annotations

import contextlib
import io
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

STAGE = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool")

REFS = [
    "cdeotte/s6e10-124-public-oof-and-test-predictions",
    "dariushafshar/s6e10-golem-oof-library",
    "sachith7/s6e10-stack-oof-predictions",
    "koumeimaki/s6e10-route-tfm-nn-oof",
    "kagankoral/s6e10-ablation-members-oof",
    "kratosyan/s6e10-busyaprime-five-members-oof",
    "goodpjw2008/s6e10-goodpjw2008-extra-members-oof",
    "megayak/s6e10-oof-library",
    "najiama/s6e10-oof",
]


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api.authenticate()
    for ref in REFS:
        slug = ref.replace("/", "__")
        out = STAGE / slug
        if out.exists() and any(out.rglob("*")):
            print(f"skip {slug}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                api.dataset_download_files(ref, path=str(out), unzip=False, quiet=True)
        except Exception as exc:  # a missing/private dataset must not stop the rest
            print(f"FAIL {slug}: {type(exc).__name__} {str(exc)[:120]}")
            continue
        zips = list(out.glob("*.zip"))
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            z.unlink()
        print(f"ok   {slug}: {len(list(out.rglob('*')))} entries")


if __name__ == "__main__":
    main()
