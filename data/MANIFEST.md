# Data manifest

All data is public and reproducible. Nothing here is committed; the repository stores only
checksums so an experiment can be tied to an exact input.

## Competition data (`data/raw/`)
Download with:
```
kaggle competitions download -c playground-series-s6e10 -p data/raw
```
| File | SHA-256 (first 16) |
|---|---|
| train.csv | see `data/raw/HASHES.txt` |
| test.csv | see `data/raw/HASHES.txt` |
| sample_submission.csv | see `data/raw/HASHES.txt` |

Shapes: train `(699635, 23)`, test `(299844, 22)`. Target `satisfaction`, positive rate
`310339 / 699635 = 0.44361`.

## External source dataset (`data/original/`)
The official competition Data page states the data was *"inspired by"* this dataset:
`arseniyshutko/binary-aviation-satisfaction-129k` (file `data.csv`).

Download with:
```
kaggle datasets download -d ars eniys hutko/binary-aviation-satisfaction-129k -p data/original/arseniyshutko__binary-aviation-satisfaction-129k --unzip
```

**Fingerprint evidence that this is the generator source (all verified locally):**

| Property | original | synthetic S6E10 |
|---|---|---|
| rows × cols | 129,880 × 22 | 999,479 × 22/23 |
| `Class` values | `Business`, `Eco`, `Eco Plus` (3) | identical 3 |
| `Baggage handling` values | 5 (never 0) | identical 5 |
| all other ratings | 0–5 (6 values) | identical 6 |
| `Age` distinct | 75 | 75 |
| synthetic values present in original | — | 100% (Age, both delays), 99.999% (Flight Distance) |
| exact full-row matches | — | 21 of 999,479 (0.002%) |
| positive rate | 0.4345 | 0.4436 |

Alternative repackagings of the same 129,880 rows (`teejmahal20/airline-passenger-satisfaction`,
`mysarahmadbhat/…`, `nilanjansamanta1210/…`, `raminhuseyn/airline-customer-satisfaction`,
`yakhyojon/customer-satisfaction-in-airline`, `binaryjoker/…`) were downloaded and compared;
none matches the synthetic value support better than `arseniyshutko`.

**Licence / rules**: S6E10 Rules §2.6 explicitly permit external data that is public, free and
equally accessible to all participants. This dataset satisfies that condition. Competition data
itself is CC BY 4.0; the survey is attributed to the public "Airline Passenger Satisfaction"
survey (Maven Analytics mirrors it as Public Domain).

## Phase22 public prediction archive (2026-10-19)

Dataset: `cdeotte/s6e10-tfm-oof-and-test-predictions`, version 1, CC0-1.0.
Local archive: `artifacts/phase22/shared_tfm_v1/dataset.zip` (37,044,032 bytes).
ZIP SHA-256: `fa819741aff0bb89ae19c2913a510d901921feaf4a611e42cabbb20548221f29`.
The 22 extracted NumPy file hashes are frozen in
`configs/phase22/private_breakthrough_20261019.json` under `external.file_sha256`.

Re-download the current release with the installed CLI (then require the exact ZIP hash):

```powershell
.\.venv\Scripts\kaggle.exe datasets download cdeotte/s6e10-tfm-oof-and-test-predictions -p artifacts/phase22/redownload_verify
```

The installed CLI has no version flag. For exact version 1 use the Kaggle connector
`download_dataset` request with `ownerSlug=cdeotte`,
`datasetSlug=s6e10-tfm-oof-and-test-predictions`, `datasetVersionNumber=1`,
`hasDatasetVersionNumber=true`. Download its returned URL without logging credentials,
and check the ZIP hash before extracting. A newer version must not silently replace v1.

These vectors are diagnostic only: producer row/context manifests and exact modified
library sources are absent; the announced seed42 folds differ from our primary folds.
No vector from this archive was promoted to a certified submission. Upstream model
licenses are reviewed separately in `reports/phase22/public_tfm_provenance_audit.json`.

## Phase24 public prediction pool (2026-10-23)

Downloaded for the external-OOF probe that produced candidate X. Local root
`data/external/s6e10_public/` is git-ignored (bulk); every file is re-downloadable:

    .\.venv\Scripts\python.exe -m kaggle datasets download -d cdeotte/s6e10-tfm-oof-and-test-predictions -p data/external/s6e10_public/s6e10-tfm-oof-and-test-predictions --unzip
    .\.venv\Scripts\python.exe -m kaggle datasets download -d goodpjw2008/s6e10-tabpfn-member -p data/external/s6e10_public/s6e10-tabpfn-member --unzip
    .\.venv\Scripts\python.exe -m kaggle datasets download -d najiama/s6e10-oof -p data/external/s6e10_public/s6e10-oof --unzip

The two files used by candidate X are frozen by SHA-256:

- `05_blend_oof.csv`   `4aa93222e885c8d3604f03f4b2297b66c31029a2d7e2d82d137738fbb06bee7a`
- `05_submission.csv`  `a6d49839ad00a20931dd884c7be5ddc7a9dad2d0e83c1298692b246b0f566794`

Provenance: author `najiama`, datasetId 12333973. The licence field is `unknown`; the
dataset description grants explicit permission to use and reproduce the files with credit.
The blend members are cited by the author (yekenot, ektarr, goodpjw2008, evgendvorkin,
kospintr, cdeotte). Their OOF uses `StratifiedKFold(5, shuffle, random_state=42)`.

Policy note: the Phase22 ruling "external vectors are diagnostic only" was written for
promoting an external *model* on its claimed merits. Candidate X is a different use: a fixed
blend whose delta is measured on our own immutable folds, replicated on shadow, with a
pre-committed public consistency rule and v7 retained as the second finalist. The promotion
remains a user decision, recorded in `reports/phase24/SUBMISSION_DECISION.md`.
