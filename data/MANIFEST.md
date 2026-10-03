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