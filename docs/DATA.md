# Data

FedIDS-Bench evaluates on five public intrusion-detection corpora. **None of them is shipped with this repository** (they are
gigabytes in size and have their own licenses); download them yourself and point the scripts at the folder.
Nothing below was verified from the build sandbox, whose network allow-list blocks every dataset host — check the landing pages
for the current download links.

| Corpus | Files the loader reads | Landing page | Rows (order of magnitude) |
|---|---|---|---|
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` (the *partitioned* CSVs) | research.unsw.edu.au/projects/unsw-nb15-dataset | 0.26 M |
| CICIDS2017 | the eight `*.pcap_ISCX.csv` files of *MachineLearningCVE* | unb.ca/cic/datasets/ids-2017.html | 2.8 M |
| CSE-CIC-IDS2018 | the ten day files `??-??-2018.csv` of the processed-traffic CSVs | unb.ca/cic/datasets/ids-2018.html | 16 M |
| IoT-23 | CSV exports of the Zeek `conn.log.labeled` files, named `dataset*.csv` | stratosphereips.org/datasets-iot23 (Zenodo 4743746) | up to ~300 M flows in the full capture set (exports are usually sub-sampled) |
| ToN_IoT (network) | `train_test_network.csv` | research.unsw.edu.au/projects/toniot-datasets | 0.46 M |

## Folder layout

`--data-dir DIR` (default `data/raw`) may either contain the files directly or one sub-folder per corpus. The loaders look, in
this order, in `DIR`, `DIR/<folder>`, `src/fedids_bench/data/<folder>` and `<repo>/data/<folder>`:

```
DIR/
├── UNSW_NB15/                                  UNSW_NB15_training-set.csv  UNSW_NB15_testing-set.csv
├── Network Intrusion dataset(CIC-IDS- 2017)/   Monday-WorkingHours.pcap_ISCX.csv  ...  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)/  02-14-2018.csv  02-15-2018.csv  ...  03-02-2018.csv
├── IOT-23 Full Dataset/                        dataset1.csv  dataset2.csv  ...
└── TON_IoT Network Dataset/                    train_test_network.csv
```

With `pool: all` (what every paper grid uses) the loader pools **every** file of the corpus that matches the corpus's pattern, so
the capture days are mixed instead of the first file only. `pool: first` (the library default, kept for the old tests) reads the
first file found.

## Check the data before running anything

```bash
python scripts/describe_datasets.py --data-dir DIR --pool all     # rows, classes, attack share, fields found, duplicate rate
python scripts/signal_check.py      --data-dir DIR               # does the feature set carry signal at all? (minutes)
FEDIDS_DATA_DIR=DIR pytest tests/test_loaders.py tests/test_real_datasets.py -v
```

`describe_datasets.py` writes `results/dataset_description.{json,csv}`, which feed the dataset table of the paper. Read its
"harmonized fields missing" line for each corpus and compare it with the table below. **A core field that should exist but is
reported missing means the column mapping in `src/fedids_bench/data/features.py` (`SOURCE_COLUMNS`) does not match your copy of the
files** — the benchmark then silently degrades to constant-zero features (this exact failure is why the earlier CICIDS2017
numbers were meaningless), so fix the mapping before training anything.

## What the loaders do

1. **Pool** — read only the needed columns, keep a uniform random fraction of the rows of the *whole* file(s) (chunked Bernoulli
   sampling; the old loaders took the first N rows of time-ordered captures). Pool size is `max(10 × n_samples, 250 000)` rows.
2. **Labels** — the native label becomes a binary label (`Benign`/`Normal`/`BENIGN`/`0` → 0, everything else → 1) and an
   attack-category code (UNSW `attack_cat`, CIC `Label`, IoT-23 detailed label, ToN-IoT `type`).
3. **Canonical fields** — map each corpus onto the same flow fields: duration (seconds), forward/backward bytes and packets,
   protocol, destination port, service name, and a connection outcome (from the Zeek/Argus state code, or from TCP-flag counts).
4. **De-duplication** — exact duplicates of the canonical fields + label are removed *before* any split (`deduplicate: true`).
5. **Sampling** — `n_samples` distinct rows, proportional per attack category, no replacement.
6. **Features** — 31-feature `core` set: log-scaled volumes, packet sizes, byte/packet asymmetry, byte/packet rates, one-hot
   protocol, service group, destination-port range and connection outcome. Fields a corpus does not ship become the explicit
   `unknown` category (never a made-up number). `core22` and `core5` are ablations.
7. **Splits** — stratified train / validation / test; the `StandardScaler` is fitted on the **train** rows only.

### Which fields each corpus provides

| Corpus | Missing canonical fields (→ `unknown`) | Connection outcome from |
|---|---|---|
| UNSW-NB15 (partitioned CSVs) | destination port | `state` |
| CICIDS2017 (MachineLearningCVE) | protocol, service | TCP-flag counts |
| CSE-CIC-IDS2018 | service | TCP-flag counts |
| IoT-23 | — | `conn_state` |
| ToN_IoT | — | `conn_state` |

### Tuning rows are never reporting rows

`row_role: tune` and `row_role: report` split the pool by a hash of each row's *content* (`tune_row_fraction`, default 20 %), so the
rows used to choose hyper-parameters (`scripts/tune_hparams.py`) never overlap the rows behind any reported number, and identical
flows cannot land on both sides. Tuning scores the **validation** split only; the test split is touched once, by the grids.

## Cache

The sampled pool is cached as a pickle in `DIR/.fedids_cache` (or `dataset.cache_dir`), keyed by loader version, file sizes and
mtimes, pool size, seed and row role. The first run per (corpus, seed, row role) therefore reads the raw files — minutes for
CSE-CIC-IDS2018 — and every later run is fast. It is safe to delete the folder; changing a data file invalidates its entries
automatically.

## Troubleshooting

* `FileNotFoundError: Data for … not found … (see docs/DATA.md)` — the corpus is not under `--data-dir`; check the layout above.
* `UnicodeDecodeError` — CIC files contain cp1252 bytes; the CIC loaders already read with `latin-1`. A re-encoded copy is fine.
* A corpus shows 100 % attacks or a handful of classes — you pointed at a single-attack day. Use `--pool all` with all day files.
* IoT-23 in the original Zeek format (`conn.log.labeled`, tab-separated with `#fields` comments) must be exported to CSV first;
  the loader expects the Zeek field names as the header and the compound `tunnel_parents   label   detailed-label` last column.
* Memory: the largest pool is ~0.5 M rows × a dozen columns, i.e. well under 1 GB; the raw CSVs are streamed in chunks.
