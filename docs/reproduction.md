# Reproducing Report 001

## 1. Install in a checkout

Use Python 3.10 and NumPy 2.2.6 for the closest documented environment. There are no GPU, pretrained-model downloads, service accounts or API keys. Run from the repository root. On Linux/macOS:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/verify.py integrity
python -m unittest discover -s tests -v
python experiments/verify.py all
```

If NumPy is available offline, use that verified environment or an approved local wheel. A wheel is not bundled. On Windows use `.venv\Scripts\python.exe` in place of `python`; the archive scorers do not use Linux hardware interfaces. Windows and macOS have not been qualified for numerical parity. No editable install or PYTHONPATH is required for these commands. The relative locations of `src/`, `results/` and `configs/` matter.

## 2. Re-score without training

`python experiments/verify.py STUDY` checks the SHA-256 manifest first and then reads saved evidence. Output goes to the terminal. A nonzero exit is a failed check. `all` runs the six entries below.

| STUDY | Primary evidence / data format | Expected outcome |
|---|---|---|
| `xyz` | Physical topology, routing impulses, 60 JSON checkpoints and per-sample cell counts | XYZ/random: 126 differing final count vectors, zero differing decisions; benefit screen false |
| `rhythm` | Float physical inputs, uint16 spike masks, gate enables and currents in `INPUTS.npz`; row/phase metadata in JSON | 12,288 scalar cross-check readings; 768 paired gains on the block prohibition, 768 paired failures on the misleading prefix |
| `confirmation` | 24 raw NPZ batches, spike and permission records; actual pre/post-bank and pending bytes; held-out and learning stimuli | Four policies/world combinations; 160 veto readings, zero emissions. Stable correct Y after true shift at K2 (single) or K3 (double) |
| `flash` | Generation parameters, 80-tick bit masks and fixed decoder | XYZ 179 vs random 190 on 192 matched-motion presentations; screen false |
| `cohort` | Original per-seed result JSONs losslessly compressed with gzip, separate panel specifications, pre-sequence checkpoints | Full numerical summary recomputed, including journals and reset checks; acquisition, retention, joint screens false |
| `prefix` | Nine numeric NPZ projections: opposite-class indices, labels, cell-local calcium after tick 63, and complete raster | 59 / 237 / 264 identical post-Q raster pairs with a different pre-Q calcium state |

In a spike mask, bit *i* is the event of cell *i* at that tick; a zero mask is silence. A summary must be interpreted with its own tie convention (see [evidence](evidence.md)). Do not average over repeated reads as though each were a newly trained network.

## 3. Fresh execution on an independent host

This is a **separate** operation. The source is a portable NumPy reference port, not the historical machine-specific launcher or a compiled binary. Scientific outputs are checked against archived outcomes, including exact event arrays; host timings are not expected to match. Stop and record discrepancies; do not round results until they agree. The supplied checkpoint is the start of the named experiment, not a claim that all its upstream development was repeated from random initialization.

Use a Linux research machine under your control. Identify a reliable, live **CPU package** temperature sensor that reports millidegrees Celsius; a GPU, battery, constant reading or guessed path is not a substitute. The supervisor requires a new output directory outside the checkout, a single allowed CPU, address space ≤1 GiB, CPU time ≤85 s, wall time ≤300 s and output ≤120 MiB. It starts below 60 °C, stops at ≥75 °C or unreadable sensor, and does not retry. These are conservative release limits; larger/slower machines and unqualified sensors may be unable to complete a command. Sampling is not hard real time. Use `--idle-unit UNIT` on a shared machine with a known protected workload; `--dedicated-host` is only for a genuinely separate dedicated host. If your site cannot enforce these conditions, use archived verification.

Examples **for an external reproducer**, after setting `CPU_SENSOR` to the validated local sysfs input and `OUT` to an existing directory outside the checkout:

```sh
# Frozen readouts: one independent unit per command where noted
python experiments/reproduce.py rhythm --seed 202611203 --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/rhythm-202611203"
python experiments/reproduce.py prefix --seed 11100 --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/prefix-11100"
python experiments/reproduce.py flash --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/flash"

# These protocols DO learn; they are not part of installation, verification or tests
python experiments/reproduce.py xyz --allow-learning --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/xyz"
python experiments/reproduce.py confirmation --allow-learning --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/confirmation"
python experiments/reproduce.py cohort --seed 11100 --arm local --policy immediate --allow-learning --sensor "$CPU_SENSOR" --dedicated-host --output "$OUT/cohort-11100-local-immediate"
```

For rhythm the other archived seeds are 202611202, 202611262 and 202611263. For prefix they are 11101 and 11102. For the sequence cohort the design is seeds 11100–11129 × `baseline/local` × `immediate/delayed/time_shuffle/label_shuffle/frozen`. One seed-condition is the unit of the fresh cohort command; there is deliberately no automatic long-training sweep. Do not retry a safety stop or increase limits to obtain a pass.

Fresh XYZ regenerates examples and compares all checkpoint files, counts and the scientific summary. Rhythm takes **recorded physical waves** and replays both gates, comparing currents, spikes and enables. Confirmation regenerates the held-out waves, runs the frozen network plus four contact lessons, and compares raw arrays, transaction logs and serialised banks. Prefix recreates the input order from the cohort specs and compares pre-Q calcium and post-Q events, including reset controls. Flash regenerates inputs and recomputes every spike and signed direction score. The fresh cohort path checks the journal after each lesson and the outputs at each named checkpoint.

## Interpretation and troubleshooting

- **Integrity check** means the listed bytes agree with this release's manifest. The manifest is not a digital signature. Obtain a trusted archive or commit.
- **Saved-data verification** means computations from those observations agree. It does not train a model or produce independent observations.
- **Frozen parity** means a specified saved example/batch gave the same internal outputs after model reload. The public test suite covers examples from XYZ, rhythm and confirmation; it does not exercise every teaching update.
- **Fresh matched replay** would test whether the port reproduces the original *same* seed and examples. Its full learning paths were not run to prepare this release. **Independent replication** calls for other researchers and new data/seeds, not merely another pass over these files.
- If NumPy import fails under a fresh-run budget, check the isolated environment and single-threaded BLAS settings. If a comparison fails, record the first mismatch and versions. Retain partial outputs separately; a partial directory is never evidence of a completed reproduction.
