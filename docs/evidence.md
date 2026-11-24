# What the measurements support

## Development history

| Period / series | Question and decision |
|---|---|
| Early laboratory and biological blueprint | Surveyed cellular compartments, transmission, local calcium and plasticity. Missing human-specific values stayed unknown. The executable cells are phenomenological approximations, not accepted physiological parameter sets. |
| V1–V14 and C1–C6 (before the N series) | Small MLPs and bounded replay established continual-learning controls. Common training could fit tasks that sequential training lost. More coverage or protected gradients did not reliably yield stable learning. These were comparison methods, not biological synapses or components of the current core. |
| 12–13 Nov 2026, N1.0–N1.8 | Integrated dendrite/soma state, local calcium and durable synaptic changes; separated those from transient excitability and threshold feedback. N1.7 synaptic protection was not adopted. Local inhibition was tested separately, rather than equating more spiking with better discrimination. |
| N1.9A/B → N1.9C (cohort completed 20 Nov) | Tested whether history-dependent responses transfer to learned discrimination of permutations under supervision. The cohort and frozen prefix diagnostic below separate retention of information from correct action. |
| N2.0/2.1 (13 Nov) and N2-XYZ (22 Nov) | Dimension-generalized sparse routing and experimental edge replacement were developed separately. The reported geometry test deliberately fixed topology after construction, isolating the effect of choosing neighbors. |
| 20–22 Nov, N1.9 auxiliary paths | Event-addressed readout, permission and post-action feedback were separated from core learning. Rhythm and second confirmation tested narrow mechanisms using frozen cell states. |
| 22 Nov, moving-flash pilot; 24 Nov, Report 001 | A direction-sensitive readout was tested on a harder spatial panel. The report's three main experiments and its earlier negative sequence results are mapped below. The flash pilot is separately identified. |

This is a report-scoped release, not an export of every exploratory run. Historical entries outside the released evidence are context only.

## Spatial wiring: a real intervention, a negative decision result

Four paired seeds (22401–22404). For each seed and arm: 32 A presentations, then 32 B with no fresh A lessons; identical starting weights and held-out inputs; cold readout every 16 lessons. Each A or B denominator is 4 seeds × 16 held-out presentations. A tie or silence is wrong.

| Wiring | A after A | A after B | B after B | Mean physical edge length (unit cube) |
|---|---:|---:|---:|---:|
| XYZ | 62/64 | 64/64 | 64/64 | 0.4161 |
| XY | 61/64 | 62/64 | 63/64 | 0.4548 |
| Random cycle | 62/64 | 64/64 | 64/64 | 0.6697 |

All arms passed the per-task 13/16 floor on all seeds. The preregistered benefit required ≥5 percentage points over **both** controls, positive paired changes on ≥3/4 seeds each, and no worse forgetting. It failed. XYZ vs random changed 126/128 final cell-count vectors and 4/4 final weight matrices, but **no final decisions**. A ceiling and four seeds limit inference: equal observed scores are not proof of equivalence. [Inputs, 60 checkpoints and seed records](../results/xyz/) · [scorer](../src/neural_engine/score_xyz.py).

## Rhythm: when a prefix cannot justify a decision

Four previously used core seeds, new physical panels; 48 histories × 2 carriers × 2 mappings × 4 seeds = 768 paired judgments per main condition. Inputs are matched within pairs. Four structured rhythms share the same per-channel sample multiset and dose. GO requires no premature spike and both the correct first margin and correct vote. Prohibition requires **complete silence**, not merely a tied vote.

| Input | Delay only | Memory of level |
|---|---:|---:|
| Clean GO, alternating GO, GO with initial plateau (each) | 768/768 | 768/768 |
| Block-shaped prohibition | 0/768 | 768/768 |
| Misleading-prefix prohibition | 0/768 | 0/768 |
| Level prohibition, no signal (each) | 768/768 | 768/768 |

On the misleading prohibition, both first emitted at offset +3; the earliest physical difference from alternating GO was +5. The memory gate reduced the number of open ticks but still answered every time. The basic criterion passed; the stronger claim of whole-rhythm discrimination failed. Reversed release dictionaries were negative controls. The two scoring routes agree on 12,288 related readings; they do not constitute independent experimental runs. [Physical inputs and raw spike masks](../results/rhythm/) · [gate oracle and scalar scorer](../src/neural_engine/score_rhythm.py).

## Confirmation: protecting a contact has a price

One frozen core (202611203), one held-out stimulus seed (202611213), four successive Y lessons. Each checkpoint is read after reload, without feedback. In the noise world teacher and post-action bit agree on one wrong correction; in the shift world the same observation reflects a real change. K1 information delivered to the learner is matched. The held-out panel excludes training words and prefixes. At each K and marker the full criterion is 16/16 conflicting, 4/4 agreeing, 4/4 vetoes silent, and no out-of-window emissions.

| Correct Y responses among the 16 conflicting examples | K1 | K2 | K3 | K4 |
|---|---:|---:|---:|---:|
| Noise, ordinary filter | 0 | 16 | 16 | 16 |
| Noise, double confirmation | 16 | 16 | 16 | 16 |
| Real shift, ordinary filter | 0 | 16 | 16 | 16 |
| Real shift, double confirmation | 0 | 0 | 16 | 16 |

At K1 in the true shift the double filter still gave the **old sign**; at K2 it was silent. X passed throughout, and there were no emissions on 160 veto readings. The noise condition demonstrates different X/Y operations on this bounded panel. After the true shift both require the same operation: an eventual merge does not demonstrate preservation of two distinct final rules. Two coherent errors in succession were not tested. [Packed banks, transactions, held-out waves and 24 raw batches](../results/confirmation/) · [independent transaction oracle](../src/neural_engine/score_confirmation.py).

## Sequence core: loss of readout, not total erasure

Thirty seeds (11100–11129), 96 sequence lessons, points 0/24/48/96, two wiring conditions × five teaching controls. Train, novel test and previously learned isolated A/B are different panels. Primary acquisition required ≥75% novel-test accuracy **and** ≥15 points gain vs baseline per seed; the cohort decision also required ≥24 successful seeds and superiority to frozen and label-shuffled controls. In local/immediate: 0/30 acquired; mean final train 49.98%, test 50.01%. In local/delayed, 45/60 old task/seed pairs crossed from ≥80% to <80%; **15 were already below criterion**, so none of these 15 was counted as lost. Changes in weights are not acquisition. Half-credit ties here differ from the strict spatial scorer. [Complete cohort scoring and archived per-seed results](../results/cohort/).

In a frozen diagnostic subset (three reused seeds, 120 opposite-order pairs per seed), inputs match from tick 44. At tick 63, immediately **before** Q, calcium differs by ≥1% of the model's threshold interval on 360/360 pairs in each condition. Despite that, the *entire 16-cell raster* on ticks [64,161) was identical across opposite histories in 59/360 pairs initially, 237/360 after 24 immediate lessons and 264/360 after 24 delayed lessons. This is a mechanistic constraint on that readout, not proof that the remaining internal states encode a generalizable class or that calcium alone causes forgetting. [Paired states and rasters](../results/prefix/).

## Supplement: moving flash

Eight new seeds (22541–22548); the fixed decoder uses left/right ordering of cell positions and a six-tick lag. Matched forward/reverse motion, common-endpoint and final-flash-only controls separate motion from endpoint cues. XYZ got 179/192 vs 190/192 for random and 178/192 for X-only. XYZ won on 0/8 seeds, with four ties; the +5 pp / ≥6 wins screen failed. [Data](../results/flash/) · [stimulus and scoring definitions](../src/neural_engine/flash.py).

## Resource interpretation

| Original run | Worker CPU | Wall | Peak worker RSS |
|---|---:|---:|---:|
| XYZ | 26.09 s | 127.19 s | 35.7 MiB |
| Rhythm | 45.67 s | 216.97 s | 190.6 MiB |
| Confirmation | 8.49 s | 40.29 s | 44,608 KiB |

These are descriptive measurements of the original implementations, including different validation and pacing overheads; they are not portable reference benchmarks. Gate state costs +8 transient bytes per stream; the confirmation rule adds +2 persistent bytes. Neither is total program memory. There is no measured energy in joules, no hardware-matched Transformer comparison and no estimate of a general energy advantage.
