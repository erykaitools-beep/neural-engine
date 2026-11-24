# Neural Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24498021.svg)](https://doi.org/10.5281/zenodo.24498021)

**Can a small spiking circuit learn a temporal relation, keep an earlier association, and wait until a decision is justified?** Neural Engine is an independent research project that tests these questions one mechanism at a time. This repository contains the code, synthetic stimuli, checkpoints and output records behind [Research Report No. 001](https://doi.org/10.5281/zenodo.24492624), including experiments that failed.

A useful distinction runs through this work: two input histories can leave different states in a neuron without producing different spikes, and different spikes need not mean the circuit has learned a rule. The experiments measure these steps separately.

## Start here

- **Read the publication:** [Report No. 001, Zenodo, v1.0](https://doi.org/10.5281/zenodo.24492624). It is a research report, not a peer-reviewed paper.
- **Check a result in a few commands:** [installation and reproduction](docs/reproduction.md). The default path reads existing records; it does not train a network or require a GPU.
- **Understand what runs:** [cell equations, readout and memory](docs/architecture.md).
- **See the evidence and its limits:** [development history, protocols and results](docs/evidence.md), including the tasks on which simpler controls matched or beat the candidate.

## What did the experiments show?

| Question | Evidence | What it does **not** establish |
|---|---|---|
| Does placing neurons in a unit cube help them retain A while learning B? | XYZ wiring shortened mean inhibitory edge length from 0.670 to 0.416. Against a random cycle the final 16-cell spike-count vector changed in 126/128 presentations, but the decision changed in **0/128**. Four paired seeds. | No demonstrated gain in classification; the task was near ceiling. No propagation delay or learned placement was modeled. |
| Can a gate recognize permission by rhythm rather than duration? | Remembering one previous signal level rejected **768/768** block-shaped prohibitions versus **0/768** for a delay-only gate. Both made a false response on **all 768** misleading-prefix prohibitions. | This was an engineered local-change detector with frozen weights, not a learned, general rhythm classifier. The trials were not 768 independent networks. |
| Can a second confirmation protect a memory against bad feedback? | On one frozen core, delaying a weakening update protected Y against one jointly wrong teacher/outcome. When Y actually changed, reliable new responses started at lesson **3 rather than 2**. X stayed functional. | The rule costs two persistent bytes. There is no independent multi-seed confirmation of the effect; after the true shift, X and Y eventually require the same operation. |
| Did the core itself learn sequence order? | In the 30-seed N1.9C cohort the primary local/immediate condition met the acquisition criterion on **0/30** seeds. Mean test accuracy was **50.01%**, with half credit for ties. | Subthreshold history is not reliable learned classification. Auxiliary contact results do not reverse this negative result. |

**Separate supporting pilot:** a spatial moving flash with matched forward/reverse inputs and a fixed direction decoder gave XYZ **179/192** versus random **190/192**. This is neither learned vision nor evidence against all geometry.

## Architecture and research aim

The reference cells have two dendritic compartments, a soma that emits spikes, eight afferent weights, local calcium-like state and bounded local plasticity. A spike, a change of threshold and a change of synaptic weight are distinct events. Some protocols add sparse inhibition or an engineered contact that selects which somatic population receives current. Only the components named by each protocol are active. In particular, the confirmation experiment updates an auxiliary memory **after** an action, not the cell core.

The long-term question is whether such mechanisms can support continual adaptation and eventually language at lower resource cost. These prototypes do not process natural language, are not a calibrated human brain model and have no measured energy advantage over Transformers. No joules were measured in the experiments here.

## Repository map

| Directory | Contents |
|---|---|
| [src/neural_engine](src/neural_engine) | NumPy reference models, gate and contact mechanisms, experiment implementations and independent score readers |
| [experiments](experiments) | Command-line entry points for archived verification and bounded fresh reproduction |
| [configs](configs) | Cell constants, seed-specific starting states, historical stimulus specifications and reference fixtures |
| [results](results) | Synthetic physical inputs and/or generation records, response arrays, packed contact states, checkpoints and scores |
| [tests](tests) | Positive controls on frozen data and negative integrity checks |
| [docs](docs) | Architecture, evidence, reproduction instructions, provenance and release review |
| [papers](papers) | Publication record and citation; the authoritative report is hosted on Zenodo |

## Use, attribution and collaboration

The code is licensed under [MIT](LICENSES/MIT.txt). All other project-owned materials, including synthetic data, checkpoints, results and documentation, are licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). The [license scope and attribution example](LICENSE) explain how to reuse each. The Zenodo report remains CC BY 4.0; third-party works retain their own terms.

See [CITATION.cff](CITATION.cff) to cite the report, [CONTRIBUTING.md](CONTRIBUTING.md) for replicating or challenging an observation, and [ROADMAP.md](ROADMAP.md) for open questions. OpenAI Codex and ChatGPT assisted with implementation, analysis and writing; the [disclosure](docs/provenance.md) explains their role. Neither is a scientific author or an independent laboratory.

**Software release:** [v0.1.0 on Zenodo](https://doi.org/10.5281/zenodo.24498021). The software and Report No. 001 have separate DOIs. See [CHANGELOG.md](CHANGELOG.md) for what is included and what has not been rerun.
