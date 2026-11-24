# Architecture: what is actually simulated?

There are two related cell implementations and several experiment-specific readouts. They are **not** the successive layers of one trained language model. All quantities below are engineering quantities unless a unit is explicitly stated. The scalar constants in [cell.json](../configs/cell.json) are not estimates of human synaptic parameters.

## One cell and one tick

For the reported circuits, eight external channels feed two compartments in groups of four. Each cell has eight afferent weights, a somatic threshold and spike-count readout. At 16 cells this gives 128 afferent weights and 16 thresholds; connectivity, auxiliary tables and fast state are additional, not included in that count.

```text
eight input channels → decaying presynaptic traces p
                    → weighted sums on two dendrites d
                    ↔ soma s → threshold crossing → spike / reset
                         ↑                          │
                 external current           adaptation, backpropagation
                 and inhibition               and neighbor inhibition
dendritic voltage + presynaptic trace → local calcium surrogate
                                      → change of efficacy (learning trials only)
```

1. Update each input trace with exponential decay towards the current input. Take the weighted sum on each branch.
2. Update the coupled branch and soma voltages from **old** voltages. Effective drive includes external current, the old adaptive current and (in the local arm) the old inhibitory current. Threshold crossings reset the soma and add to the spike count.
3. Spiking increments a backpropagating trace and an adaptation current. Neighbor spikes produce inhibition for later steps. This is a discrete-time current model, not a reconstructed neuron morphology.
4. A bounded voltage gate on the **new dendritic voltage**, combined with presynaptic trace, contributes an NMDA-like source. A second voltage-only term is VDCC-like. Local calcium decays and integrates those sources. These labels describe dependencies, not specific molecular channels.
5. When learning is enabled, calcium above the upper threshold sets a high weight target, calcium between thresholds sets a low target, and calcium below the lower threshold leaves the weight unchanged. The increment is `(1 − exp(−dt × rate × p)) × (target − weight)`. There is no gradient descent or backpropagation of error in this rule. The externally addressed teacher is a substantial supervision assumption.

Read [base_cell.py](../src/neural_engine/base_cell.py) (`relax`, `Network.step`), [adaptation.py](../src/neural_engine/adaptation.py), [homeostasis.py](../src/neural_engine/homeostasis.py), and [circuit.py](../src/neural_engine/circuit.py) in that order for the N1 path. The N2 [sparse implementation](../src/neural_engine/sparse.py) stores a sparse inhibitory neighbor list; its optional event route dispatches neighbor events, **not** all cellular computation. Both spatial experiments here use pull routing.

## What persists?

| Path | Survives restoration | Cleared or rebuilt |
|---|---|---|
| N1/N2 cell core | Afferent weights, thresholds, fixed graph or graph identity, seed, model/config identity | Voltages, presynaptic trace, Ca, adaptation, inhibition and counters |
| Temporal selection | Archived ring of signed pair observations and its count; trained context and release-table snapshots | Prior address, event/latch and gate state for each stream |
| Confirmation | Serialised contact bank (initially five bytes); two additional bytes of pending evidence for double confirmation | The episode-local permission latch, provisional selection, current and spikes |

A restored factorized system consists of **all** its specified starting fixtures, not of a five-byte brain. Byte counts for the contact exclude the cell core, upstream features, Python overhead and episode buffers.

## Where behavior comes from

**Spatial A→B.** [xyz.py](../src/neural_engine/xyz.py) samples positions once, constructs nearest-unvisited cycles in XYZ or XY versus a random cycle, and checks actual spike delivery. The cycles have the same number of edges, common strength and identical initial afferent weights. Coordinates affect the choice of neighbors. They are not looked up to compute conduction delays, and are not learned. Graphs stay fixed while afferent weights learn. [spatial_protocol.py](../src/neural_engine/spatial_protocol.py) handles distinct A and B lessons, preview-based threshold updates, the teaching current and restored measurements. [flash.py](../src/neural_engine/flash.py) additionally assigns spatial input fields and a fixed lagged direction decoder. It does not train them.

**Sequence core.** In [cohort.py](../src/neural_engine/cohort.py) the same input templates occur in different orders with a common query Q. Six permutations are balanced across two classes; evaluation counts spikes from Q to the end in two groups of eight cells. For the old isolated-association test, the same output groups are scored on a different task. A high old-task score is not evidence of learning the new order. Immediate, delayed, shuffled and frozen conditions distinguish changes in weights from useful credit assignment.

**Temporal gate.** [context.py](../src/neural_engine/context.py) samples channel-pair history at input edges. [replay.py](../src/neural_engine/replay.py) delivers or withholds prepared somatic current; it does not alter the sensory input. Both gate variants use the same learned release table and require continuity over samples. The candidate adds one float64 per stream; it compares consecutive weak signal levels, not the complete future rhythm. The scorer knows the onset; the running gate does not.

**Contact memory.** A frozen bank of channel-pair and context features proposes an addressed response. [observed_stream.py](../src/neural_engine/observed_stream.py) passes the actual current through a permission gate. [credit_port.py](../src/neural_engine/credit_port.py) latches which contact was selected; [outcome_port.py](../src/neural_engine/outcome_port.py) processes feedback after the action. [confirm.py](../src/neural_engine/confirm.py), [merge.py](../src/neural_engine/merge.py) and [writer.py](../src/neural_engine/writer.py) govern changes to compact memory. Only proposed **weakening of a nonzero contact** requires two corroborating episodes in the new arm. For a zero contact after silence the teacher can still supply the sign without an outcome bit. The pending bytes are not an inference feature. This is supervised, engineered memory, not evidence that calcium learned the first/last operator.

## Boundaries

No molecular kinetics, human calibration, continuous 3D migration, learned conduction speeds, general-purpose attention replacement, unsupervised language acquisition or neuromorphic energy result is demonstrated. Structural rewiring was a distinct earlier study, not part of the static XYZ experiment.
