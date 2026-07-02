# Learning

Training paradigms beyond single-node STDP: BPTT, truncated BPTT,
eligibility traces, reward-modulated learning, meta-learning,
homeostatic scaling, short-term plasticity, structural plasticity,
federated learning, lifelong/continual learning, neuroevolution,
learning-rate schedulers, and training callbacks.

## BPTT Learner

::: sc_neurocore.learning.advanced.BPTTLearner

## Truncated BPTT (Williams & Peng 1990)

Chunks long sequences into windows of k timesteps, backpropagating
gradients within each chunk while carrying membrane state forward.
Memory O(k) instead of O(T).

::: sc_neurocore.learning.advanced.TBPTTLearner

## Eligibility Traces (e-prop, Bellec et al. 2020)

::: sc_neurocore.learning.advanced.EligibilityTrace

## Reward-Modulated STDP

::: sc_neurocore.learning.advanced.RewardModulatedLearner

## Learning-Rate Schedulers

Import these schedulers from either `sc_neurocore.learning` or
`sc_neurocore.learning.schedulers`. They keep only deterministic local state and
return the current learning rate from each `step()` call.

::: sc_neurocore.learning.schedulers.StepScheduler

::: sc_neurocore.learning.schedulers.ExponentialScheduler

::: sc_neurocore.learning.schedulers.CosineScheduler

::: sc_neurocore.learning.schedulers.WarmupCosineScheduler

## Training Callbacks

Import callbacks from either `sc_neurocore.learning` or
`sc_neurocore.learning.callbacks`. `CSVCallback` has no optional dependencies;
`TensorBoardCallback` and `WandBCallback` fail closed with `SCDependencyError`
when their optional runtime packages are not installed.

::: sc_neurocore.learning.callbacks.TrainingCallback

::: sc_neurocore.learning.callbacks.CSVCallback

::: sc_neurocore.learning.callbacks.TensorBoardCallback

::: sc_neurocore.learning.callbacks.WandBCallback

## Bounded O(1) Online Learning

`sc_neurocore.learning.online_o1` defines the hardware-facing local-learning
contract for streamed online updates. One synapse stores only four bounded
state fields: current weight, pre trace, post trace, and eligibility trace. The
memory proof emitted by `build_online_o1_memory_proof(...)` is independent of
sequence length and reports the exact per-synapse bit count used by the HDL
emitter.

The first supported rule family is reward-modulated STDP with fixed-point
saturation. The Python reference and optional Rust C-FFI wrapper share the same
hardware domain: `weight_bits` is `1..=31`, `trace_bits` is `2..=30`,
`reward_bits` is `1..=30`, learning and decay shifts are `0..=30`, and bool or
non-integral aliases are rejected before state mutation or ctypes conversion.
Runtime rewards remain saturating fixed-point inputs. `OnlineO1Config.to_scnir_annotation(...)`
emits deterministic SC-NIR metadata for online-learning-capable synapses, and
`sc_neurocore.hdl_gen.OnlineO1LearningEmitter` emits a one-lane Verilog update
block with the same bounded state fields and saturation policy. This software
and RTL contract is local handoff evidence; it does not claim board synthesis
or physical FPGA learning evidence until those runs are attached separately.
When the optional native library is present, `RustOnlineO1Synapse` provides the
same bounded fixed-point step contract through the Rust C-FFI bridge.

`OnlineO1LearningEmitter.estimate_resources(...)` reports deterministic
pre-synthesis BRAM/LUT/DSP planning estimates for a chosen synapse count. The
reproducible adaptation benchmark command is:

```bash
PYTHONPATH=src python tools/online_o1_adaptation_benchmark.py \
  --output benchmarks/results/online_o1_adaptation_benchmark.json
```

The default local report is deterministic simulation evidence: it measures
pre/post reward-pairing adaptation speed, records Python/Rust parity when the
native library is present, and marks `hardware_measurement_claimed=false`.

::: sc_neurocore.learning.online_o1.OnlineO1Config

::: sc_neurocore.learning.online_o1.OnlineO1Snapshot

::: sc_neurocore.learning.online_o1.OnlineO1Synapse

::: sc_neurocore.learning.online_o1.build_online_o1_memory_proof

::: sc_neurocore._native.learning_bridge.RustOnlineO1Synapse

## Meta-Learning (MAML, Finn et al. 2017)

::: sc_neurocore.learning.advanced.MetaLearner

## Homeostatic Plasticity (Turrigiano 2008)

::: sc_neurocore.learning.advanced.HomeostaticPlasticity

## Short-Term Plasticity (Tsodyks-Markram 1997)

::: sc_neurocore.learning.advanced.ShortTermPlasticity

## Structural Plasticity

::: sc_neurocore.learning.advanced.StructuralPlasticity

## Federated

::: sc_neurocore.learning.federated

## Lifelong (EWC)

Elastic Weight Consolidation with active penalty: pushes drifted weights
back toward consolidated values, weighted by Fisher information.

::: sc_neurocore.learning.lifelong

## Neuroevolution

::: sc_neurocore.learning.neuroevolution
