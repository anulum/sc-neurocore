# Alpha-Synapse LIF — Source Fidelity

This page records exactly which parts of the primary literature the
maintained `AlphaNeuron` implements, which parts it does not, and the
executed evidence behind every claim.

## Primary-source boundary

- **Rall (1967), *Distinguishing theoretical synaptic potentials computed
  for different soma-dendritic distributions of synaptic input*, J.
  Neurophysiol. 30(5), 1138–1168.** The alpha function as a postsynaptic
  response shape. The maintained two-state cascade (rise `a`, current `i`)
  per synapse reproduces this kernel exactly for a pulse input.
- **Gerstner & Kistler (2002), *Spiking Neuron Models*, Cambridge
  University Press, §4.1, [DOI 10.1017/CBO9780511815706](https://doi.org/10.1017/CBO9780511815706).**
  The leaky integrate-and-fire framing with filtered synaptic currents.

The model is the dual excitatory/inhibitory current-based alpha-synapse
LIF, and the identity **holds** against both sources. An earlier
public-docs line citing Rall 1962 (Ann. N.Y. Acad. Sci. 96) was a
misattribution for this artefact and is corrected: Rall 1962's *Theory of
physiological properties of dendrites* is a different paper.

Defaults (`tau_v=20`, `tau_exc=5`, `tau_inh=10`, dimensionless scale) are
catalogue/model-family choices, not source-derived parameters. The exact
piecewise-constant-input timestep is a derived exact flow — an engineering
contract, not a biological publication claim.

## Exact flow and the event convention

Each step applies the exact filter relaxation per cascade and the exact
alpha-current convolution for the membrane, with the equal-time-constant
limit handled analytically in every production lane:

```
a' = ss + (a - ss) * exp(-dt / tau),           ss = tau * I
i' = ss + exp(-dt / tau) * ((i - ss) + (a - ss) * dt / tau)
v' = v_ss + (v - v_ss) * exp(-dt / tau_v) + C_exc - C_inh
on candidate crossing v' >= v_threshold: only v <- v_rest
```

`C` is the exact alpha-current convolution; the equal-tau limit is the
analytic `rate * decay * (i*dt + a*dt^2 / (2*tau))` branch. The event is
detected on the post-update candidate at maintained step boundaries; no
within-step root localisation is claimed.

## Independent reference evidence

The committed reference trace
`reference_trace_data/alpha_dual_synapse_doi.json` is reproduced by an
independent re-derivation (no production `step` code): constant dual drive
`I_exc = 2.5`, `I_inh = 0.5`, 160 steps, 51 spikes with the first at step
8. Every feature (spike count, first spike step, and final/min/max/mean of
all five states) matches at `1e-12` absolute.

## Numerical and atomic contract

- Invalid input, invalid configuration, and non-finite candidates are
  rejected before any state mutation in every maintained lane.
- `reset()` restores only the documented state (`v = v_rest`, cascades to
  zero) and preserves the complete configuration.
- The batch contract returns all five state trajectories plus the spike
  trace, five final-state receipts, and the spike count; the dispatcher
  validator re-checks the somatic reset on every spike.

## Executable parity matrix

| Lane | Evidence | Result |
|---|---|---|
| Python golden | `tests/test_model_alpha.py` | exact reference |
| Rust engine (PyO3) | `tests/test_alpha_parity.py` | complete traces within `1e-12` |
| Standalone Rust safety | `tests/test_alpha_rust_parity.py` | complete traces within `2e-15` |
| Julia (juliacall) | `tests/test_alpha_julia_parity.py` | within `1e-12`; typed buffer failures atomic |
| Go (C ABI) | `tests/test_alpha_go_parity.py`, `tests/test_alpha_native_abi.py` | within `1e-12`; all overlap/null/status classes rejected without writes |
| Mojo (C ABI) | `tests/test_alpha_mojo_parity.py`, `tests/test_alpha_native_abi.py` | within `1e-10`; identical ABI classes |
| Dispatch contracts | `tests/test_alpha_accel_dispatch_contracts.py` | selection, validation, reload, malformed-result boundaries |

## Paired schema and Q32.32 co-simulation

The TOML and JSON schema models are structurally identical and preserve the
hand model's exact flow within `5e-12` over a varied 256-step dual drive
(36/36 events). The schema names the excitatory drive `I` and exposes the
inhibitory level as the overridable parameter `inh_drive`, so the generated
single-input RTL stays well-formed; the production model accepts a full
per-step inhibitory vector.

Generated Q32.32 SystemVerilog is validated at the enrolled grid-exact
operating point (`tau_v = 8`, `tau_exc = 4`, `tau_inh = 2`, `dt = 1` — all
exponential arguments on the 0.125-step lookup grid, all rates distinct):

- measured maximum state error over the 256-step sign-changing drive:
  `v: 1.95e-8`, cascades below `7.6e-9` (declared envelope `0.01`);
- the complete 24-entry event vector is identical at both enrolled
  inhibitory levels (0.0 and 0.5);
- every RTL spike resets `v` to `v_rest`;
- a depth-4 Z3 bounded job proves reset safety only
  (`hdl/formal/catalogue/sc_alpha_synapse_lif.sby`, PASS);
- Yosys `synth` completes the generated module.

The equal-time-constant limit is implemented in all production lanes; the
schemas carry the general branch with the documented boundary. No formal
equivalence, synthesis timing, device, or PPA claim is made; the silicon
tier is H1.

## Controlled benchmark

`benchmarks/results/bench_alpha.json` records the five-runtime
200,000-step dual-drive batch (non-default initial state) measured with
five repeats on one pinned logical CPU, with host load, tool versions,
source and binary SHA-256 hashes, and run order recorded. All five lanes
return matching events, final states, and trace digests within the
declared tolerances: Rust/Julia byte-identical to Python, Go within
`3.8e-14`, Mojo within `8.9e-15`. This is local regression evidence only;
no production speed claim is made.

## Boundaries

- No within-step root localisation; events are sampled at maintained step
  boundaries.
- The equal-tau convolution limit is production-only; schemas document the
  boundary.
- Defaults are model-family choices, not source parameters.
- The benchmark is non-exclusive single-CPU evidence, not an isolated
  production measurement.

## Reproduction

```bash
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/alpha.rs -o /tmp/alpha && /tmp/alpha
python -m pytest \
  tests/test_model_alpha.py \
  tests/test_alpha_dynamics.py \
  tests/test_alpha_backends.py \
  tests/test_alpha_parity.py \
  tests/test_alpha_rust_parity.py \
  tests/test_alpha_julia_parity.py \
  tests/test_alpha_go_parity.py \
  tests/test_alpha_mojo_parity.py \
  tests/test_alpha_native_abi.py \
  tests/test_alpha_accel_dispatch_contracts.py \
  tests/test_cosim_alpha.py \
  tests/test_reference_alpha.py \
  tests/test_bench_alpha.py -q
cd hdl/formal/catalogue && sby -f sc_alpha_synapse_lif.sby
taskset -c 0 env PYTHONPATH=$WHEEL_SITE:src:. python \
  benchmarks/bench_model_alpha.py
```
