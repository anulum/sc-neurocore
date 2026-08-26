# Hill–Tononi source-fidelity validation

## Claim

`HillTononiNeuron` implements the single-cell hybrid recurrence described by
Hill and Tononi (2005), using the paper's cortical-excitatory waking defaults
and its fixed `0.25 ms` classical RK4 step. The implementation is validated at
the recurrence, state, and event levels across Python, Rust, Julia, Go, and
Mojo.

## Sources and transcription boundary

The primary authority is Hill and Tononi,
*Modeling sleep and wakefulness in the thalamocortical system*,
[doi:10.1152/jn.00915.2004](https://doi.org/10.1152/jn.00915.2004). The
maintained NEST `ht_neuron` equations are used only to disambiguate parentheses
in the publication's printed persistent-sodium, depolarisation, and
low-threshold-calcium expressions. NEST's documented modifications are not
silently imported.

The default activates the currents assigned to a cortical excitatory cell in
the waking profile. `I_h` and `I_T` are available as explicit optional currents
but have zero default conductance. Network connectivity, synaptic kinetics,
miniature events, and population-level sleep/wake parameters are excluded.

## Evidence

- `tests/test_model_hill_tononi_source_fidelity.py` independently transcribes
  the derivatives, RK4 update, threshold event, and post-spike pulse. It checks
  defaults, gating anchors, optional currents, reset, and failure atomicity.
- `src/sc_neurocore/neurons/reference_receipts/hill_tononi_2005.json` fixes a
  768-step mixed-drive trace with two exact events and SHA-256
  `64aaf9659f1c9c3e4233dfd73f5f21f143a1718e2dc29d69a6db657e1d911b9b`.
- Native Rust, Go, Julia, and Mojo tests execute the source recurrence rather
  than a placeholder. The source-bound benchmark compares the five maintained
  runtimes for 200,000 steps at `I_ext = 20`, where each emits 538 events.
- Paired TOML and JSON schemas contain the complete state, parameter,
  integration, threshold, and identity contract.

## Identity correction

The former `HillTononiNeuron` used a six-state Hodgkin–Huxley-like recurrence
with intracellular sodium and a pump. Those states and equations do not occur
in the paper's model neuron. The recurrence has not been deleted: it is retained
under the count-neutral internal name `SCSixStateThalamocorticalNeuron` without
Hill–Tononi attribution, preserving compatibility while removing the false
scientific identity.

## Limits

This evidence validates a scalar software model, not the full published
thalamocortical network. It does not establish biological fit beyond the source
profile, equivalence to adaptive solvers, or universal binary64 identity across
math libraries. No RTL, synthesis, formal-equivalence, timing, PPA, board, or
device result is claimed.
