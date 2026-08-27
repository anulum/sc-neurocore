# SCExponentialTwoCompartmentLIFNeuron

**Module:** `sc_neurocore.neurons.models.sc_exponential_tc_lif`
**Family:** Compartmental (soma + dendrite, exponential decay, hard reset)
**State variables:** `v_s` (soma potential), `v_d` (dendrite potential)
**Identity:** count-neutral SC identity — a preserved engine
recurrence, not a publication-exact model.

## Equations

$$V_d[t] = e^{-dt/\tau_d} V_d[t-1] + I_{dend}[t]$$
$$V_s[t] = e^{-dt/\tau_s} V_s[t-1] + I_{soma}[t] + \kappa V_d[t]$$

Spike when $V_s \ge \theta$; the soma HARD-resets to $V_{reset}$, the
dendrite is untouched. Two external currents (`i_soma`, `i_dend`) —
the original two-current API is preserved.

## Provenance and preservation

This is the recurrence formerly published as the production-engine
`TwoCompartmentLIFNeuron`. It is structurally distinct from both the
Zhang et al. (2024) TC-LIF and the SC leaky variant — see
[TwoCompartmentLIFNeuron](tc_lif.md) for the canonical paper-exact
model. The production Rust engine keeps this recurrence **verbatim**
as `SCExponentialTwoCompartmentLIF` (PyO3:
`SCExponentialTwoCompartmentLIF`), anchored to trajectories captured
from the pre-2026-08-27 built engine; the Python reference here
reproduces those anchors bit-exactly.

## Invalid-Input Atomicity

The Python `step(i_soma, i_dend=0.0)` raises `ValueError` with the
pre-step state preserved exactly for non-finite currents, an
out-of-bounds configuration, or an overflowing candidate. The engine
class keeps the historical permissive behaviour verbatim (preservation
takes precedence on that surface) and is exercised against the Python
reference in `tests/test_model_sc_exponential_tc_lif.py`.

## Custody boundary

| Surface | Status |
|---------|--------|
| Python reference + tests | implemented (`tests/test_model_sc_exponential_tc_lif.py`) |
| Production Rust engine + PyO3 | implemented verbatim (`SCExponentialTwoCompartmentLIF`), frozen anchors |
| Go / Julia / Mojo / RTL | intentionally not implemented for this preserved identity; no parity claimed |
