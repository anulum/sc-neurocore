# SCLeakyTwoCompartmentLIFNeuron

**Module:** `sc_neurocore.neurons.models.sc_leaky_tc_lif`
**Family:** Compartmental (soma + dendrite, leaky, hard reset)
**State variables:** `v_s` (soma potential), `v_d` (dendrite potential)
**Identity:** count-neutral SC identity — a preserved repository
recurrence, not a publication-exact model.

## Equations

$$\tau_s \frac{dV_s}{dt} = -(V_s - V_{rest}) + \kappa (V_d - V_s) + I_{soma}$$
$$\tau_d \frac{dV_d}{dt} = -(V_d - V_{rest}) + I_{dend}$$

Euler step of size `dt`; the soma derivative consumes the freshly
updated dendrite value. Spike when $V_s \ge \theta$; the soma
HARD-resets to $V_{reset}$, the dendrite is untouched. Two external
currents (`i_soma`, `i_dend`) — the original two-current API is
preserved.

## Provenance and preservation

This is the recurrence formerly published as
`TwoCompartmentLIFNeuron` in the Python reference. It is structurally
distinct from the Zhang et al. (2024) TC-LIF (which is a leak-free
integrator map with a negative somato-dendritic coupling and soft
subtraction resets) — see [TwoCompartmentLIFNeuron](tc_lif.md) for the
canonical paper-exact model. Finite-input trajectories are preserved
**bit-for-bit** from the pre-2026-08-27 implementation; the frozen
anchors live in `tests/test_model_sc_leaky_tc_lif.py`.

## Invalid-Input Atomicity

`step(i_soma, i_dend=0.0)` raises `ValueError` with the pre-step state
preserved exactly for non-finite currents, an out-of-bounds
configuration, or an overflowing candidate; the added validation does
not alter any finite-input trajectory.

## Custody boundary

| Surface | Status |
|---------|--------|
| Python reference + tests | implemented (`tests/test_model_sc_leaky_tc_lif.py`) |
| Engine / Go / Julia / Mojo / RTL | intentionally not implemented for this preserved identity; no parity claimed |
