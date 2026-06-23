# ChayKeizerNeuron

**Module:** `sc_neurocore.neurons.models.chay_keizer`
**Reference:** Chay, T.R. & Keizer, J. (1983). Minimal model for membrane oscillations
in the pancreatic beta-cell. *Biophysical Journal* 42:181–190.
DOI [10.1016/S0006-3495(83)84384-7](https://doi.org/10.1016/S0006-3495(83)84384-7)
**Family:** Bursting (conductance-based pancreatic beta-cell)
**State variables:** `v` (membrane potential), `m`/`h` (calcium-current activation and
inactivation gates), `n` (delayed-rectifier potassium activation), `ca` (free cytosolic
calcium).

This is the original five-dimensional Chay–Keizer model. A voltage-gated calcium current
with Hodgkin–Huxley `m³h` kinetics and a delayed-rectifier potassium current with `n⁴`
kinetics generate fast spikes; calcium entering during the active phase slowly accumulates
and activates a calcium-dependent potassium conductance, which eventually terminates the
burst. Calcium then decays through a silent phase until the next burst begins, so calcium is
the slow variable that packages spikes into square-wave bursts.

---

## Equations

### Membrane potential (Eq. 8)

$$C_m \frac{dV}{dt} = I_{app} + 2\,I_{Ca} + I_K + I_{K(Ca)} + I_L$$

Currents follow the conductance convention $g\,(E_{rev}-V)$ (inward positive). The factor
two on $I_{Ca}$ reflects the divalent calcium charge.

$$I_{Ca} = \bar g_{Ca}\, m^3 h\, (E_{Ca}-V), \qquad
I_K = \bar g_K\, n^4\, (E_K-V)$$
$$I_{K(Ca)} = \bar g_{K(Ca)}\, \frac{[Ca]}{[Ca]+K_d}\, (E_K-V), \qquad
I_L = g_L\, (E_L-V)$$

### Gate kinetics

Each gate follows $\dot x = \phi\,(\alpha_x(V)(1-x) - \beta_x(V)\,x)$, with the
Hodgkin–Huxley 1952 rate functions evaluated at the shifted potential — $V+V'$ for the
calcium gates $m,h$ and $V+V^{*}$ for the potassium gate $n$ — and the temperature factor
$\phi = 3^{(T-6.3)/10}$.

$$\alpha_m = \frac{-0.1\,(U_m-25)}{\exp(-(U_m-25)/10)-1},\quad \beta_m = 4\,e^{-U_m/18}$$
$$\alpha_h = 0.07\,e^{-U_m/20},\quad \beta_h = \frac{1}{\exp(-(U_m-30)/10)+1}$$
$$\alpha_n = \frac{-0.01\,(U_n-10)}{\exp(-(U_n-10)/10)-1},\quad \beta_n = 0.125\,e^{-U_n/80}$$

where $U_m = V+V'$ and $U_n = V+V^{*}$.

### Calcium dynamics (Eq. 9)

$$\frac{d[Ca]}{dt} = f\left(\frac{3}{r F}\,I_{Ca} - k_{Ca}\,[Ca]\right)$$

The factor $3/r$ is the surface-to-volume ratio of the spherical cell, $F$ the Faraday
constant, $f$ the fraction of free (unbuffered) calcium, and $k_{Ca}$ the pump removal rate.
The small $f$ makes calcium the slow burst variable.

---

## Parameters (paper Table I; calcium removal $k_{Ca}=0.04\ \text{ms}^{-1}$ of Fig. 1b)

| Symbol | Field | Value | Unit |
|---|---|---|---|
| $C_m$ | `c_m` | 1.0 | µF/cm² |
| $\bar g_{Ca}$ | `g_ca` | 6.5 | mS/cm² |
| $\bar g_K$ | `g_k` | 12.0 | mS/cm² |
| $\bar g_{K(Ca)}$ | `g_kca` | 0.09 | mS/cm² |
| $g_L$ | `g_l` | 0.04 | mS/cm² |
| $E_{Ca}$ | `e_ca` | +100 | mV |
| $E_K$ | `e_k` | −75 | mV |
| $E_L$ | `e_l` | −40 | mV |
| $V'$ | `v_prime` | 50 | mV |
| $V^{*}$ | `v_star` | 30 | mV |
| $K_d$ | `k_dis` | 1.0 | µM |
| $r$ | `radius_cm` | 8.9×10⁻⁴ | cm |
| $F$ | `faraday` | 96487 | C/mol |
| $f$ | `f_ca` | 0.004 | — |
| $k_{Ca}$ | `k_ca` | 0.04 | ms⁻¹ |
| $T$ | `temp_celsius` | 20 | °C |

---

## Behaviour

With the published parameters the model bursts autonomously at zero applied current
(the glucose-stimulated regime). It reproduces the paper's figures:

- **Square-wave bursts** — fast ~12 mV spikes ride an active-phase plateau, separated by
  silent hyperpolarised phases; the membrane stays in the paper's −57 mV to roughly −20 mV
  band rather than producing large overshooting action potentials.
- **Slow calcium oscillation** — cytosolic calcium swings of order one micromolar
  (≈0.3–1.1 µM here) with a burst period of order ten to twenty seconds (Fig. 1, Fig. 2).
- **Calcium-activated potassium paces the burst** — removing $\bar g_{K(Ca)}$ abolishes the
  oscillation: nothing terminates the active phase, so the cell spikes continuously and
  calcium only climbs.

A non-zero `current` is an applied membrane current (for example the negative
pump-mimicking current of the paper's Na/K-pump extension).

## Implementation notes

The stiff system is integrated with guarded sub-steps; the default `dt` is sub-divided to
at most `_MAX_SUBSTEP`. State and parameters are validated each step (gates in [0, 1],
calcium non-negative, finite voltage within a wide divergence envelope), so a numerically
diverging trajectory raises rather than returning corrupted state. Spike detection registers
an upward crossing of `v_threshold` (−30 mV, between the spike trough ≈ −39 mV and the
plateau spike peak ≈ −25 mV).

The equations and parameters were transcribed from the original paper and cross-checked
against the Wolfram Demonstrations reference implementation; the two pieces that a naive
reduction drops are the cell radius in centimetres (so the calcium influx coefficient
$3/(rF)\approx0.035$) and the temperature factor $\phi\approx4.5$ multiplying the gate
kinetics.
