# ChayKeizerMinimalNeuron

**Module:** `sc_neurocore.neurons.models.chay_keizer_minimal`
**Reference:** Bertram, R., Marinelli, I., Fletcher, P.A., Satin, L.S. & Sherman, A.S.
(2023). Deconstructing the integrated oscillator model for pancreatic β-cells.
*Mathematical Biosciences* 365:109085, Table 1.
DOI [10.1016/j.mbs.2023.109015](https://doi.org/10.1016/j.mbs.2023.109015) —
the three-dimensional reduction of Chay & Keizer (1983).
**Family:** Bursting (conductance-based pancreatic beta-cell)
**State variables:** `v` (membrane potential), `n` (delayed-rectifier potassium
activation), `c` (free cytosolic calcium).

This is the reduced three-state version of the Chay–Keizer model. Taking the calcium-
current activation at its instantaneous steady state collapses the original five
variables to three while preserving the bursting mechanism: the fast `v`–`n` subsystem
generates spikes, and calcium — the slow variable — accumulates during the active phase,
activates a calcium-dependent potassium conductance that terminates the burst, and is then
pumped down through the silent phase. A constant ATP-sensitive potassium conductance sets
the excitability (the glucose handle in the full model).

The companion `ChayKeizerNeuron` is the literal five-dimensional 1983 model; this reduced
form is the one used in fast/slow analysis and as the kernel of the integrated oscillator
model.

---

## Equations

Currents use the convention $g\,(V - E_{rev})$ (outward positive), in the reference's
physical units (pS, fF, fA, mV, µM, ms).

$$C_m \frac{dV}{dt} = I_{app} - (I_{Ca} + I_K + I_{K(Ca)} + I_{K(ATP)})$$
$$\frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n}, \qquad
\frac{dc}{dt} = f_c\, J_{mem}$$

$$I_{Ca} = g_{Ca}\, m_\infty(V)\,(V - V_{Ca}), \qquad I_K = g_K\, n\,(V - V_K)$$
$$I_{K(Ca)} = g_{K(Ca)}\, \frac{c^{\,n_k}}{K_d^{\,n_k} + c^{\,n_k}}\,(V - V_K), \qquad
I_{K(ATP)} = g_{K(ATP)}\,(V - V_K)$$

$$m_\infty(V) = \frac{1}{1 + \exp((v_m - V)/s_m)}, \qquad
n_\infty(V) = \frac{1}{1 + \exp((v_n - V)/s_n)}$$

$$J_{mem} = -(\alpha\, I_{Ca} + k_{pmca}\, c)$$

---

## Parameters (reference Table 1)

| Symbol | Field | Value | Unit |
|---|---|---|---|
| $g_{Ca}$ | `g_ca` | 1000 | pS |
| $g_K$ | `g_k` | 2700 | pS |
| $g_{K(Ca)}$ | `g_kca` | 400 | pS |
| $g_{K(ATP)}$ | `g_katp` | 180 | pS |
| $C_m$ | `c_m` | 5300 | fF |
| $V_{Ca}$ | `e_ca` | 25 | mV |
| $V_K$ | `e_k` | −75 | mV |
| $v_m$ | `v_m` | −20 | mV |
| $v_n$ | `v_n` | −16 | mV |
| $s_m$ | `s_m` | 12 | mV |
| $s_n$ | `s_n` | 5 | mV |
| $n_k$ | `n_hill` | 3 | — |
| $K_d$ | `k_d` | 0.3 | µM |
| $f_c$ | `f_c` | 0.001 | — |
| $\alpha$ | `alpha` | 1.125×10⁻⁶ | µM·fA⁻¹·ms⁻¹ |
| $k_{pmca}$ | `k_pmca` | 0.045 | ms⁻¹ |
| $\tau_n$ | `tau_n` | 20 | ms |

---

## Behaviour

At zero applied current the cell bursts autonomously, reproducing the reference Fig. 1:
fast spikes ride an active-phase plateau near −20 mV with silent phases near −65 mV, while
cytosolic calcium traces a slow sawtooth. Removing $g_{K(Ca)}$ abolishes the oscillation —
nothing terminates the active phase, so the cell spikes continuously and calcium climbs.
A larger $g_{K(ATP)}$ hyperpolarises the cell toward silence (low glucose), a smaller one
toward continuous activity (high glucose).

A non-zero `current` is an applied membrane current in femtoamperes.

## Implementation notes

The system is integrated with guarded sub-steps; state and parameters are validated each
step (gate in [0, 1], calcium non-negative, finite voltage within a wide divergence
envelope). Spike detection registers an upward crossing of `v_threshold` (−35 mV, below the
plateau spike peak near −21 mV). Parameters were transcribed from the reference Table 1 and
the model verified to reproduce its Fig. 1 bursting.
