# Reduced Chay–Keizer (Bertram et al. 2023) — transcribed parameters

Transcribed from **Bertram, Marinelli, Fletcher, Satin & Sherman (2023), *Mathematical
Biosciences* 365:109085, Table 1**,
DOI [10.1016/j.mbs.2023.109015](https://doi.org/10.1016/j.mbs.2023.109015) (open access,
CC BY 4.0). These are the facts used to parameterise `ChayKeizerMinimalNeuron`, the
three-dimensional reduction of the Chay–Keizer model.

## State (three variables)

`v` membrane potential (mV) · `n` delayed-rectifier potassium activation · `c` free
cytosolic calcium (µM). Initial conditions: `v = −60 mV`, `n = 0`, `c = 0.1 µM`.

## Table 1 parameters

| Symbol | Field | Value | Unit |
|---|---|---|---|
| g_Ca | `g_ca` | 1000 | pS |
| g_K | `g_k` | 2700 | pS |
| g_K(Ca) | `g_kca` | 400 | pS |
| g_K(ATP) | `g_katp` | 180 | pS |
| C_m | `c_m` | 5300 | fF |
| f_c | `f_c` | 0.001 | — |
| α | `alpha` | 1.125×10⁻⁶ | µM·fA⁻¹·ms⁻¹ |
| k_pmca | `k_pmca` | 0.045 | ms⁻¹ |
| K_d | `k_d` | 0.3 | µM |
| V_Ca | `e_ca` | 25 | mV |
| V_K | `e_k` | −75 | mV |
| v_m | `v_m` | −20 | mV |
| v_n | `v_n` | −16 | mV |
| s_m | `s_m` | 12 | mV |
| s_n | `s_n` | 5 | mV |
| n_k (Hill) | `n_hill` | 3 | — |
| τ_n | `tau_n` | 20 | ms |

## Equations (reference Eqs 1–10)

Currents use g(V − E_rev) (outward positive):

- I_Ca = g_Ca · m∞(V) · (V − V_Ca)
- I_K = g_K · n · (V − V_K)
- I_K(Ca) = g_K(Ca) · c^n_k/(K_d^n_k + c^n_k) · (V − V_K)
- I_K(ATP) = g_K(ATP) · (V − V_K)
- m∞(V) = 1/(1 + exp((v_m − V)/s_m)),  n∞(V) = 1/(1 + exp((v_n − V)/s_n))

Dynamics:

- C_m dV/dt = I_app − (I_Ca + I_K + I_K(Ca) + I_K(ATP))
- dn/dt = (n∞(V) − n)/τ_n
- dc/dt = f_c · J_mem,  with J_mem = −(α · I_Ca + k_pmca · c)

## Verified behaviour

With the above the model bursts: fast spikes on an active-phase plateau (V band ≈ −66 to
−21 mV, matching Fig. 1A) with a slow cytosolic-calcium sawtooth — reproducing the
reference Fig. 1.
