# Chay–Keizer 1983 — transcribed parameters

Transcribed from **Chay, T.R. & Keizer, J. (1983), *Biophys. J.* 42:181–190**,
DOI [10.1016/S0006-3495(83)84384-7](https://doi.org/10.1016/S0006-3495(83)84384-7),
and cross-checked against the Wolfram Demonstrations reference implementation. These are
the facts used to parameterise `ChayKeizerNeuron`; the full paper PDF is in `../pdfs/`
(git-ignored, copyright-bound).

## State (five variables)

`v` membrane potential (mV) · `m`, `h` calcium-current activation / inactivation gates ·
`n` delayed-rectifier potassium activation · `ca` free cytosolic calcium (µM).
Initial conditions (Fig. 1): `v = −54.774 mV`, `ca = 0.8 µM`, gates at steady state.

## Table I parameters

| Symbol | Field | Value | Unit |
|---|---|---|---|
| C_m | `c_m` | 1.0 | µF/cm² |
| ḡ_K,Ca | `g_kca` | 0.09 | mS/cm² |
| ḡ_K,HH | `g_k` | 12.0 | mS/cm² |
| ḡ_Ca,HH | `g_ca` | 6.5 | mS/cm² |
| g_L | `g_l` | 0.04 | mS/cm² |
| V_K | `e_k` | −75 | mV |
| V_Ca | `e_ca` | +100 | mV |
| V_L | `e_l` | −40 | mV |
| V* (n shift) | `v_star` | 30 | mV |
| V′ (m,h shift) | `v_prime` | 50 | mV |
| K_dis | `k_dis` | 1.0 | µM |
| r (cell radius) | `radius_cm` | 8.9×10⁻⁴ | cm |
| f (free-Ca fraction) | `f_ca` | 0.004 | — |
| T | `temp_celsius` | 20 | °C |
| F (Faraday) | `faraday` | 96487 | C/mol |
| k_Ca (Fig. 1b burst) | `k_ca` | 0.04 | ms⁻¹ |
| ℓ (Hill coefficient) | — | 1 | — |

Figure 1 varies k_Ca for the burst pattern (a: 0.02, b: 0.04, c: 0.06 ms⁻¹); we use the
Fig. 1b value 0.04. Hill coefficient ℓ = 1 (Fig. 5 uses ℓ = 4).

## Equations

Currents use the convention g(E_rev − V) (inward positive):

- I_Ca = ḡ_Ca,HH · m³h · (V_Ca − V)
- I_K = ḡ_K,HH · n⁴ · (V_K − V)
- I_K(Ca) = ḡ_K,Ca · [Ca]/([Ca] + K_dis) · (V_K − V)   (Eq. 1, ℓ = 1)
- I_L = g_L · (V_L − V)

Membrane (Eq. 8) — note the factor 2 on the divalent calcium current:

> C_m dV/dt = I_app + 2 I_Ca + I_K + I_K(Ca) + I_L

Gates use the Hodgkin–Huxley 1952 rate functions at the shifted potential
U_m = V + V′ (for m, h) and U_n = V + V* (for n), each scaled by the temperature factor
φ = 3^((T − 6.3)/10):

- α_m = −0.1(U_m − 25)/(exp(−(U_m − 25)/10) − 1),  β_m = 4 exp(−U_m/18)
- α_h = 0.07 exp(−U_m/20),  β_h = 1/(exp(−(U_m − 30)/10) + 1)
- α_n = −0.01(U_n − 10)/(exp(−(U_n − 10)/10) − 1),  β_n = 0.125 exp(−U_n/80)
- ẋ = φ(α_x(1 − x) − β_x x) for x ∈ {m, h, n}

Calcium (Eq. 9):

> f⁻¹ d[Ca]/dt = (3 / (r F)) I_Ca − k_Ca [Ca]

The factor 3/r is the spherical surface-to-volume ratio.

## Two conventions a naive reduction drops

1. **Cell radius in centimetres** — `radius_cm = 8.9×10⁻⁴ cm` (not 8.9 µm), giving the
   calcium influx coefficient 3/(rF) ≈ 0.035 µM·ms⁻¹ per µA·cm⁻². Using the µm value
   collapses calcium to zero (tonic firing, no bursts).
2. **Temperature factor** — φ = 3^((20 − 6.3)/10) ≈ 4.5 multiplies the gate kinetics.
   Omitting it leaves the gates too slow and the cell settles to a subthreshold oscillation
   instead of square-wave bursting.

## Verified behaviour

With the above, the model bursts: fast ~12 mV spikes on an active-phase plateau, cytosolic
calcium oscillating ≈0.3–1.1 µM, burst period of order ten to twenty seconds — reproducing
the paper's Figs. 1 and 2.
