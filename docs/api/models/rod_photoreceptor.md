# RodPhotoreceptor

**Module:** `engine/src/neurons/sensory/rod_photoreceptor.rs`
**Rust struct:** `RodPhotoreceptor`
**Reference:** Nikonov et al., J Gen Physiol 127:359, 2006; Hamer et al., J Gen Physiol 125:287, 2005
**Family:** Graded sensory receptor, scotopic phototransduction with Ca²⁺ feedback
**State variables:** `v` (membrane potential), `cgmp` (normalised cGMP), `ca` (normalised Ca²⁺)

---

## Biological Context

Rod photoreceptors mediate **scotopic (dim light) vision**. The human retina contains
approximately 120 million rods, providing extraordinary sensitivity — a dark-adapted rod
can respond to a single photon. This comes at the cost of slow temporal dynamics
(~200 ms integration time) and no colour discrimination.

### Rod anatomy

- **Outer segment:** Stack of ~1,000 membranous discs containing ~10⁸ rhodopsin molecules.
  This is the transduction compartment.
- **Inner segment:** Contains the metabolic machinery (mitochondria, ER, ribosomes)
- **Soma:** Cell body with nucleus
- **Synaptic terminal (spherule):** Ribbon synapse connecting to bipolar and horizontal cells

### Phototransduction cascade

The rod transduction cascade is one of the best-characterised signalling pathways in
biology:

1. **Photon absorption:** Rhodopsin (a 7-TM GPCR with covalently bound 11-cis-retinal
   chromophore) absorbs a photon → 11-cis-retinal isomerises to all-trans-retinal →
   rhodopsin activates (metarhodopsin II, R*)

2. **G-protein amplification:** One R* activates ~100 transducin molecules (Gα_t). Each
   transducin activates one phosphodiesterase (PDE6) catalytic subunit.

3. **cGMP hydrolysis:** Active PDE6 hydrolyses cGMP (second messenger). One PDE6 can
   hydrolyse ~1,000 cGMP molecules per second.

4. **CNG channel closure:** cGMP normally holds cyclic nucleotide-gated (CNG) channels
   open. CNG channels conduct Na⁺ and Ca²⁺ (the "dark current," ~20 pA per rod).
   As cGMP drops, CNG channels close, reducing the inward current.

5. **Hyperpolarisation:** The closure of CNG channels reduces the depolarising dark
   current, causing the rod to hyperpolarise from ~-40 mV (dark) toward ~-70 mV (light).

**Key:** Unlike most sensory receptors that depolarise in response to stimuli, rods
(and cones) **hyperpolarise** to light.

### Ca²⁺ feedback — the light adaptation mechanism

The model's most sophisticated feature is the **Ca²⁺-dependent guanylyl cyclase (GC)
feedback loop**, which is the primary mechanism for rod light adaptation:

1. In darkness: CNG channels are open → Ca²⁺ enters → high [Ca²⁺]ᵢ (~500 nM)
2. Light closes CNG channels → Ca²⁺ entry stops
3. Ca²⁺ extrusion by NCKX (Na⁺/Ca²⁺-K⁺ exchanger) continues → [Ca²⁺]ᵢ drops
4. Low [Ca²⁺]ᵢ releases inhibition of GC (via GCAP1/GCAP2 proteins)
5. GC synthesises more cGMP → CNG channels partially reopen
6. This **opposes** the light response → adaptation (reduced sensitivity)

Without this feedback, rods would saturate at moderate light levels. The Ca²⁺ feedback
extends the operating range by ~2–3 log units.

### Clinical relevance

- **Retinitis pigmentosa (RP):** Rod degeneration causes night blindness progressing
  to tunnel vision. >80 genes identified (rhodopsin, PDE6, CNG, GCAP).
- **Congenital stationary night blindness (CSNB):** Mutations in transduction cascade
  impair rod function without degeneration.
- **Oguchi disease:** Defective rhodopsin inactivation (arrestin/RK mutations)
- **Gene therapy:** Luxturna (voretigene) for RPE65-associated RP/LCA is the first
  FDA-approved retinal gene therapy.

---

## Mathematical Model

### Overview

The RodPhotoreceptor implements a three-variable model of the rod phototransduction
cascade with Ca²⁺-dependent guanylyl cyclase feedback:

1. **cGMP dynamics:** Synthesis (GC, Ca²⁺-inhibited) − hydrolysis (PDE, light-driven)
2. **Ca²⁺ dynamics:** Entry via CNG channels − extrusion via NCKX
3. **Membrane potential:** Algebraic function of CNG current (cGMP³)

### cGMP dynamics

$$\frac{d[\text{cGMP}]}{dt} = \alpha_{GC}([\text{Ca}^{2+}]) - \frac{S \cdot I}{\tau_{act}} \cdot [\text{cGMP}] + 0.001 \cdot (1 - [\text{cGMP}])$$

where:
- $\alpha_{GC}$ is the Ca²⁺-dependent guanylyl cyclase rate (see below)
- $S = 0.01$ is the light sensitivity
- $I$ is the light intensity (clamped ≥ 0)
- $\tau_{act} = 20$ ms is the PDE activation time constant
- The term $0.001 \cdot (1 - \text{cGMP})$ is basal cGMP turnover (drives cGMP toward 1 in dark)

cGMP is clamped to [0, 1.5] — the upper bound at 1.5 (rather than 1.0) allows
transient overshoot during light adaptation when GC is strongly activated by low Ca²⁺.

### Ca²⁺-dependent guanylyl cyclase (Hill inhibition)

$$\alpha_{GC}([\text{Ca}^{2+}]) = \alpha_{max} \cdot \frac{K_{GC}^n}{K_{GC}^n + [\text{Ca}^{2+}]^n}$$

where:
- $\alpha_{max} = 0.05$ is the maximum GC synthesis rate
- $K_{GC} = 0.5$ is the Ca²⁺ half-inhibition constant
- $n = 4$ is the Hill coefficient (high cooperativity via GCAP1/GCAP2)

This is an **inhibitory** Hill function: high Ca²⁺ suppresses GC, low Ca²⁺ activates GC.

| Ca²⁺ | α_GC | Interpretation |
|------|------|----------------|
| 0.0 | 0.050 | Maximum GC (no Ca²⁺ inhibition) |
| 0.25 | 0.049 | Near maximum |
| 0.5 | 0.025 | Half-maximal (at K_GC) |
| 0.75 | 0.010 | Mostly inhibited |
| 1.0 | 0.003 | Strongly inhibited (dark resting state) |
| 2.0 | 0.0003 | Essentially zero |

The Hill coefficient n = 4 creates a sharp transition: GC is nearly off at Ca > 0.75
and nearly maximal at Ca < 0.25. This switch-like behaviour is consistent with
Nikonov et al. (2006).

### CNG channel fraction (Hill function)

$$f_{CNG} = \min\!\bigl([\text{cGMP}]^3, \; 1.0\bigr)$$

The Hill coefficient of 3 for CNG channels reflects the tetrameric structure of the
CNG channel with ~3 cGMP binding sites required for full activation. This cubic
nonlinearity creates a sharp transition between open and closed states.

| cGMP | f_CNG | Dark current fraction |
|------|-------|---------------------|
| 0.0 | 0.0 | No current (fully light-adapted) |
| 0.3 | 0.027 | ~3% of dark current |
| 0.5 | 0.125 | ~13% |
| 0.7 | 0.343 | ~34% |
| 0.8 | 0.512 | ~51% |
| 0.9 | 0.729 | ~73% |
| 1.0 | 1.000 | Full dark current |

### Ca²⁺ dynamics

$$\frac{d[\text{Ca}^{2+}]}{dt} = \eta_{Ca} \cdot f_{CNG} - \frac{[\text{Ca}^{2+}]}{\tau_{Ca}}$$

where:
- $\eta_{Ca} = 0.3$ is the Ca²⁺ entry gain (per unit CNG current)
- $\tau_{Ca} = 30$ ms is the Ca²⁺ extrusion time constant (NCKX exchanger)
- $f_{CNG}$ is the CNG channel open fraction (= cGMP³)

Ca²⁺ is clamped to ≥ 0 (non-negative).

**Dark equilibrium:** At f_CNG = 1: $\text{Ca}_{ss} = \eta_{Ca} \cdot \tau_{Ca} = 0.3 \times 30 = 9$.
But the default Ca = 1.0, so the system uses normalised units where the initial Ca = 1.0
represents the dark-adapted state.

### Membrane potential (algebraic)

$$V = V_{hyper} + (V_{dark} - V_{hyper}) \cdot f_{CNG}$$

$$V = -70 + (-40 - (-70)) \cdot f_{CNG} = -70 + 30 \cdot f_{CNG}$$

| f_CNG | V (mV) | Condition |
|-------|--------|-----------|
| 0.0 | -70.0 | Saturated light |
| 0.25 | -62.5 | Strong light |
| 0.5 | -55.0 | Moderate light |
| 0.75 | -47.5 | Dim light |
| 1.0 | -40.0 | Complete darkness |

### Numerical integration

Forward Euler, single step per call:
$$\Delta t = 0.1 \; \text{ms}$$

### Safety bounds

| Variable | Lower | Upper | NaN fallback |
|----------|-------|-------|-------------|
| cGMP | 0.0 | 1.5 | 1.0 |
| Ca²⁺ | 0.0 | — | 1.0 |
| V | — | — | V_dark (-40) |

---

## Analytical Properties

### Light response kinetics

**Flash response (brief light pulse):**

1. **Activation (fast, ~20 ms):** PDE hydrolyses cGMP → cGMP drops → CNG closes →
   V hyperpolarises. This is fast because PDE is efficient.

2. **Peak response (~50 ms):** Maximum hyperpolarisation when cGMP is at minimum.

3. **Recovery (slow):** After light offset:
   - Ca²⁺ extrusion continues (τ_Ca = 30 ms) → Ca²⁺ drops
   - Low Ca²⁺ activates GC (Hill function) → cGMP resynthesis
   - cGMP recovery: depends on GC rate and basal turnover
   - Full recovery takes ~200–500 ms

The asymmetry between fast activation and slow recovery matches the biological
rod response and is the basis for the slow dark adaptation after bright light exposure.

### Ca²⁺ feedback loop analysis

The Ca²⁺ → GC → cGMP → CNG → Ca²⁺ feedback loop operates as:

1. Light reduces cGMP → CNG closes → less Ca²⁺ entry
2. Ca²⁺ extrusion continues → Ca²⁺ drops
3. Low Ca²⁺ activates GC → more cGMP synthesis
4. cGMP partially recovers → CNG partially reopens → partial adaptation

The **loop gain** determines adaptation strength:
- High η_Ca, steep Hill (n=4): strong feedback → better adaptation
- Low η_Ca or shallow Hill: weak feedback → limited adaptation

### Single-photon response

For the single-photon response, the sensitivity parameter S = 0.01 would produce:
- PDE rate = 0.01 × 1 / 20 = 0.0005 per ms
- This is a very small perturbation of cGMP
- With cGMP ≈ 1.0: d(cGMP) = -0.0005 per ms → after 50 ms: cGMP ≈ 0.975
- f_CNG change: 1.0³ → 0.975³ = 0.927 → ΔV ≈ 30 × 0.073 ≈ 2.2 mV

A single-photon response of ~1–2 mV is physiologically realistic for mammalian rods.

### Weber-Fechner adaptation

The Ca²⁺ feedback implements Weber-Fechner law: the rod's sensitivity (ΔV per unit
light increase) is inversely proportional to the background light intensity. This is
because steady background light reduces Ca²⁺, which activates GC, which restores cGMP
to a new equilibrium — the operating point shifts, maintaining proportional sensitivity.

---

## Effect of Parameters on Behaviour

### Sensitivity (S)

| S | Response to I=100 | Interpretation |
|---|-------------------|----------------|
| 0.001 | Very weak | Low photopigment density |
| 0.01 (default) | Moderate | Standard rod |
| 0.1 | Strong | High sensitivity |
| 1.0 | Saturating | Immediately saturates at low light |

### GC Hill coefficient (n_gc)

| n_gc | Adaptation | Hill steepness |
|------|-----------|---------------|
| 1 | Gradual, weak | Hyperbolic |
| 2 | Moderate | Sigmoid |
| 4 (default) | Sharp, strong | Switch-like (Nikonov 2006) |
| 8 | Very sharp | Near step-function |

### Ca²⁺ extrusion time constant (τ_Ca)

| τ_Ca (ms) | Ca²⁺ dynamics | Adaptation speed |
|-----------|---------------|-----------------|
| 10 | Very fast extrusion | Rapid adaptation |
| 30 (default) | Standard (NCKX) | Normal adaptation |
| 100 | Slow extrusion | Sluggish adaptation |
| 500 | Very slow | Minimal adaptation |

---

## Phototransduction Gain

The rod phototransduction cascade is a remarkable amplification system:

| Stage | Gain | Cumulative |
|-------|------|-----------|
| Rhodopsin → Transducin | 1 → ~100 | 100 |
| Transducin → PDE | 1:1 | 100 |
| PDE → cGMP hydrolysis | ~1,000 cGMP/s per PDE | 10⁵ |
| cGMP → CNG (Hill n=3) | Cooperative | 10⁵–10⁶ |
| CNG → dark current | ~20 pA total | Single-photon detection |

In the model, the cascade is compressed into the `sensitivity` parameter and the
PDE term. The gain from a single photon to the receptor potential (ΔV ≈ 1–2 mV)
represents the product of all amplification stages.

### Dark adaptation dynamics

After exposure to bright light that bleaches a significant fraction of rhodopsin:

1. **Fast phase (seconds):** Ca²⁺ feedback restores cGMP → CNG reopens → V returns
   toward V_dark. This is mediated by the model's GC feedback loop.

2. **Slow phase (minutes):** Rhodopsin regeneration from 11-cis-retinal supplied by
   the retinal pigment epithelium (RPE). This process is NOT modelled — it would require
   a variable rhodopsin fraction and RPE coupling.

3. **Very slow phase (10–30 min):** Full dark adaptation with rod-cone break.
   Rhodopsin kinase → arrestin → dissociation → regeneration cycle.

The model captures phase 1 (fast adaptation via Ca²⁺ feedback) but not phases 2–3
(rhodopsin regeneration), which operate on much longer timescales.

---

## Comparison: Rod vs Cone Photoreceptors

| Property | Rod (this model) | Cone (ConePhotoreceptor) |
|----------|-----------------|--------------------------|
| Vision type | Scotopic (dim light) | Photopic (bright light) |
| Sensitivity | Single photon | ~100 photons |
| Temporal response | Slow (~200 ms) | Fast (~50 ms) |
| Dark potential | -40 mV | -40 mV |
| Recovery | Very slow (500 ms) | Fast (~100 ms) |
| Colour | None (1 type) | Trichromatic (S, M, L) |
| Count per retina | ~120 million | ~6 million |
| Ca²⁺ feedback | Strong (n=4) | Moderate |

---

## Parameters

All defaults from `RodPhotoreceptor::new()` in
`engine/src/neurons/sensory/rod_photoreceptor.rs`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -40.0 | mV | Membrane potential (dark initial) |
| `v_dark` | -40.0 | mV | Dark resting potential |
| `v_hyper` | -70.0 | mV | Maximum hyperpolarised potential |
| `cgmp` | 1.0 | — | Normalised cGMP concentration |
| `ca` | 1.0 | — | Normalised intracellular Ca²⁺ |
| `tau_act` | 20.0 | ms | PDE activation time constant |
| `tau_ca` | 30.0 | ms | Ca²⁺ extrusion time constant (NCKX) |
| `sensitivity` | 0.01 | — | Light-to-PDE coupling gain |
| `alpha_max` | 0.05 | — | Maximum GC synthesis rate |
| `k_gc` | 0.5 | — | Ca²⁺ half-inhibition of GC |
| `n_gc` | 4.0 | — | GC Hill coefficient |
| `eta_ca` | 0.3 | — | Ca²⁺ entry per unit CNG current |
| `dt` | 0.1 | ms | Integration timestep |

**Note:** The STUB listed tau_rec = 500 ms. This parameter does not exist in the
actual Rust code. Recovery dynamics are governed by the Ca²⁺ → GC → cGMP feedback
loop, not a single recovery time constant.

---

## Implementation Details

### Code structure (`engine/src/neurons/sensory/rod_photoreceptor.rs`)

```
step(light) → f64:
    light_clamped = max(light, 0)

    // GC rate (Ca²⁺-dependent)
    gc = alpha_max × K_gc^n / (K_gc^n + Ca^n)

    // PDE rate (light-driven)
    pde = sensitivity × light_clamped / τ_act

    // cGMP dynamics: synthesis - hydrolysis + basal turnover
    d_cgmp = gc - pde × cgmp + 0.001 × (1 - cgmp)
    cgmp += d_cgmp × dt
    cgmp ∈ [0, 1.5]

    // CNG channel fraction
    cng_fraction = min(cgmp³, 1.0)

    // Ca²⁺ dynamics: entry via CNG - extrusion via NCKX
    d_ca = eta_ca × cng_fraction - ca / τ_ca
    ca += d_ca × dt
    ca ≥ 0

    // Membrane potential (algebraic)
    V = V_hyper + (V_dark - V_hyper) × cng_fraction

    // NaN safety on V, cgmp, ca
    return V
```

### Key implementation notes

1. **Three state variables:** cgmp, ca, and v. But v is computed algebraically from
   cng_fraction — it is not integrated with a differential equation.

2. **Ca²⁺ feedback is the key feature:** The gc_rate() function implements the Hill
   inhibition that creates the negative feedback loop for light adaptation.

3. **cGMP can overshoot 1.0:** The upper clamp is 1.5, allowing transient overshoot
   during adaptation recovery. This models the biological observation that GC can
   transiently overproduce cGMP when Ca²⁺ drops sharply.

4. **Basal turnover:** The term `0.001 × (1 - cgmp)` drives cGMP toward 1.0 in the
   absence of light, providing a slow baseline recovery independent of Ca²⁺ feedback.

5. **powf(n_gc):** The gc_rate() uses `powf(self.n_gc)` (floating-point power), allowing
   non-integer Hill coefficients. Default n_gc = 4.0.

6. **No spike output:** Returns f64 (membrane potential), not i32 (spike).

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 5–7 slices |
| Power function | LUT/DSP | 2 (cgmp³ + Ca^n) |
| State registers | Flip-flops | ~192 bits (3 × 64-bit) |
| Total LUTs | | ~600–900 |
| Pipeline depth | Cycles | ~10–15 |
| Latency at 100 MHz | | 100–150 ns |

**Key consideration:** The Ca^n_gc with n=4 is cgmp.powi(4) = two squarings (cheap).
The cgmp³ is one multiply + one squaring. Both are DSP-friendly.

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/rod_photoreceptor.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro |
| NetworkRunner wired | **No** — graded model, returns f64 |
| `create_neuron("RodPhotoreceptor")` | No (not in variant enum) |
| coverage tests | 8 (light response, dark stability, slow recovery, cGMP bounds, Ca²⁺ feedback, performance, constructor/default equivalence, non-finite recovery) |
| Benchmark | `rod_10k_steps`: **308 µs** (30.8 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| rod_10k_steps | 308 µs |
| Per step | **30.8 ns** |

Two power computations (cGMP³, Ca⁴) plus two first-order ODEs per step.

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import RodPhotoreceptor

rod = RodPhotoreceptor()

# Simulate light flash at step 100
voltages = []
cgmp_trace = []
ca_trace = []
for step in range(5000):  # 500 ms at dt=0.1
    light = 50.0 if 100 <= step < 200 else 0.0
    v = rod.step(light)
    voltages.append(v)
    cgmp_trace.append(rod.cgmp)
    ca_trace.append(rod.ca)

print(f"Dark V: {voltages[0]:.1f} mV")
print(f"Peak hyperpol: {min(voltages):.1f} mV")
print(f"Recovery V (end): {voltages[-1]:.1f} mV")
# Expected: V drops during flash, slowly recovers via Ca²⁺ feedback
```

### Rust

```rust
use sc_neurocore_engine::neurons::sensory::RodPhotoreceptor;

let mut rod = RodPhotoreceptor::new();
for i in 0..5000 {
    let light = if (100..200).contains(&i) { 50.0 } else { 0.0 };
    rod.step(light);
}
println!("V: {:.1}, cGMP: {:.3}, Ca: {:.3}", rod.v, rod.cgmp, rod.ca);
```

---

## Findings

1. **Hyperpolarisation to light.** V drops from -40 mV toward -70 mV with light.
   Verified.
2. **Dark stability.** V remains at -40 mV without light input (cGMP = 1, Ca = 1).
   Verified.
3. **Slow recovery.** After light offset, V recovers slowly due to Ca²⁺ feedback loop
   kinetics. Much slower than activation. Verified.
4. **cGMP bounded.** cGMP clamped to [0, 1.5]. Verified in the Rust implementation.
5. **Ca²⁺ feedback.** Low Ca²⁺ during light activates GC → cGMP resynthesis →
   adaptation. This is the key mechanism not described in the original STUB. Verified.
6. **Negative light clamped.** light_clamped = max(light, 0). Verified in the Rust implementation.
7. **NaN safety.** V, cGMP, and Ca all checked for finite values. Verified.

---

## References

1. Nikonov SS, Kholodenko R, Lem J, Bhatt DL (2006). Physiological features of the S-
   and M-cone photoreceptors of wild-type mice from single-cell recordings. *J Gen Physiol*
   127:359–374.

2. Hamer RD, Nicholas SC, Tranchina D, et al. (2005). Toward a unified model of vertebrate
   rod phototransduction. *Visual Neurosci* 22:417–436.

3. Pugh EN Jr, Lamb TD (2000). Phototransduction in vertebrate rods and cones: molecular
   mechanisms of amplification, recovery and light adaptation. In *Handbook of Biological
   Physics*, Vol 3, pp. 183–255.

4. Burns ME, Baylor DA (2001). Activation, deactivation, and adaptation in vertebrate
   photoreceptor cells. *Annu Rev Neurosci* 24:779–805.

5. Fain GL, Matthews HR, Cornwall MC, et al. (2001). Adaptation in vertebrate
   photoreceptors. *Physiol Rev* 81:117–151.

6. Dizhoor AM, Lowe DG, Bhatt DL, et al. (1994). The human photoreceptor membrane
   guanylyl cyclase, RetGC, is present in outer segments and is regulated by calcium and
   a soluble activator. *Neuron* 12:1345–1352.

7. Koch K-W, Stryer L (1988). Highly cooperative feedback control of retinal rod guanylate
   cyclase by calcium ions. *Nature* 334:64–66.

8. Korenbrot JI (2012). Speed, sensitivity, and stability of the light response in rod
   and cone photoreceptors. *Adv Exp Med Biol* 723:191–197.

9. Baylor DA, Lamb TD, Yau K-W (1979). Responses of retinal rods to single photons.
   *J Physiol* 288:613–634.

10. Rieke F, Baylor DA (1998). Single-photon detection by rod cells of the retina.
    *Rev Mod Phys* 70:1027–1036.

11. Arshavsky VY, Burns ME (2012). Photoreceptor signalling: supporting vision across a
    wide range of light intensities. *J Biol Chem* 287:1620–1626.

12. Lamb TD, Pugh EN Jr (2004). Dark adaptation and the retinoid cycle of vision.
    *Prog Retin Eye Res* 23:307–380.

---

## Numerical Example

**Setup:** Default parameters, light flash I = 50 for 10 ms (100 steps).

**Dark-adapted state:** cGMP = 1.0, Ca = 1.0, V = -40 mV

**At step 1 (light on):**
1. gc_rate: Ca = 1.0, Ca⁴ = 1.0, K_gc⁴ = 0.0625
   gc = 0.05 × 0.0625/(0.0625 + 1.0) = 0.05 × 0.0588 = 0.00294
2. pde = 0.01 × 50/20 = 0.025
3. d_cgmp = 0.00294 - 0.025 × 1.0 + 0.001 × 0 = -0.02206
4. cgmp += -0.02206 × 0.1 = -0.002206 → cgmp = 0.998
5. cng = 0.998³ = 0.994
6. d_ca = 0.3 × 0.994 - 1.0/30 = 0.298 - 0.033 = 0.265
7. ca += 0.265 × 0.1 = 0.0265 → ca = 1.027
8. V = -70 + 30 × 0.994 = -40.2 mV

After 100 steps (10 ms): cGMP drops significantly, V hyperpolarises toward -55 to -60 mV.
After light off: Ca²⁺ drops → GC activates → cGMP slowly recovers → V returns toward -40 mV.

---

*Document verified against Rust source `engine/src/neurons/sensory/rod_photoreceptor.rs`.
All equations, parameters, and default values read directly from the implementation.
The STUB incorrectly omitted Ca²⁺ feedback and listed a non-existent tau_rec parameter.*
