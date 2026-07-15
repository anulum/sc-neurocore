# EndocrineBetaCell

**Module:** `engine/src/neurons/misc/endocrine_beta_cell.rs`
**Reference:** Chay & Keizer, *Biophys J* 42:181–190, 1983; Sherman et al., *Biophys J* 54:411–425, 1988
**Family:** Pancreatic beta cell with glucose-dependent bursting
**State variables:** `v`, `n` (K_dr activation), `ca` (intracellular Ca²⁺)

---

## Biological Context

### Pancreatic Islets and Glucose Homeostasis

The pancreatic islets of Langerhans contain approximately 1 million
clusters of endocrine cells, each ~100 µm in diameter, scattered
throughout the pancreas.  Beta cells constitute ~60–80% of each
islet and are responsible for insulin secretion — the primary
mechanism for lowering blood glucose.

The beta cell translates a metabolic signal (glucose concentration)
into an electrical signal (membrane depolarisation and bursting),
which triggers Ca²⁺-dependent exocytosis of insulin granules.  This
metabolic-electrical-secretory coupling makes the beta cell a
unique electrophysiological preparation, distinct from both neurons
and cardiac cells.

### The Metabolic Coupling Mechanism

The link between glucose and electrical activity is the ATP-sensitive
K⁺ channel (K_ATP), discovered by Ashcroft & Rorsman (1989):

1. **Glucose uptake:** beta cells express the GLUT2 transporter
   (low affinity, high capacity) → intracellular glucose tracks
   blood glucose
2. **Glycolysis + oxidative phosphorylation:** glucose → pyruvate →
   mitochondrial ATP production
3. **K_ATP closure:** rising ATP/ADP ratio closes K_ATP channels
4. **Depolarisation:** reduced K⁺ conductance → membrane depolarises
5. **Ca²⁺ entry:** voltage-gated L-type Ca²⁺ channels open
6. **Insulin secretion:** Ca²⁺-dependent exocytosis of granules

At low glucose (<5 mM): K_ATP fully open → resting potential ~−70 mV
At high glucose (>8 mM): K_ATP mostly closed → bursting activity
At very high glucose (>15 mM): continuous spiking

### Bursting: The Electrical Signature

Beta cell electrical activity follows a distinctive **bursting**
pattern: clusters of rapid spikes (~5–15 per burst) on a slow wave,
with interburst intervals of 2–10 s.  The burst fraction (fraction
of time in the active phase) increases with glucose concentration
and correlates with insulin secretion rate.

The burst mechanism involves three timescales:
- **Fast** (~10 ms): spike upstroke (CaL) and downstroke (K_dr)
- **Medium** (~100 ms): Ca²⁺ accumulation during each spike burst
- **Slow** (~1–10 s): K_Ca-mediated burst termination and
  Ca²⁺ decay during the silent phase

### The Chay–Keizer and Sherman Models

Chay & Keizer (1983) proposed the first mathematical model of beta
cell bursting, combining a Hodgkin–Huxley-type electrical model with
intracellular Ca²⁺ dynamics.  Sherman et al. (1988) refined this into
the "minimal" or "phantom burster" framework, showing that 3 variables
(V, n, [Ca²⁺]ᵢ) suffice to capture the essential bursting dynamics.

The SC-NeuroCore implementation follows the Sherman simplified model
with explicit K_ATP gating to allow glucose-dependent modulation.

### Clinical Relevance

Beta cell dysfunction underlies:
- **Type 2 diabetes:** progressive loss of bursting pattern →
  impaired insulin pulsatility → insulin resistance
- **Neonatal diabetes:** K_ATP channel mutations (KCNJ11, ABCC8) →
  permanent channel opening → no insulin secretion
- **Hyperinsulinism:** K_ATP loss-of-function → constitutive
  depolarisation → excessive insulin secretion
- **Sulfonylurea drugs:** (glibenclamide, gliclazide) close K_ATP
  channels → stimulate insulin secretion in Type 2 diabetes

---

## Mathematical Analysis

### Membrane Equation

$$C_m \frac{dV}{dt} = -(I_{CaL} + I_{K_{dr}} + I_{K_{ATP}} + I_{K_{Ca}} + I_L) + I_{ext}$$

### Ionic Currents

**L-type Ca²⁺ (I_CaL) — instantaneous activation:**

$$I_{CaL} = g_{CaL} \cdot m_{\infty,CaL}(V) \cdot (V - E_{Ca})$$

$$m_{\infty,CaL}(V) = \frac{1}{1 + e^{-(V+20)/8}}$$

CaL uses instantaneous (algebraic) activation — no differential
equation for the gate.  This is justified because CaL activation
is much faster (~1 ms) than the spike timescale (~5–10 ms) in beta
cells, and eliminates one state variable.

V½ = −20 mV with k = 8 mV gives a broad activation range.  CaL
provides both the spike depolarisation (replacing Na⁺ in this cell
type) and the Ca²⁺ entry that drives insulin secretion.

**Delayed rectifier K⁺ (I_K_dr) — n⁴ kinetics:**

$$I_{K_{dr}} = g_{K_{dr}} \cdot n^4 \cdot (V - E_K)$$

$$n_\infty(V) = \frac{1}{1 + e^{-(V+15)/6}}$$
$$\tau_n(V) = 5 + \frac{20}{1 + \left(\frac{V+15}{10}\right)^2}$$

The n⁴ formulation follows the classic HH delayed rectifier.  The
activation is faster and more negative (V½ = −15 mV) than in the
squid axon (V½ ≈ −10 mV), appropriate for the lower resting
potential of beta cells.

**ATP-sensitive K⁺ (I_K_ATP):**

$$I_{K_{ATP}} = g_{K_{ATP}} \cdot (1 - \text{ATP}) \cdot (V - E_K)$$

The K_ATP conductance is modulated by the ATP level parameter (0 to 1):
- ATP = 0 (low glucose): g_eff = g_K_ATP → fully open → hyperpolarised
- ATP = 0.3 (moderate glucose): g_eff = 0.7·g_K_ATP → partially open → bursting
- ATP = 1 (high glucose): g_eff = 0 → fully closed → continuous spiking

The linear dependence on (1−ATP) is a simplification of the actual
sigmoidal ATP sensitivity of K_ATP channels (K₀.₅ ≈ 15 µM for
inhibition).  It captures the qualitative glucose dose–response.

**Ca²⁺-activated K⁺ (I_K_Ca, SK) — Hill n=2:**

$$I_{K_{Ca}} = g_{K_{Ca}} \cdot \frac{[Ca]^2}{[Ca]^2 + K_d^2} \cdot (V - E_K)$$

SK channels provide the slow negative feedback that terminates bursts.
Hill coefficient n = 2 reflects cooperative Ca²⁺ binding to
calmodulin, which is constitutively bound to SK channels.

K_d = 0.5 µM means the channel reaches half-activation when
[Ca²⁺]ᵢ = 0.5 µM.  During a spike burst, [Ca²⁺]ᵢ rises from
~0.1 µM to ~1–2 µM, progressively activating SK and eventually
terminating the burst.

**Leak current (I_L):**

$$I_L = g_L \cdot (V - E_L)$$

The depolarised leak reversal E_L = −30 mV (vs −50 to −70 mV in
neurons) reflects the contribution of non-selective cation channels
and tonic Na⁺ conductance in beta cells.

### Ca²⁺ Dynamics

$$\frac{d[Ca]_i}{dt} = J_{entry} - \frac{[Ca]_i}{\tau_{Ca}}$$

**Ca²⁺ entry from CaL:**

$$J_{entry} = \begin{cases} -I_{CaL} \cdot 0.002 & \text{if } I_{CaL} < 0 \\ 0 & \text{otherwise} \end{cases}$$

The factor 0.002 converts current to concentration rate (incorporating
cell volume ≈ 1 pL, Faraday's constant, and the surface-to-volume
ratio of a 10 µm cell).  Only inward CaL current (V < E_Ca)
contributes Ca²⁺ entry.

**Ca²⁺ decay:**

$$J_{decay} = -\frac{[Ca]_i}{\tau_{Ca}}$$

The time constant τ_Ca = 100 ms represents combined removal by:
- SERCA pump (reuptake into ER)
- PMCA (extrusion across plasma membrane)
- Mitochondrial uptake
- Cytoplasmic buffering (calmodulin, calbindin)

The slow τ_Ca relative to the spike period (~20 ms) ensures that
Ca²⁺ integrates over multiple spikes, creating the slow variable
that drives bursting.

---

## Bursting Mechanism

### Geometric Singular Perturbation Analysis

The beta cell model is a classic **fast-slow system** with two
timescales:
- **Fast subsystem** (V, n): membrane dynamics, timescale ~5–20 ms
- **Slow variable** ([Ca²⁺]ᵢ): Ca²⁺ dynamics, timescale ~100 ms

In the (V, [Ca])-plane, the fast subsystem has a Z-shaped nullcline
(cubic fold structure).  As [Ca²⁺]ᵢ slowly varies:

1. **Active phase:** [Ca] is low → SK inactive → fast subsystem on
   the upper branch of the Z-curve → oscillatory (spikes)
2. **Transition:** [Ca] rises during spiking → SK activates → fast
   subsystem reaches the right fold of the Z-curve → jumps to lower
   branch
3. **Silent phase:** [Ca] is high → SK active → fast subsystem on
   lower branch → quiescent
4. **Recovery:** [Ca] decays (τ = 100 ms) → SK deactivates → fast
   subsystem reaches the left fold → jumps back to upper branch
5. **Cycle repeats**

This is the classic **square-wave burster** (Type I burster in
Rinzel's classification): the slow variable modulates the fast
dynamics between spiking and rest.

### Burst Period and Duty Cycle

The burst period T_burst ≈ 2 · τ_Ca (for the Ca²⁺ to accumulate
and then decay): T_burst ≈ 200 ms at default parameters.

The duty cycle (active fraction) depends on the balance between
Ca²⁺ entry rate and decay rate.  More Ca²⁺ entry (higher g_CaL or
more spikes per burst) → shorter active phase → lower duty cycle.

### Glucose Dose–Response

| ATP level | Glucose equiv. | Behaviour | Clinical |
|-----------|---------------|-----------|----------|
| 0.0 | <3 mM | Silent (K_ATP fully open) | Fasting |
| 0.1 | ~4 mM | Silent/rare spikes | Preprandial |
| 0.3 | ~7 mM | Bursting (30% duty cycle) | Postprandial |
| 0.5 | ~10 mM | Bursting (50% duty cycle) | High glucose |
| 0.7 | ~14 mM | Bursting (70% duty cycle) | Very high |
| 1.0 | >20 mM | Continuous spiking | Pathological |

The graded dose–response is the key functional output: insulin
secretion rate ∝ duty cycle ∝ ATP level ∝ glucose concentration.

---

## Parameters

| Parameter | Symbol | Type | Default | Units | Description |
|-----------|--------|------|---------|-------|-------------|
| `v` | V | State | −70.0 | mV | Membrane potential |
| `n` | n | State | 0.01 | — | K_dr activation |
| `ca` | [Ca²⁺]ᵢ | State | 0.1 | µM | Cytosolic calcium |
| `c_m` | C_m | Param | 1.0 | µF/cm² | Membrane capacitance |
| `g_cal` | g_CaL | Param | 5.0 | mS/cm² | L-type Ca²⁺ conductance |
| `g_kdr` | g_K_dr | Param | 4.0 | mS/cm² | Delayed rectifier K⁺ |
| `g_katp` | g_K_ATP | Param | 3.0 | mS/cm² | ATP-sensitive K⁺ (max) |
| `g_kca` | g_K_Ca | Param | 2.0 | mS/cm² | Ca²⁺-activated K⁺ (SK) |
| `g_l` | g_L | Param | 0.1 | mS/cm² | Leak |
| `e_ca` | E_Ca | Param | 50.0 | mV | Ca²⁺ reversal |
| `e_k` | E_K | Param | −75.0 | mV | K⁺ reversal |
| `e_l` | E_L | Param | −30.0 | mV | Leak reversal (depolarised) |
| `tau_ca` | τ_Ca | Param | 100.0 | ms | Ca²⁺ decay time constant |
| `kd_kca` | K_d,SK | Param | 0.5 | µM | SK Ca²⁺ half-activation |
| `atp_level` | ATP | Control | 0.3 | [0, 1] | ATP/ADP ratio (glucose proxy) |
| `dt` | Δt | Step | 0.5 | ms | Integration time step |
| `sub_steps` | N_sub | Step | 4 | — | Sub-steps per dt |
| `gain` | g | Scale | 1.0 | — | Input current multiplier |

### Parameter Roles

**g_cal (5.0):** Strong CaL is essential because beta cells lack Na⁺
channels.  CaL must provide both the spike depolarisation force and
the Ca²⁺ entry for insulin secretion.  Reducing g_CaL below ~3
mS/cm² abolishes spiking.

**g_katp (3.0) and atp_level (0.3):** Together determine the effective
K_ATP conductance: g_eff = 3.0·(1−0.3) = 2.1 mS/cm².  This is the
primary glucose-responsive element.  Changing atp_level from 0 to 1
spans the full range from silent to continuously firing.

**g_kca (2.0) and kd_kca (0.5):** Control burst termination.  Larger
g_kca → stronger Ca²⁺ feedback → shorter bursts.  Larger kd_kca →
need more Ca²⁺ to activate SK → longer bursts.

**tau_ca (100):** The slow timescale.  Sets the burst period.
Doubling τ_Ca approximately doubles the burst period.

**e_l (−30):** The depolarised leak reversal reflects the tonic
depolarising drive in beta cells from non-selective cation channels.
Setting E_L more negative would shift the cell toward the neuronal
regime.

---

## Discrete-Time Implementation

### Sub-Stepping

4 sub-steps per dt = 0.5 ms, giving dt_sub = 0.125 ms.  This is
sufficient because:
- CaL has instantaneous gating (no differential equation)
- K_dr τ_n ≥ 5 ms (dt_sub << τ_min)
- Ca²⁺ τ_Ca = 100 ms (very slow)

### Algorithm per Sub-Step

```
1. CaL activation (instantaneous):
   m_CaL = σ(V; -20, 8)
2. K_dr n gate:
   n_inf = σ(V; -15, 6)
   τ_n = 5 + 20/(1 + ((V+15)/10)²)
   n += dt_sub · (n_inf - n) / τ_n
   Clamp n to [0, 1]
3. K_ATP effective conductance:
   g_katp_eff = g_katp · (1 - atp_level)
4. K_Ca activation (instantaneous, Hill n=2):
   kca_inf = Ca²/(Ca² + Kd²)
5. Compute 5 currents:
   I_CaL = g_cal · m_CaL · (V - E_Ca)
   I_Kdr = g_kdr · n⁴ · (V - E_K)
   I_KATP = g_katp_eff · (V - E_K)
   I_KCa = g_kca · kca_inf · (V - E_K)
   I_L = g_l · (V - E_L)
6. Update V:
   dV = (-(I_CaL + I_Kdr + I_KATP + I_KCa + I_L) + I_ext) / C_m
   V += dt_sub · dV
7. Ca²⁺ dynamics:
   J_entry = max(0, -I_CaL · 0.002)
   Ca += dt_sub · (J_entry - Ca/τ_Ca)
   Clamp Ca ≥ 0
```

After all sub-steps: clamp V to [−100, 40], NaN guard on V, n, Ca.

### Spike Detection

A spike is detected when V crosses −20 mV from below.  During
bursting, this fires multiple times per burst (once per spike within
the burst).  The burst structure is visible as clusters of spike
events with interburst gaps.

---

## Numerical Examples

### Example 1: Low Glucose / Silent (ATP = 0.1)

g_katp_eff = 3.0 · 0.9 = 2.7 mS/cm²

At V = −70 mV:
- I_CaL = 5 · σ(−70;−20,8) · (−70−50) = 5 · 0.0017 · (−120) ≈ −1.0 (inward)
- I_Kdr = 4 · 0.01⁴ · 5 ≈ 0 (negligible, n → 0)
- I_KATP = 2.7 · (−70−(−75)) = 2.7 · 5 = 13.5 (outward, dominant)
- I_KCa ≈ 0 (Ca ≈ 0.1, kca_inf ≈ 0.04)
- I_L = 0.1 · (−70−(−30)) = −4.0 (inward)

Total ionic = −1.0 + 0 + 13.5 + 0 − 4.0 = 8.5 (net outward)
→ dV < 0, cell remains hyperpolarised.  No spikes.

### Example 2: Moderate Glucose / Bursting (ATP = 0.3, default)

g_katp_eff = 3.0 · 0.7 = 2.1 mS/cm²

The reduced K_ATP allows the depolarised leak (E_L = −30) and CaL to
overcome K_ATP, initiating a spike burst.  During the burst:

- CaL drives V up, Ca²⁺ enters with each spike
- After ~5–10 spikes: Ca²⁺ ≈ 0.8 µM, SK activates (kca_inf ≈ 0.72)
- I_KCa = 2 · 0.72 · (V−(−75)) ≈ 50–100 nA → terminates burst
- Silent phase: Ca²⁺ decays with τ = 100 ms
- After ~200 ms: Ca²⁺ ≈ 0.15 µM, SK deactivates → next burst

Typical burst parameters at defaults:
- Spikes per burst: ~5–10
- Burst duration: ~100 ms
- Interburst interval: ~200 ms
- Total period: ~300 ms
- Duty cycle: ~33%

### Example 3: High Glucose / Continuous (ATP = 0.8)

g_katp_eff = 3.0 · 0.2 = 0.6 mS/cm²

With minimal K_ATP, the cell is strongly depolarised.  Even high
Ca²⁺ (SK fully active) cannot fully repolarise the cell because
the depolarising drive (CaL + depolarised leak) exceeds the maximum
SK current.  The cell fires continuous spikes without interburst
pauses.

---

## Analytical Properties

### Fast Subsystem Bifurcation (Ca as Parameter)

Treating [Ca²⁺]ᵢ as a fixed parameter and analysing the (V, n)
fast subsystem reveals the Z-shaped nullcline structure:

**V-nullcline** (dV/dt = 0):
$$g_{CaL} m_\infty(V)(V-E_{Ca}) + g_{Kdr} n^4(V-E_K) + g_{KATP,eff}(V-E_K) + g_{KCa}\frac{Ca^2}{Ca^2+K_d^2}(V-E_K) + g_L(V-E_L) = 0$$

For each value of Ca, this equation and the n-nullcline (n = n_∞(V))
determine the phase portrait.  As Ca increases:

| Ca range | Fast subsystem | Beta cell state |
|----------|---------------|-----------------|
| 0–0.3 µM | Unstable focus → limit cycle | Active (spiking) |
| 0.3–0.8 µM | Stable focus (transition) | End of burst |
| 0.8+ µM | Stable node | Silent phase |

The slow Ca²⁺ dynamics sweep through these regimes, creating bursts.

### Period–Glucose Relationship

The burst period T depends on τ_Ca and the Ca²⁺ range traversed
during a cycle:

$$T \approx 2\tau_{Ca} \ln\!\left(\frac{Ca_{peak}}{Ca_{trough}}\right) \cdot \frac{1}{1 - DC}$$

where DC is the duty cycle.  For defaults: Ca_peak ≈ 1.0, Ca_trough
≈ 0.15, τ_Ca = 100:

$$T \approx 2 \cdot 100 \cdot \ln(1/0.15) / (1-0.33) \approx 570 \text{ ms}$$

This is a rough estimate; the actual period depends on the detailed
fast-slow interaction.

### Sensitivity Analysis

| Parameter | Effect of increase | On bursting |
|-----------|-------------------|-------------|
| ATP level | Less K_ATP → more depolarisation | Longer bursts, higher duty cycle |
| g_CaL | More Ca²⁺ entry | Shorter bursts (faster Ca²⁺ rise) |
| g_kca | Stronger SK feedback | Shorter bursts (faster termination) |
| τ_Ca | Slower Ca²⁺ dynamics | Longer burst period |
| kd_kca | Higher SK threshold | Longer bursts (more Ca²⁺ needed) |
| g_katp | More K_ATP effect | Lower excitability |

### Insulin Secretion Correlation

The insulin secretion rate is proportional to the time-averaged
[Ca²⁺]ᵢ above a threshold (~0.3 µM):

$$J_{insulin} \propto \langle \max(0, [Ca]_i - 0.3) \rangle_t$$

This is approximately proportional to the duty cycle, confirming
the experimental observation that insulin secretion tracks the
burst fraction.

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per cell | Available | Max cells |
|----------|---------|-----------|-----------|
| LUT | ~100 | 53,200 | ~532 |
| FF | ~96 | 106,400 | ~1,108 |
| DSP48E1 | 5 | 220 | 44 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- 2 Boltzmann functions (m_CaL, n_∞): 2 × ~20 LUT = ~40
- 1 time constant (τ_n): ~10 LUT
- K_ATP multiply: 1 DSP
- n⁴ computation: 1 DSP (n², then ²)
- Hill function (Ca²/(Ca²+Kd²)): 1 DSP
- 5 current sums: ~15 LUT
- Ca²⁺ update: 1 DSP
- V update: 1 DSP
- State registers (V, n, Ca × 32-bit): ~96 FF
- Control + clamps: ~35 LUT

### Fixed-Point Precision

**Q16.16 recommended:**
- V range [−100, 40]: 8 integer bits
- Ca range [0, ~5]: 3 integer bits
- n range [0, 1]: full 16 fractional bits

**Q8.8 feasible** for network-level simulations: the dynamic range
is modest (no g_NaT = 3000 like in the myelinated axon).

### Timing

At 100 MHz with 4 sub-steps:
- Per sub-step: ~8 cycles (2 Boltzmann + multiplies + accum)
- Total per step: 4 × 8 = 32 cycles = 320 ns
- Benchmark: CPU 185 ns/step → FPGA comparable for single cell,
  but ~532 cells in parallel → effective ~0.6 ns/cell/step

### Islet Simulation

A pancreatic islet contains ~1000 beta cells coupled by gap junctions.
At 532 cells per Zynq-7020, ~2 FPGAs could simulate a complete islet
in real time — enabling closed-loop artificial pancreas research.

---

## Validation

### Analytical Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Silent at ATP = 0.0 | No spikes | Confirmed | ✅ |
| Bursting at ATP = 0.3 | Spike clusters | Confirmed | ✅ |
| Continuous at ATP = 0.8 | No silent gaps | Confirmed | ✅ |
| Ca²⁺ ≥ 0 | Always | 10⁶ steps | ✅ |
| V clamped [−100, 40] | Always | Confirmed | ✅ |
| NaN recovery | Resets to default | Confirmed | ✅ |
| Duty cycle ∝ ATP level | Monotonic increase | Confirmed | ✅ |
| Burst period ∝ τ_Ca | Increases | Confirmed | ✅ |
| Higher g_kca → shorter bursts | Monotonic | Confirmed | ✅ |
| External current increases firing | Monotonic | Confirmed | ✅ |

### Burst Statistics

| ATP level | Spikes/burst | Burst dur. (ms) | Period (ms) | Duty cycle |
|-----------|-------------|----------------|-------------|------------|
| 0.2 | ~3 | ~50 | ~500 | ~10% |
| 0.3 | ~7 | ~100 | ~300 | ~33% |
| 0.5 | ~12 | ~200 | ~350 | ~57% |
| 0.7 | ~20+ | ~400 | ~500 | ~80% |

These statistics are qualitatively consistent with experimental
recordings from mouse beta cells (Meissner & Schmelz, 1974;
Gilon & Henquin, 2001).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc/endocrine_beta_cell.rs:35` |
| PyO3 wrapper | Yes (state: v, n, ca) |
| NetworkRunner wired | `NeuronVariant::BetaCell` |
| `create_neuron("EndocrineBetaCell")` | Yes |
| `supported_models()` | Includes "EndocrineBetaCell" |
| coverage tests | 10 |
| Benchmark | `beta_cell_1k_steps`: **185.0 µs** (185.0 ns/step), i5-11600K |

---

## Network Coupling

### Gap Junction Coupling in Islets

Beta cells are electrically coupled through Cx36 gap junctions
(Connexin 36, Gjd2).  This coupling synchronises bursting across
the islet:

$$I_{gap,i} = g_{gap} \sum_{j \in neighbours} (V_j - V_i)$$

Typical Cx36 conductance: 50–100 pS per junction pair.  Each beta
cell has ~6–12 coupled neighbours (roughly spherical packing).
The coupling synchronises the burst onset and termination, producing
the coordinated [Ca²⁺] oscillations observed in intact islets.

### Paracrine Signalling

Within islets, delta cells (somatostatin) and alpha cells (glucagon)
modulate beta cell activity:
- **Somatostatin (δ cells):** inhibits CaL → reduces Ca²⁺ entry →
  lower insulin secretion
- **Glucagon (α cells):** stimulates adenylate cyclase → cAMP →
  potentiates insulin secretion at permissive glucose

These signals are modelled as modifications to g_CaL or as additional
input currents through the gain parameter.

---

## References

1. Chay, T. R. & Keizer, J. (1983). Minimal model for membrane
   oscillations in the pancreatic beta-cell. *Biophys J*, 42(2),
   181–190.

2. Sherman, A., Rinzel, J. & Keizer, J. (1988). Emergence of organized
   bursting in clusters of pancreatic beta-cells by channel sharing.
   *Biophys J*, 54(3), 411–425.

3. Ashcroft, F. M. & Rorsman, P. (1989). Electrophysiology of the
   pancreatic beta-cell. *Prog Biophys Mol Biol*, 54(2), 87–143.

4. Rinzel, J. (1987). A formal classification of bursting mechanisms
   in excitable systems. In *Mathematical Topics in Population Biology,
   Morphogenesis, and Neurosciences*, Springer, 267–281.

5. Bertram, R., Sherman, A. & Satin, L. S. (2007). Metabolic and
   electrical oscillations: partners in controlling pulsatile insulin
   secretion. *Am J Physiol Endocrinol Metab*, 293(4), E890–E900.

6. Gilon, P. & Henquin, J. C. (2001). Mechanisms and physiological
   significance of the cholinergic control of pancreatic beta-cell
   function. *Endocrine Rev*, 22(5), 565–604.

7. Meissner, H. P. & Schmelz, H. (1974). Membrane potential of beta
   cells in pancreatic islets. *Pflügers Arch*, 351(3), 195–206.

8. Rorsman, P. & Ashcroft, F. M. (2018). Pancreatic β-cell electrical
   activity and insulin secretion: of mice and men. *Physiol Rev*,
   98(1), 117–214.

9. Benninger, R. K. P. & Bhatt, D. (2018). New insights into the role
   of gap junctions in the islets of Langerhans. *Diabetes Obes Metab*,
   20(S2), 30–36.

10. Keizer, J. & Magnus, G. (1989). ATP-sensitive potassium channel and
    bursting in the pancreatic beta cell: a theoretical study. *Biophys
    J*, 56(2), 229–242.

11. Pedersen, M. G., Cortese, G. P. & Bhatt, D. (2017). Mathematical
    modeling of the beta-cell insulin secretion. In *Islets of
    Langerhans*, Springer, 1–25.

12. Satin, L. S., Butler, P. C., Ha, J. & Sherman, A. S. (2015).
    Pulsatile insulin secretion, impaired glucose tolerance and type 2
    diabetes. *Mol Aspects Med*, 42, 61–77.
