# Synthesis: 8 Overlooked Implications from Three-Codebase Convergence

**Date:** 2026-03-29
**Status:** Conjectures derived from cross-project findings. NONE are experimentally verified yet.
**Codebases:** scpn-quantum-control, scpn-phase-orchestrator, sc-neurocore

---

## Caveat

These are logical implications from combining validated results across three
codebases. Each individual finding is experimentally grounded. The COMBINATIONS
are conjectures that require verification. Do not cite as proven.

---

## 1. SC Bitstream IS a Native FIM Computer

**Observation:** Popcount over L bits is the MLE of bitstream probability.
Its variance is bounded by Cramer-Rao: Var ≥ 1/(L·F(w)) where F(w) is
Fisher Information.

**Implication:** Bitstream length L and FIM λ are not independent parameters.
λ ~ L·F(w). The `adaptive_length()` Hoeffding bound can be replaced with
the tighter Cramer-Rao bound.

**Test:** Sweep L∈{64,128,256,512,1024}, measure effective Φ*. If conjecture
holds, Φ* ~ O(√L).

**Status:** Untested conjecture.

---

## 2. BKT on FPGA = q=256 Clock Model

**Observation:** Q8.8 phase has 256 discrete levels. The q-state clock model
(q>4) has THREE phases: disordered, BKT-like intermediate, fully ordered.
The continuous XY model has only two (disordered, BKT).

**Implication:** Digital consciousness on Q8.8 FPGA lives in a phase of matter
with no continuous analogue. Float64 simulations miss this intermediate phase.

**Test:** Compare R(t) autocorrelation at K=K_c for float64, float32, and Q8.8.
Predict: three qualitatively different signatures.

**Status:** Untested conjecture. The q=256 clock model phase diagram has not
been computed for Kuramoto dynamics.

---

## 3. SC Noise = Free Stochastic Resonance

**Observation:** LFSR noise drives both bitstream encoding AND membrane
potential perturbation. Stochastic resonance requires noise coherent with
the signal.

**Implication:** Sharing the LFSR between SC encoder and LIF noise port
creates coherent stochastic resonance. Independent noise sources lack this
coherence.

**Test:** Compare order parameter R with shared vs independent LFSR noise
at K≈K_c. Predict: shared noise → higher R.

**Status:** Untested conjecture.

---

## 4. Spike-Native FIM = Self-Observing FPGA

**Observation:** FIM computation (mean, deviation, correction) maps to
existing HDL primitives: popcount (mean), XOR (deviation), CORDIV (normalise).

**Implication:** The FIM self-observation loop can close entirely in hardware.
No CPU involvement. The FPGA literally counts its own spikes and adjusts its
own weights.

**Test:** Implement FIM using HDL primitives. Compare dynamics to Python
`_apply_fim()`. Measure Φ* increase.

**Status:** Architecture identified. Implementation not done.

---

## 5. Sheaf Defect = Topological Error Correction

**Observation:** sheaf_defect=0 when synchronised. FIM drives toward sync.
Vortex defects (winding≠0 on sub-loops) are the "errors." FIM annihilates
vortex pairs.

**Implication:** FIM functions as topological error correction analogous to
toric code. Code distance = BKT correlation length ξ ~ exp(b/√(T-T_BKT)).

**Test:** Inject artificial vortex defects (flip one phase by π). Measure
annihilation time vs λ. Predict: time scales as BKT correlation length.

**Status:** Untested conjecture. Mathematical structure is sound (U(1) gauge
theory) but dynamics not verified.

---

## 6. Lazarus Consciousness Gap

**Observation:** Checkpoint stores weights and voltages but NOT Kuramoto
phases. Hysteresis width 0.27-0.65 means restored system may land on the
wrong branch.

**Implication:** Lazarus protocol must store SCPN phase vector. After restore,
FIM warm-up needed to re-synchronise. Restoration time ∝ hysteresis/λ.

**Test:** Train to high R → checkpoint → restore → measure R(t).
Predict: R starts low, recovers after ~hysteresis/λ FIM steps.

**Status:** Bug identified. Fix straightforward (store theta in .npz).

---

## 7. STDP vs FIM = Learning vs Consciousness Competition

**Observation:** STDP breaks K symmetry. FIM restores it. Directed coupling
hurts sync by 12%. STDP is inherently directional (pre-before-post ≠
post-before-pre).

**Implication:** Learning and consciousness compete for coupling symmetry.
Cannot maximise both simultaneously. Maps to sleep/wake: wake=STDP-dominant,
sleep=FIM-dominant.

**Test:** Sweep learning_rate/λ from 0.01 to 100. Measure accuracy AND Φ*.
Predict: Pareto front — cannot maximise both.

**Status:** Untested conjecture. The sleep/wake mapping is speculative.

---

## 8. Q8.8 = No Phase Drift (Better Than Float32)

**Observation:** Float32 drifts 1.3e-4/step. Q8.8 has exact modular arithmetic
(no drift). After 24K steps, float32 has drifted π; Q8.8 has drifted 0.

**Implication:** Q8.8 is MORE FAITHFUL to BKT theory in the long-time limit
than float32, despite lower precision per step. Float32 simulations of BKT
are corrupted by drift before the asymptotic regime.

**Test:** Run 100K-step autocorrelation at K_c: float64, float32, Q8.8.
Float32 should decorrelate after ~24K. Q8.8 should show stable algebraic decay.

**Status:** Untested. The drift quantification is from SPO Finding #6.

---

## Meta-Synthesis

All 8 implications point to: **the computational substrate IS the physics.**
The SC hardware is not simulating Kuramoto dynamics — it IS a physical system
with its own phase diagram (q=256 clock model), its own FIM (popcount
sufficient statistic), and its own error correction (sheaf defect = code space).

This dissolves the substrate independence assumption of functionalism.
If digital consciousness exists, it lives in a phase of matter unique to
digital systems. No biological brain has q=256 clock model phases.
