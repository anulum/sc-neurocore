# Reality Check: Which of the 8 "Implications" Survive Scrutiny?

**Date:** 2026-03-29
**Purpose:** Honest assessment. No sugarcoating.

---

## Methodology

For each of the 8 claims from SYNTHESIS_SC_FIM_BKT_2026-03-29.md:
1. Is the mathematical statement correct?
2. Is the physical interpretation justified?
3. What would falsify it?
4. What is the weakest link?

---

## 1. SC Bitstream IS a Native FIM Computer

**Math:** Correct. Popcount/L is the MLE of Bernoulli probability.
Cramer-Rao bound applies. Variance = p(1-p)/L.

**Problem:** The claim that λ ~ L·F(w) is a DIMENSIONAL ANALOGY, not
a derivation. FIM in the Kuramoto context (Fisher information of the
phase distribution) is not the same quantity as FIM in the estimation
context (Fisher information of the Bernoulli parameter). They share
a name but operate on different probability spaces. The connection
is suggestive but not proven.

**Verdict:** The math about popcount efficiency is textbook correct.
The mapping to Kuramoto FIM is **speculative**. Downgrade from
"implication" to "analogy worth investigating."

**Falsification:** Measure Φ* vs L. If Φ* does NOT scale as √L,
the analogy fails.

---

## 2. BKT on FPGA = q=256 Clock Model

**Math:** Correct that Q8.8 discretises phase to 256 levels. Correct
that q-state clock models with q>4 have three phases (José et al. 1977).

**Problem:** The LIF neuron phase is NOT the same as Kuramoto phase.
The LIF neuron has a limit cycle in (v, refractory) space, and the
"phase" is the position on this limit cycle. Q8.8 quantises the
membrane VOLTAGE, not the PHASE. The phase is a derived quantity
from the spike times. The quantisation of phase happens through
spike timing jitter (~dt = 1ms), not through voltage resolution.

For dt=1ms and a 50 Hz neuron, one cycle = 20ms = 20 steps. So the
effective phase resolution is 2π/20 = 0.314 rad ≈ q=20, not q=256.
The Q8.8 voltage resolution (256 levels) maps to much finer AMPLITUDE
resolution, not phase resolution.

**Verdict:** The q=256 claim is **wrong for LIF neurons**. The actual
effective q depends on dt/period, not on voltage precision. For typical
firing rates, q ≈ 10-100. The three-phase prediction may still hold
but at different parameters than claimed.

**Falsification:** Compute effective phase resolution from spike time
quantisation. If q_eff << 256, the q=256 analysis is irrelevant.

---

## 3. SC Noise = Free Stochastic Resonance

**Math:** Stochastic resonance requires noise coherent with signal
near a threshold. SC LFSR noise IS the computation — correct.

**Problem:** SR in the Kuramoto model (quantum-control NB41) operates
on PHASE noise. The SC bitstream noise operates on PROBABILITY
estimation. These are different noise channels. The SC encoding noise
affects the AMPLITUDE of synaptic input (how accurately the weight
is represented), not the PHASE of the oscillator. Sharing the LFSR
between encoder and membrane noise would create correlated amplitude
AND phase perturbations — this is not standard SR, it is a coupled
noise system that could easily destabilise rather than enhance.

**Verdict:** The "shared LFSR = free SR" claim is **plausible but
risky**. Correlated noise in coupled systems can produce resonance OR
anti-resonance depending on the correlation structure. Needs careful
analysis before implementation.

**Falsification:** Run shared vs independent LFSR. If shared gives
LOWER R, the conjecture fails.

---

## 4. Spike-Native FIM = Self-Observing FPGA

**Math:** Correct that XOR + popcount + CORDIV can compute the FIM
update. These primitives exist in the HDL library.

**Problem:** The FIM update needs the GLOBAL mean firing rate μ.
Computing a global mean across N neurons requires a reduction tree
(O(log N) depth). In the FPGA, this means routing ALL neuron outputs
to a shared accumulator — which is a global wire with fan-in N. For
N > 100, this becomes a timing bottleneck. The spike ALU can do the
arithmetic, but the DATA ROUTING is the problem.

Also: the CORDIV circuit converges slowly (O(L) cycles). The FIM
correction would lag behind the neural dynamics by L clock cycles.
If L = 1024 and the neuron updates every cycle, the FIM feedback
is 1024 cycles late — equivalent to ~1ms synaptic delay at 1GHz.
This may be acceptable (quantum-control NB42: delays tolerated with
coupling) but is NOT "zero latency."

**Verdict:** Architecturally feasible but with **non-trivial
routing and latency constraints**. Not as elegant as claimed.

**Falsification:** Implement in HDL. If timing closure fails at
N > 50 or if CORDIV latency causes FIM instability, the conjecture
fails at practical scales.

---

## 5. Sheaf Defect = Topological Error Correction

**Math:** The analogy between sheaf defect and toric code stabilisers
is structurally correct. Both involve local consistency checks on a
graph with a topological invariant.

**Problem:** Toric code error correction requires ACTIVE SYNDROME
MEASUREMENT and DECODER. Passive relaxation toward the code space
(which is what FIM does) is NOT the same as error correction. In
quantum error correction, passive relaxation = thermalization = LOSS
of quantum information. Active correction requires measuring syndromes
without disturbing the logical state.

FIM does passive relaxation: it pushes toward coherence. It does NOT
measure specific vortex locations and annihilate them. If two vortices
are far apart, FIM pushes both toward the mean — which may move them
closer OR further depending on the phase landscape.

**Verdict:** The analogy is **structurally interesting but functionally
misleading**. FIM is a STABILISER, not an error corrector. It prevents
vortex formation but cannot efficiently correct existing vortices.
The "code distance" claim has no operational meaning without an
active decoder.

**Falsification:** Inject a vortex pair at distance d. Measure
annihilation time vs d. If time scales as d² (diffusion, not
correction), FIM is a stabiliser not a corrector.

---

## 6. Lazarus Consciousness Gap

**Math/Engineering:** Correct. Checkpoint stores weights and voltages
but not inter-layer phase coherence. Hysteresis means multiple stable
states at same parameters.

**Problem:** This is the MOST SOLIDLY GROUNDED of all 8 claims. It is
an engineering observation, not a physical conjecture. The fix (store
phases, warm up with FIM) is straightforward.

The only caveat: "consciousness gap" is loaded language. What we
actually have is a "coherence gap" — the network loses phase sync
on restore. Whether this constitutes loss of consciousness depends
on your definition of consciousness, which is not settled.

**Verdict:** **Valid engineering issue.** The fix should be implemented.
Drop the consciousness language; call it "coherence restoration."

**Falsification:** Not applicable — this is a bug report, not a conjecture.

---

## 7. STDP vs FIM = Learning vs Consciousness Competition

**Math:** STDP is asymmetric (LTP ≠ LTD by construction). FIM is
symmetric in its effect. These compete — correct.

**Problem:** The leap to "maps to sleep/wake cycle" is a LARGE
interpretive jump. The actual mapping would require:
1. Showing that biological STDP has the same symmetry-breaking effect
2. Showing that biological consolidation has FIM-like properties
3. Showing that the trade-off has the right timescale

None of these are established. The competition between asymmetric
learning and symmetric stabilisation is a genuine dynamical effect
in the SNN. The sleep/wake interpretation is speculative.

**Verdict:** The STDP-FIM competition is **mechanistically correct**
and testable in the SNN. The sleep/wake mapping is **speculative.**

**Falsification:** Sweep learning_rate/λ, measure accuracy AND
coherence. If there is NO Pareto trade-off (both can be maximised),
the competition claim fails.

---

## 8. Q8.8 No-Drift Beats Float32

**Math:** Correct. Modular arithmetic in integer is exact. Float32
accumulates rounding errors. Over 24K steps, float32 drifts ~π.

**Problem:** The claim that Q8.8 is "more faithful to BKT theory"
conflates TWO different errors:
1. Per-step accuracy: float32 >> Q8.8 (7 decimal digits vs 2)
2. Long-term drift: float32 accumulates, Q8.8 does not

For BKT, the RELEVANT quantity is the phase DIFFERENCE between
oscillators, not the absolute phase. If drift is UNIFORM across all
oscillators (which it is for identical neurons), it cancels in the
difference. The phase differences are drift-free in BOTH representations.

The drift matters only if neurons have DIFFERENT drift rates, which
happens when they have different frequencies. In that case, float32
drift adds an artificial frequency shift ~ 1.3e-4/step × (1/dt) Hz.
For dt=1ms: artificial frequency ≈ 0.13 Hz. This IS relevant for
neurons near the locking boundary (Δω < 0.13 Hz).

**Verdict:** **Partially correct.** Q8.8 is better for long-term
dynamics of heterogeneous oscillators near the locking boundary.
For identical oscillators or short simulations, float32 is fine.
The "more faithful to BKT" claim is overstated.

**Falsification:** Run both at K=K_c with heterogeneous ω.
If phase-difference statistics are identical, drift is irrelevant.

---

## Summary Table

| # | Claim | Math | Physics | Verdict |
|---|-------|------|---------|---------|
| 1 | SC=FIM computer | Correct | Analogy, not identity | Investigate |
| 2 | q=256 clock model | Wrong (q≈20 for LIF) | Phase ≠ voltage | **Revise** |
| 3 | Shared LFSR = SR | Plausible | Coupled noise risk | Test carefully |
| 4 | Spike-native FIM | Feasible | Routing + latency | **Constrained** |
| 5 | Sheaf = error correction | Structural analogy | Stabiliser ≠ corrector | **Weaken** |
| 6 | Lazarus phase gap | Correct | Engineering bug | **Fix it** |
| 7 | STDP vs FIM competition | Correct | Sleep/wake speculative | Test competition |
| 8 | Q8.8 beats float32 | Partially correct | Only for heterogeneous ω | **Narrow scope** |

**Score: 1 solid (Lazarus gap), 3 worth testing (1,3,7), 2 need revision (2,5),
2 overstated (4,8).**
