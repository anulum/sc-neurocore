# Model Fidelity & Polyglot-Completion Status

This page states, per model, which neuron models are **polyglot-complete** — the strictest
completion bar SC-NeuroCore tracks. It is the single tracked, public answer to "which models can be
considered full-fidelity / completed / implemented across the whole compute chain?".

## The bar — what "polyglot-complete" means

A model is **polyglot-complete** only when every one of its four acceleration-language kernels
implements the *real* model dynamics and each is proven against the Python reference:

1. **Real dynamics in all four accel lanes** — `accel/rust/safety`, `accel/go`, `accel/julia`,
   `accel/mojo` — the actual model (integrator and sub-stepping included), not a stub, a pasted
   comment body, a dummy return, or fake placeholder dynamics.
2. **Proven parity** — an executed test asserting each lane reproduces the Python golden. Where the
   right-hand side is exact arithmetic the parity is **bit-for-bit**; where gating uses
   transcendentals (`exp`/`tanh`/…) the declared stable observable is either event count or the
   complete continuous trajectory inside an explicit numerical tolerance.
3. **Honest benchmark** — a runnable, committed per-backend benchmark producing a real number (no
   fabricated figures).

This bar is **stricter** than, and must not be confused with, two weaker properties that most of the
catalogue already has:

- **Faithful Python reference** — every catalogued model in
  [the neuron model catalogue](neuron_models.md) has a real, tested Python implementation. That is
  the ground truth, but it is *not* polyglot-completion.
- **Real Rust engine acceleration** — many models are accelerated by the `engine/` Rust+PyO3 crate
  (the genuine performance path today). That is real and benchmarked, but it is the *engine* lane,
  not the four-language `accel/` chain this page scores.

## Legend

| Mark | Meaning |
|---|---|
| ✅ | Real dynamics, executed parity test passes |
| 🔶 | Real dynamics but the lane is not yet in its final executable form (a maintained parity note, or a strengthening pass still open) |
| ⬜ | Stub / fake / unverified |
| — | Not applicable for this model |

## Polyglot-complete models

Every lane below carries the real model dynamics with an executed Python-parity proof. Every Mojo
surface in this promoted set is now executable; no lane below is a fake stub or documentation-only
parity note.

| Model | Rust safety | Go | Julia | Mojo | Parity basis (golden) | Landed |
|---|---|---|---|---|---|---|
| Wang-Buzsaki | ✅ | ✅ | ✅ | ✅ executable | spike count — 3 AP @ I=10, 20 macro steps (Gauss-Seidel) | `0ebf4ea88` |
| FitzHugh-Nagumo | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 1 AP @ I=10/100 steps, 5-spike train @ I=0.5/2000 steps (RK4, exact RHS) | `729e0a2ea` |
| Morris-Lecar | ✅ | ✅ | ✅ | ✅ executable | spike count — 0/3/5 @ I=0/50/100 over 2000 steps (RK4, tanh/cosh) | `bc46a0fb5` |
| Connor-Stevens | ✅ | ✅ | ✅ | ✅ shared-lib | spike count — 0/2/9 @ I=0/10/20 over 100 macro steps (candidate-first RK4 with 100 sub-steps, exp); Mojo C ABI preserves every event and the six-state trace within `2e-6` over the enrolled envelope | `this commit` |
| Hodgkin-Huxley | ✅ | ✅ | ✅ | ✅ shared-lib | spike count — 0/6/9 @ I=0/10/20 over 100 macro steps (gate-first baseline-Euler with 100 sub-steps, exp); Mojo C ABI preserves every event and the four-state trace within `2e-9` over the enrolled envelope | `this commit` |
| AdEx | ✅ | ✅ | ✅ | ✅ shared-lib | accel event count — 0/4/12 @ I=0/200/500 over 1000 baseline-Euler steps; all compiled traces stay within `5e-12` of Python and transport the complete maintained numeric contract except the factory-default Rust engine boundary; existing Q16.16 RTL co-simulation retains its declared two-percent event-count envelope at I=1000 over 500 steps | `this commit` |
| ExpIF | ✅ | ✅ | ✅ | ✅ shared-lib | accel event count — 0/0/2 @ I=0/5/20 over 1000 candidate-first RK4 steps; every compiled lane preserves the events and stays within `5e-8` of Python, with the factory-default Rust engine boundary stated separately; hand/TOML/JSON and Q32.32 RTL preserve the enrolled event counts, and the generated depth-4 Z3 job passes | `this commit` |
| Lapicque | ✅ | ✅ | ✅ | ✅ shared-lib | accel event count — 0/0/71/200/500 @ I=0/0.5/2/5/20 over 1,000 exact constant-current RC steps; all compiled events are exact and traces stay within `2e-15` of Python, with the factory-default Rust engine boundary stated separately; hand/TOML/JSON events are exact and state error stays within `2e-15`, Q16.16 RTL preserves the complete 0/83/500 event vectors at I=0.333/2.3/20.25 with voltage error below 0.04, and the generated depth-20 Z3 job passes | `this commit` |
| Perfect Integrator | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel traces and event counts — 0/32/66/200/250/500/1000 @ I=0/0.333/0.7/2/3/5/20 over 1,000 candidate-first Euler steps; the Rust engine retains its factory-default boundary while Julia/Go/Mojo carry the complete numeric contract; hand/schema/Q8.8 RTL are event-exact with 66 events at I=0.7 over 1,000 steps, the quantisation boundary at I=0.333 is declared as 32/32/31, and the generated depth-20 Z3 job passes | `this commit` |
| Quadratic IF | ✅ | ✅ | ✅ | ✅ shared-lib | exact accel events — 0/2/3/6/11/26/100/250 @ I=0/0.333/0.5/1/2/5/20/50 over 1,000 exact Riccati-flow steps, with every compiled trace within `2e-12` of Python; the Rust safety module is exercised independently and the engine retains its factory-default boundary; hand/TOML/JSON events remain exact with state error below `0.006`, both schema formats reset inclusively at configured `v_peak`, Q16.16 RTL preserves the enrolled cycle-level event vectors with voltage error below `0.011`, the I=0.1 timing boundary is declared, and the generated depth-20 Z3 job passes | `this commit` |
| Theta | ✅ | ✅ | ✅ | ✅ shared-lib | exact accel events — 0/0/0/1/2/2/3/5/7/14/23 @ I=-1/-0.5/0/0.1/0.333/0.5/1/2/5/20/50 over 1,000 tangent-half-angle exact-flow steps, with every compiled trace within `2e-12` circular phase error; the Rust safety module is exercised independently and the engine retains its factory-default boundary; paired Euler schemas preserve the enrolled counts, generated Q16.16 RTL is count-exact across the vector with circular phase error below `0.17` rad at the declared moderate regimes, the I=1 one-cycle timing displacement is explicit, and the generated depth-6 Z3 job passes | `this commit` |
| DPI | ✅ | ✅ | ✅ | ✅ shared-lib | exact accel events — 0/0/0/0/1/3/6/11/21 @ I=-0.1/0/1/2/3/5/10/20/50 over 1,000 simultaneous-Euler steps of the coupled Indiveri-Stefanini-Chicca (2010) current-domain equations; compiled states remain within `5e-13`, the Rust safety module carries the complete 18-field contract and the engine retains its factory-default boundary; hand/TOML/JSON states remain within `1e-13`, generated Q16.16 RTL preserves 13 events at I=5 over 5,000 steps with declared state/timing envelopes, and the generated depth-4 Z3 job passes | `this commit` |
| COBA LIF | ✅ | ✅ | ✅ | ✅ shared-lib | exact 3,077-event parity over the controlled 200,000-step complete non-default contract; Rust/Julia/Go traces are exact and Mojo stays within `7.11e-15`; hand/TOML/JSON preserve the complete four-state RK4 trace and six enrolled events exactly, generated Q24.24 RTL preserves all six event indices with voltage/conductance/timer errors below `1e-5`/`5e-6`/`3e-6`/`2e-6`, the independent Brette-2007 DOI trace matches every feature within `1e-12`, and the generated depth-4 Z3 reset-safety job passes | `this commit` |
| Escape Rate | ✅ | ✅ | ✅ | ✅ shared-lib | seeded exact 29-event parity over the configured 4,096-step complete contract: Rust/Julia/Go voltage traces are exact, Mojo stays within `2e-14`, and every lane finishes at RNG state 45,999; the independent five-seed/full-period artifact pins event hashes, rate, and geometric-ISI statistics; hand/TOML/JSON preserve private RNG and failure atomicity; generated Q24.24 RTL reproduces the complete 65,535-bit event stream with 14,496 events and final seed `0xACE1`; the depth-4 Z3 safety job passes; the controlled 200,000-step benchmark records 1,523 exact events and final RNG 46,746 in all five lanes | `this commit` |
| Poisson | ✅ | ✅ | ✅ | ✅ shared-lib | seeded exact 918-event parity over the configured 4,096-bin rate/dt/RNG contract, with every runtime finishing at RNG state 45,999; the independent exhaustive artifact pins the 65,535-state event hash, 14,496-event rate, and geometric-ISI statistics; hand/TOML/JSON preserve private RNG and failure atomicity; registered and folded Q24.24 RTL reproduce the complete event stream, threshold, probability word, and final seed `0xACE1`; the depth-4 Z3 safety job passes; the source-hashed 200,000-bin benchmark records 44,256 exact events and final RNG 46,746 in all five lanes | `this commit` |
| IQIF | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact signed-integer parity for the pinned Wu et al. source tutorial: all five runtimes, the independent source recurrence, and paired TOML/JSON schemas reproduce 26 events and every one of 400 states; registered and folded Q32.0 RTL preserve the complete event/state vector with two signed Q0.3 shifts; the depth-4 Z3 safety job passes; the source-hashed 200,000-step benchmark records 13,333 events, final state 165, and trajectory SHA-256 `b5c84ffb…a4f4` in every lane | `this commit` |
| McCulloch-Pitts | ✅ | ✅ | ✅ | ✅ shared-lib | McCulloch and Pitts' 1943 all-or-none rule: a positive active-excitatory-afferent count threshold with absolute veto by any active inhibitory afferent, without later real-weight substitution or fake internal state; Python, Rust engine+safety, Julia, Go, Mojo, paired stateless TOML/JSON schemas, and the independent primary-paper truth table are bit-exact; registered and folded signed-Q32.0 RTL preserve every enrolled row using `-1` only as the inhibition sentinel; the depth-4 Z3 safety job passes; the source/binary-bound 200,000-row benchmark emits 102,273 events and trace SHA-256 `52a05b62…aee4` with zero mismatch in every lane | `this commit` |
| Sigmoid Rate | ✅ | ✅ | ✅ | ✅ shared-lib | complete configurable exact-relaxation rate traces over 200,000 steps: Python, Rust, Julia and Go are byte-identical; Mojo remains within `3.08e-14`, below the declared `5e-12` float tolerance; reset preserves `tau`, `beta`, `theta` and `dt`; paired schemas preserve the varied hand trajectory within `5e-12`, and generated Q32.32 RTL keeps the 256-step rate trace within `0.016` through public outputs while remaining event-silent; the source/binary-bound five-backend benchmark is local non-exclusive evidence, and no formal, synthesis, timing, device, or PPA claim is made | `this commit` |
| Threshold-linear Rate | ✅ | ✅ | ✅ | ✅ shared-lib | complete configurable algebraic rate traces over 200,000 evaluations: Python, Rust, Julia, Go, and Mojo are bit-exact for `r=gain*max(0,I-theta)`; below-threshold, equality, and above-threshold branches are executed; reset preserves `theta` and `gain`; paired schemas match the hand transfer exactly, and generated Q16.16 RTL preserves all 193 public rate words from I=-4 through 8 in 1/16 increments while remaining event-silent; the source/binary-bound benchmark is local non-exclusive evidence, and no formal, synthesis, timing, device, or PPA claim is made | `this commit` |
| Wilson-Cowan | ✅ | ✅ | ✅ | ✅ shared-lib | normalised coupled E/I population reduction with shifted sigmoid and candidate-first RK4; complete configurable 100,000-step Rust/Julia/Go trajectories and final rates remain within `1e-9` of Python and Mojo remains within `1e-8`; reset preserves all dynamics parameters; native failures are atomic; paired schemas match the hand trajectory within `1e-15`, and generated Q32.32 RTL keeps both public rates within `0.021` across the 96-sample mixed-drive trace while remaining event-silent; availability/refractory factors and independent inhibitory drive are explicitly outside scope; continuous rates are not spikes, and no formal, synthesis, timing, device, or PPA claim is made | `this commit` |
| Jansen–Rit | ✅ | ✅ | ✅ | ✅ shared-lib | Jansen and Rit 1995 equation-(6) neural mass with the published `C2*S(C1*y0)` and `C4*S(C3*y0)` wiring; all five runtimes return seven complete traces plus six final states, with Rust/Julia/Go within `1e-11` of Python and Mojo within `1e-8`; the independent 256-step DOI/Brian2-pinned trace is byte-exact, paired schemas match the hand model, and generated Q32.32 RTL preserves the enrolled trace within declared potential/derivative envelopes; the 0.1 ms Euler step follows the pinned implementation rather than the continuous paper equations; the EEG proxy is continuous and higher silicon rungs are not claimed | `this commit` |
| Montbrió–Pazó–Roxin | ✅ | ✅ | ✅ | ✅ shared-lib | Montbrió, Pazó, and Roxin 2015 dimensionless equations (12a–b), restored through `R=tau*r` and `t'=t/tau` and exposed through the legacy compatibility class `ErmentroutKopellPopulation`; all five runtimes return both complete state traces and final states, with Rust/Julia/Go within `1e-12` of Python and Mojo within `1e-10`; the independent 256-step DOI trace is byte-exact, paired schemas match the hand model, generated Q32.32 RTL preserves the enrolled trajectory within `2e-6`, and a depth-4 catalogue job proves bounded reset and event-silence safety only; the Euler step is implementation scope, firing rate is continuous, and formal equivalence or higher silicon rungs are not claimed | `this commit` |
| Wong-Wang | ✅ | ✅ | ✅ | ✅ shared-lib | Wong and Wang 2006 Appendix two-choice reduction with simultaneous explicit-Euler NMDA gating and AMPA Ornstein-Uhlenbeck current states; all five runtimes consume the same explicit Gaussian samples and return six complete traces plus four final states, with Rust/Julia/Go within `1e-12` of Python and Mojo within `1e-9`; the independent 256-step DOI trace is byte-exact, paired schemas match the hand model, and generated Q32.32 RTL preserves the enrolled trace within declared state/rate envelopes; the paper's 0.1 ms timestep is used and the author-code 0.5 ms discrepancy is recorded; rates are continuous and higher silicon rungs are not claimed | `this commit` |
| McKean | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 0/1/7 @ I=0/0.2/0.5 over 20000 steps (RK4, exact piecewise-linear RHS) | `059871140` |
| Hindmarsh-Rose | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/26/52 @ I=0/3/5 over 2000 steps; co-sim exact hand/TOML/JSON/Q16.16 counts — 0/0/26/40/52 @ I=0/2/3/4/5 over 2000 RK4 steps; declared Q16.16 +1 crossing boundary over 5000 steps at I=2 through I=5; formal Q8.8 BMC depth 4 | `this commit` |
| FitzHugh-Rinzel | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/1/8 @ I=0/0.3/0.5; co-sim exact spike count — hand/schema/Q16.16 RTL 8 @ I=0.5 (3000 RK4 steps, cubic RHS) | `498376221` |
| Pernarowski | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 7/17/27 @ I=0 over 2000/5000/8000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 17 @ I=-0.1/0/0.1/0.2 (5000 RK4 steps, autonomous cubic burster); formal Q8.8 reset-spike BMC depth 4 | `c384ac0cd` |
| Terman-Wang | ✅ | ✅ | ✅ | ✅ shared-lib | accel spike count — 0/1/3 @ I=-1/0/0.5 over 8000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 0/1/3 at the same operating points (RK4, cubic + tanh gate); formal Q8.8 reset-spike BMC depth 4 | `ce04dd6f9` |
| Wilson-HR | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/1/4 @ I=0/2/10 over 5000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 0/1/4 at the same operating points (RK4, polynomial RHS, hard voltage reset preserving recovery); formal Q8.8 reset-spike BMC depth 4 | `4d9810e2f` |
| Rulkov map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo ULP-bounded, 0/4/34 @ I=0/0.1/0.5 over 2000 iterations; co-sim: hand/TOML/JSON exact and Q16.16 RTL ten-event short-window trajectory within 0.001 state error @ I=1.5/30 iterations; Yosys Q16.16 H2 synthesis + formal Q8.8 BMC depth 4 | `7d5889e10` |
| GLIF | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/54/95 @ I=0/30/50 over 1000 steps; co-sim exact hand/TOML/JSON/Q16.16 spike counts — 0/0/23/54/86/95 @ I=0/15/22/30/45/50 (four-state RK4, candidate-first adaptive reset); formal Q8.8 BMC depth 6 | `ecd799d58` |
| Mihalas-Niebur | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/142/333 @ I=0/2/5 over 1000 steps; co-sim exact hand/schema/Q16.16 RTL — 0/0/0/31/60/87/131/157/207/256 @ I=0/0.5/1/1.5/2/2.5/3.5/4/5/6 over 1000 RK4 steps; explicit I=3 boundary 111/111/112; formal Q8.8 BMC depth 3 | `081dd569c` |
| Medvedev map | ✅ | ✅ | ✅ | ✅ shared-lib | Source-derived slow-calcium first-return map: Rust/Julia/Go bit-exact and Mojo bounded to `5e-13`, with exact 750-event parity at I=2 over 1000 iterations; hand/TOML/JSON exact and Q16.16 RTL preserves the complete 75-event vector at I=2 over 100 iterations with maximum u error below 0.007813; DOI feature trace + Q16.16 depth-4 Z3 BMC | `this commit` |
| Cazelles map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo per-step ULP-bounded, 5/182/204 @ I=0/0.5/1.0 over 1000 iterations; co-sim: hand/TOML/JSON exact and Q16.16 RTL event-exact at I=0.5/1.0/2.0 over 30 iterations with state error below 0.0004; I=0.05 is excluded; formal Q8.8 BMC depth 4 | `22110c66d` |
| Chialvo map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: all compiled lanes ULP-bounded to the source recurrence and event-count exact at I=-0.05/0/0.01/0.05/0.1/1.0 over 1000 iterations (0/26/30/0/0/1); pinned 500,000-iteration benchmark records 12,935 events in every lane; co-sim: hand/TOML/JSON exact and Q16.16 event counts 0/2/3/0/1 at I=-0.05/0/0.01/0.1/1.0 over 100 iterations, with stable-point x/y errors below 0.055/0.093 and oscillatory timing explicitly excluded; formal Q8.8 BMC depth 4 | `this commit` |
| Courbage-Nekorkin map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo per-step ULP-bounded, 157/193/168 @ I=-0.3/0/0.3 over 1000 iterations; co-sim: hand/TOML/JSON exact, Q16.16 event-exact at I=-0.3/0/0.3 over bounded 30/20/30-iteration windows, and Q32.32 event-exact at all three inputs over 30 iterations with state error below 0.00003; autonomous Q16.16 30-iteration trace is an explicit 4/6-event boundary; formal Q8.8 BMC depth 4 | `63826b513` |
| Izhikevich 2007 | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 0/3/14 @ I=0/100/400 over 2000 steps (RK4, quadratic v-nullcline, spike reset v→c / u+=d; Mojo ULP-bounded but the per-spike reset re-synchronises the trace so its counts always match) | `75b32d935` |
| Ibarz-Tanaka map | ✅ | ✅ | ✅ | ✅ shared-lib | Ibarz et al. (2007), Eqs. 2–3: source four-branch fast map plus simultaneous slow update; accel reset events 9/33/195 @ I=0/0.2/1 over 1000 iterations, Rust/Julia/Go bit-exact and Mojo within `1.5e-8`; hand/TOML/JSON exact, Q16.16 event-vector exact over the bounded 30-step `I=0.2` co-sim with v/u errors below 0.003/0.0001; formal Q16.16 BMC depth 4 | corrected source implementation |
| Ermentrout-Kopell | ✅ | ✅ | ✅ | ✅ shared-lib | accel spike count — 0/20/64 @ I=-0.5/0.1/1.0 over 2000 steps (theta flow, forward Euler, modulo 2π); schema enrolment — hand/TOML/JSON exact, Q16.16 RTL spike counts 0/45/64 @ I=-0.5/0.5/1.0 over 2000 steps with circular phase error below 0.081/0.089/0.025 rad; generated integer C/Rust and Verilog event/state words cycle-exact over 240 steps at both current signs; formal Q8.8 safety BMC depth 4 | `2669a831f` |

Each model carries a committed benchmark: a Go `Benchmark*` in the services lane for the
conductance models (Wang-Buzsaki, Morris-Lecar, Connor-Stevens, Hodgkin-Huxley, FitzHugh-Nagumo),
plus source-hashed executable closure benchmarks for Connor-Stevens, Hodgkin-Huxley, AdEx, ExpIF,
Lapicque, Perfect Integrator, Quadratic IF, Theta, DPI, COBA LIF, Escape Rate, Poisson, IQIF,
McCulloch-Pitts, Sigmoid Rate, Threshold-linear Rate, Wilson-Cowan, Jansen–Rit,
Montbrió–Pazó–Roxin, and Wong-Wang,
and a committed `benchmarks/bench_<model>*.py` harness with its recorded per-backend result for the
FFI-dispatched models (McKean, Hindmarsh-Rose, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR,
Rulkov map, GLIF, Mihalas-Niebur, Medvedev map, Cazelles map, Courbage-Nekorkin map, Izhikevich 2007,
Chialvo map, Ibarz-Tanaka map, and Ermentrout-Kopell). A dedicated per-kernel benchmark harness for the Rust
`accel/rust/safety` crate remains a tracked open lane item.

## In progress

No model is mid-flight right now. The next remediation unit will appear here when it opens; see the
internal working tracker for the queue.

## The rest of the catalogue

Every other catalogued model has a **faithful, tested Python reference** (and most have the real
Rust engine acceleration path), but its four `accel/{rust,go,julia,mojo}` kernels are **stubs, fakes,
or not yet verified** — the polyglot-stub-remediation sweep is replacing them model-by-model. Those
models are **deliberately not ticked here**: per-model status is promoted onto this page only after a
unit is closed and verified at source, so this table never claims completion it has not proven. As of
the latest landed commit that is **thirty-nine polyglot-complete models** out of the full catalogue;
the remaining **114** source-model units are Python-faithful with an acceleration chain still under
remediation.

## How a model graduates onto this page

A model moves from "the rest" to **polyglot-complete** when a remediation unit delivers, and this
page records, all of:

- real executable dynamics in `accel/rust/safety`, `accel/go`, `accel/julia`, and `accel/mojo`;
- an executed Python-parity test per lane (bit-exact, or spike-count with the transcendental caveat);
- an honest committed benchmark.

The internal working tracker for the sweep (with per-unit close-out detail) is
`docs/internal/POLYGLOT_STUB_REMEDIATION_BACKLOG.md` (developer-local, not published). This page is
its public, per-model summary.
