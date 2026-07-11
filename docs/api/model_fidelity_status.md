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
   transcendentals (`exp`/`tanh`/…) the trace is not bit-exact across libms, so the **spike count**
   is the declared, stable parity observable.
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

Every lane below carries the real model dynamics with an executed Python-parity proof. The Mojo
column distinguishes an executable kernel from a maintained parity note: the Mojo neuron-kernel lane
has not yet been promoted to a build/CI target (a separate cross-cutting task), so some models
specify the Mojo contract as an honest note rather than executable code. No lane below is a fake
stub.

| Model | Rust safety | Go | Julia | Mojo | Parity basis (golden) | Landed |
|---|---|---|---|---|---|---|
| Wang-Buzsaki | ✅ | ✅ | ✅ | ✅ executable | spike count — 3 AP @ I=10, 20 macro steps (Gauss-Seidel) | `0ebf4ea88` |
| FitzHugh-Nagumo | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 1 AP @ I=10/100 steps, 5-spike train @ I=0.5/2000 steps (RK4, exact RHS) | `729e0a2ea` |
| Morris-Lecar | ✅ | ✅ | ✅ | ✅ executable | spike count — 0/3/5 @ I=0/50/100 over 2000 steps (RK4, tanh/cosh) | `bc46a0fb5` |
| Connor-Stevens | ✅ | ✅ | ✅ | 🔶 parity note | spike count — 0/2/9 @ I=0/10/20 over 100 macro steps (RK4, exp) | `4776283ba` |
| Hodgkin-Huxley | ✅ | ✅ | ✅ | 🔶 parity note | spike count — 0/6/9 @ I=0/10/20 over 100 macro steps (baseline-Euler, exp) | `0b23b5653` |
| McKean | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 0/1/7 @ I=0/0.2/0.5 over 20000 steps (RK4, exact piecewise-linear RHS) | `059871140` |
| Hindmarsh-Rose | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/26/52 @ I=0/3/5 over 2000 steps; co-sim exact hand/TOML/JSON/Q16.16 counts — 0/0/26/40/52 @ I=0/2/3/4/5 over 2000 RK4 steps; declared Q16.16 +1 crossing boundary over 5000 steps at I=2 through I=5; formal Q8.8 BMC depth 4 | `this commit` |
| FitzHugh-Rinzel | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/1/8 @ I=0/0.3/0.5; co-sim exact spike count — hand/schema/Q16.16 RTL 8 @ I=0.5 (3000 RK4 steps, cubic RHS) | `498376221` |
| Pernarowski | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 7/17/27 @ I=0 over 2000/5000/8000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 17 @ I=-0.1/0/0.1/0.2 (5000 RK4 steps, autonomous cubic burster); formal Q8.8 reset-spike BMC depth 4 | `c384ac0cd` |
| Terman-Wang | ✅ | ✅ | ✅ | ✅ shared-lib | accel spike count — 0/1/3 @ I=-1/0/0.5 over 8000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 0/1/3 at the same operating points (RK4, cubic + tanh gate); formal Q8.8 reset-spike BMC depth 4 | `ce04dd6f9` |
| Wilson-HR | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/1/4 @ I=0/2/10 over 5000 steps; co-sim exact spike count — hand/schema/Q16.16 RTL 0/1/4 at the same operating points (RK4, polynomial RHS, hard voltage reset preserving recovery); formal Q8.8 reset-spike BMC depth 4 | `4d9810e2f` |
| Rulkov map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo ULP-bounded, 0/4/34 @ I=0/0.1/0.5 over 2000 iterations; co-sim: hand/TOML/JSON exact and Q16.16 RTL ten-event short-window trajectory within 0.001 state error @ I=1.5/30 iterations; Yosys Q16.16 H2 synthesis + formal Q8.8 BMC depth 4 | `7d5889e10` |
| GLIF | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/54/95 @ I=0/30/50 over 1000 steps; co-sim exact hand/TOML/JSON/Q16.16 spike counts — 0/0/23/54/86/95 @ I=0/15/22/30/45/50 (four-state RK4, candidate-first adaptive reset); formal Q8.8 BMC depth 6 | `ecd799d58` |
| Mihalas-Niebur | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact accel — 0/142/333 @ I=0/2/5 over 1000 steps; co-sim exact hand/schema/Q16.16 RTL — 0/0/0/31/60/87/131/157/207/256 @ I=0/0.5/1/1.5/2/2.5/3.5/4/5/6 over 1000 RK4 steps; explicit I=3 boundary 111/111/112; formal Q8.8 BMC depth 3 | `081dd569c` |
| Medvedev map | ✅ | ✅ | ✅ | ✅ shared-lib | Rust/Julia/Go bit-exact — 0/92/112 @ I=0/0.2/0.5 over 1000 iterations (expanding chaotic circle map). Mojo per-step ULP-bounded only: FMA fusion amplifies on the chaotic map, so it does not reproduce the exact spike count (by design) | `9936d997d` |
| Cazelles map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo per-step ULP-bounded, 5/182/204 @ I=0/0.5/1.0 over 1000 iterations; co-sim: hand/TOML/JSON exact and Q16.16 RTL event-exact at I=0.5/1.0/2.0 over 30 iterations with state error below 0.0004; I=0.05 is excluded; formal Q8.8 BMC depth 4 | `22110c66d` |
| Chialvo map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: all compiled lanes ULP-bounded to the source recurrence and event-count exact at I=-0.05/0/0.01/0.05/0.1/1.0 over 1000 iterations (0/26/30/0/0/1); pinned 500,000-iteration benchmark records 12,935 events in every lane; co-sim: hand/TOML/JSON exact and Q16.16 event counts 0/2/3/0/1 at I=-0.05/0/0.01/0.1/1.0 over 100 iterations, with stable-point x/y errors below 0.055/0.093 and oscillatory timing explicitly excluded; formal Q8.8 BMC depth 4 | `this commit` |
| Courbage-Nekorkin map | ✅ | ✅ | ✅ | ✅ shared-lib | accel: Rust/Julia/Go bit-exact, Mojo per-step ULP-bounded, 157/193/168 @ I=-0.3/0/0.3 over 1000 iterations; co-sim: hand/TOML/JSON exact, Q16.16 event-exact at I=-0.3/0/0.3 over bounded 30/20/30-iteration windows, and Q32.32 event-exact at all three inputs over 30 iterations with state error below 0.00003; autonomous Q16.16 30-iteration trace is an explicit 4/6-event boundary; formal Q8.8 BMC depth 4 | `63826b513` |
| Izhikevich 2007 | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 0/3/14 @ I=0/100/400 over 2000 steps (RK4, quadratic v-nullcline, spike reset v→c / u+=d; Mojo ULP-bounded but the per-spike reset re-synchronises the trace so its counts always match) | `75b32d935` |
| Ibarz-Tanaka map | ✅ | ✅ | ✅ | ✅ shared-lib | bit-exact — 0/69/235 @ I=1.0/1.5/2.0 over 2000 iterations (modified-Rulkov 2-D map, rational/linear fast branch, reset-on-spike; Mojo ULP-bounded but the reset re-synchronises so its counts match) | `780d29050` |
| Ermentrout-Kopell | ✅ | ✅ | ✅ | ✅ shared-lib | accel spike count — 0/20/64 @ I=-0.5/0.1/1.0 over 2000 steps (theta flow, forward Euler, modulo 2π); schema enrolment — hand/TOML/JSON exact, Q16.16 RTL spike counts 0/45/64 @ I=-0.5/0.5/1.0 over 2000 steps with circular phase error below 0.081/0.089/0.025 rad; generated integer C/Rust and Verilog event/state words cycle-exact over 240 steps at both current signs; formal Q8.8 safety BMC depth 4 | `2669a831f` |

Each model carries a committed benchmark: a Go `Benchmark*` in the services lane for the
conductance models (Wang-Buzsaki, Morris-Lecar, Connor-Stevens, Hodgkin-Huxley, FitzHugh-Nagumo),
and a committed `benchmarks/bench_<model>*.py` harness with its recorded per-backend result for the
FFI-dispatched models (McKean, Hindmarsh-Rose, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR,
Rulkov map, GLIF, Mihalas-Niebur, Medvedev map, Cazelles map, Courbage-Nekorkin map, Izhikevich 2007,
Chialvo map, Ibarz-Tanaka map, Ermentrout-Kopell). A dedicated per-kernel benchmark harness for the Rust
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
the latest landed commit that is **twenty-one polyglot-complete models** out of the full catalogue; the
remainder are Python-faithful with an acceleration chain still under remediation.

## How a model graduates onto this page

A model moves from "the rest" to **polyglot-complete** when a remediation unit delivers, and this
page records, all of:

- real dynamics in `accel/rust/safety`, `accel/go`, `accel/julia`, `accel/mojo` (Mojo may be a
  maintained parity note until the Mojo build-target lane is promoted — never a fake stub);
- an executed Python-parity test per lane (bit-exact, or spike-count with the transcendental caveat);
- an honest committed benchmark.

The internal working tracker for the sweep (with per-unit close-out detail) is
`docs/internal/POLYGLOT_STUB_REMEDIATION_BACKLOG.md` (developer-local, not published). This page is
its public, per-model summary.
