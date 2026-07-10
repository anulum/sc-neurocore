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

Each of the five carries a committed benchmark in its lane (Go `Benchmark*` for the models above;
the Rust-safety per-kernel benchmark harness is a tracked open lane item, so those numbers currently
come from the Go/engine lanes rather than a `benches/` target in the safety crate).

## In progress

| Model | Rust engine | Rust safety | Go (cgo) | Julia | Mojo | Notes |
|---|---|---|---|---|---|---|
| McKean | ✅ bit-exact | 🔶 real, strengthen | ✅ bit-exact | ✅ bit-exact | 🔶 built lib, band | Class-A FFI model: the Go-cgo, Julia, Mojo and Rust-engine backends are already real and parity-tested bit-for-bit via `tests/test_mckean_backends.py` (the PWL right-hand side is exact arithmetic). The `accel/rust/safety` kernel is already real RK4 with an independent-RK4 cross-check, but still carries a vestigial `#![allow(…)]`, lacks the golden spike-count test, and lacks a `Default` — the remaining strengthening pass to graduate it to ✅. |

## The rest of the catalogue

Every other catalogued model has a **faithful, tested Python reference** (and most have the real
Rust engine acceleration path), but its four `accel/{rust,go,julia,mojo}` kernels are **stubs, fakes,
or not yet verified** — the polyglot-stub-remediation sweep is replacing them model-by-model. Those
models are **deliberately not ticked here**: per-model status is promoted onto this page only after a
unit is closed and verified at source, so this table never claims completion it has not proven. As of
the latest landed commit that is **five polyglot-complete models (plus McKean in progress)** out of
the full catalogue; the remainder are Python-faithful with an acceleration chain still under
remediation.

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
