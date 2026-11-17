<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Release notes 4.2.0 -->

# SC-NeuroCore 4.2.0

These notes are the GitHub Release text for `v4.2.0`. The complete list of changes is the `[4.2.0]` section
of [CHANGELOG.md](https://github.com/anulum/sc-neurocore/blob/main/CHANGELOG.md).

## Highlights

- **Behaviour cases in model descriptors.** A descriptor may state what its model must do in an optional
  `[[behaviour]]` list:
  - an `interval` bounds the spike count or a state variable after a number of steps at constant inputs;
  - a `contrast` states that one drive gives more than another by a margin;
  - a case may set declared parameters within their declared ranges.

  `tools/model_behaviour_report.py` runs every case through the model's public class and records whether it held and
  whether it was exercised. `tools/model_behaviour_variants.py` flips the sign of one term of the bound schema at a
  time and requires every such variant to break at least one case. 19 catalogue models state cases. The cases take
  no part in a readiness tier.
- **Backend lanes with their own evidence.**
  - A descriptor's `[backends.<lane>]` table takes its lane from a closed list (`python`, `rust`, `rust_safety`,
    `julia`, `go`, `mojo`) and may name the tests that are the lane's evidence and the tracked files that
    implement it.
  - Every lane has a readiness facet, `backend:rust_safety` among them.
  - A lane's receipt records the lane that ran and the libraries it loaded.
  - No shipped descriptor names lane evidence yet, so no model changes tier.
- **Benchmark records say what they still describe.**
  - The coverage census marks each record `current`, `outdated` (with the changed paths) or `unbound`.
  - A record whose sources changed only in comments or documentation can be marked `reviewed`, with a recorded
    reason. The census accepts the reason only while the code, compared without comments, equals the code the
    record measured.
  - The evidence gate checks every source digest a record holds in its tables. Hosted CI holds the committed gate
    report to what the gate writes.
- **A documented engine.** The Rust engine is documented module by module from its code. The undocumented public
  items fell from 3,646 in 292 files at 4.1.0 to 478 in 96 files, and each documented module denies missing docs.
  Reading the code against its documentation found the behaviour differences listed below.
- **One toolchain manifest.**
  - `tools/toolchain_manifest.toml` gives every tool version the CI declares. Its `--check`, also a pre-commit
    hook, refuses a declaration that differs, a floating runner, an undated nightly and an unpinned install.
  - Hosted jobs name their runner images.
- **In the wheel.**
  - `sc_neurocore.spike_train_statistics` provides the core spike-train statistics with NumPy as its only
    dependency.
  - `mux_dot_product_variance` and `interleaved_dot_product_variance` give the variance behind a random and a
    round-robin multiplexer.

## Behaviour changes

These change results or refuse input that 4.1.0 accepted. In each case the earlier behaviour was a defect.

- **Fractional-order LIF (engine).** The memory keeps the potential after a reset. Before, the engine fired on every
  step after its first spike; it now matches the Python model.
- **Loihi CUBA and Loihi 2 (Python).** Each state decays by its quotient truncated toward zero, as in the engine. A
  negative state no longer leaks more than the same positive one.
- **Akida (engine).** A zero weight is no input event, the weight factor is a running product of `modulation`, and
  the accumulator keeps integrating after its spike, as in the Python model.
- **Continuous-attractor ring.** The engine and the Python model agree bit for bit, and both put the bump at the
  first of equal maxima.
- **Hodgkin-Huxley (Go service and Rust safety kernel).** Both run `round(1/dt)` sub-steps rounded half to even, as
  the Python model does. This changes results where `1/dt` is a half-integer, for example at dt 0.4, and does not at
  the default dt of 0.01.
- **Dendritic NMDA neuron.** The NMDA current depolarises below `E_NMDA` in every lane. Before, its sign was
  reversed, so glutamate removed spikes.
- **`HomeostaticLIFNeuron.reset_state`.** It restores the threshold and the rate trace the neuron was built with.
- **Fixed-point LFSR.** A seed or a `reset` value with no set bit in the register is refused.
- **`SurrogateLif` (engine).**
  - A configuration the fixed-point LIF refuses raises `ValueError` with the reason. Before, it ended the Python
    process.
  - On a spike step the surrogate is evaluated at the membrane that crossed threshold, before the reset (Neftci,
    Mostafa and Zenke 2019, eq. 5), not at the reset voltage. With the defaults a spike step got about 1/24 of that
    gradient before.
  - A refractory step's gradient is 0, the derivative of its forward pass.
- **DNA kinetic simulators (engine binding and Python `KineticSimulator`).** Both refuse, before anything runs, what
  they cannot simulate, with the same words. Before, some of these ended the Python process, and an unknown gate type
  ran as AND. They refuse:
  - a `dt` that is not finite and above 0, and a `duration_s` that is not finite and at least `dt`;
  - a concentration that is not finite or is below 0;
  - `NAND` and `XOR`, and in Python `CATALYTIC`, which have no rate law;
  - a gate with an input count its type does not read;
  - an output named `"time"` or written twice;
  - a run above 2²⁵ samples.

  The binding also refuses unequal lists, a repeated input name and an unknown gate type. The Python time axis steps
  by `dt`, as the engine's does.
- **Stochastic-computing helpers.**
  - `low_discrepancy_error_bound` refuses a length that is not a power of two (`GuaranteeNotDetermined`).
  - `bipolar_decode`, `bipolar_multiply` and the cross-correlation diagnostics refuse input that is not bits.
- **Synthesis exit status.** `compile --synthesize` and `deploy` exit with status 1 when nextpnr or the bitstream
  packer fails or a step exceeds its time limit.
- **Descriptor parsing.** A `[dynamics]` entry that is not expression text is refused, and so is a `[backends]` entry
  outside the lane list.
- **Names a module only imported.** Five modules no longer bind, at module level, names they imported from elsewhere.
  Three now import them inside the functions that use them, and two no longer use them. These names can no longer be
  imported from those modules; import each from its own module:
  - `spike_train_pca` and `functional_connectivity` from `sc_neurocore.analysis`, not from
    `sc_neurocore.identity.decoder` or `sc_neurocore.identity.substrate`;
  - `FeynmanKacHeatSolver` from `sc_neurocore.physics.heat`, not from `sc_neurocore.experimental.physics_routes`;
  - `build_model_compile_traceability` from `sc_neurocore.studio.compile_traceability`, not from
    `sc_neurocore.studio.platform.model_compile_process`.

  `sanitize_ident` is no longer importable from `sc_neurocore.compiler.verilog_expr_emitter`; its home is the
  private `sc_neurocore.hdl_gen._ident`. No name that a module defines is removed.

## SNN Studio: development preview

SNN Studio remains a **development preview**, labelled experimental. Changes in this release:

- **Accessibility.**
  - Screen readers get the Studio's own names, and actions are announced.
  - Keyboard focus is visible and lands on a named element.
  - Tab and Shift+Tab leave the equation editor.
- **The equation editor.** It no longer scrambles a typed equation and keeps every `/` and `?` typed into it.
- **'Generate RTL'.**
  - It compiles the nine catalogue models whose parameter names differ from their schema.
  - It offers only the Q-formats that hold the model.
  - It answers a refused compile or co-simulation with HTTP 422 and the reason.
- **Process and signal handling.**
  - `sc-neurocore studio` stopped with SIGTERM runs its exit handlers.
  - The storage service and the worker launcher stop on SIGTERM whatever signal mask they inherit.
  - A Studio without a configured job root removes its scratch root on a normal exit.

## Supply chain and security

- tornado 6.6.1 in the optics and workstation locks, for CVE-2026-126347.
- Go 1.27.4 in every module and workflow, for GO-2026-6859.
- The toolchain manifest's `apt install` pattern no longer takes exponential time on crafted input (CodeQL
  `py/redos`).
- The Rust documentation ratchet refuses an original-image folder reached through a link.
- Dependency pins that no recorded hold keeps back are moved to current releases.

## Corrections

- **Citations.** 85 citations in model pages, engine docs, two model sources, a descriptor summary and two test
  docstrings named a false co-author in place of the real authors. Each is now its DOI's registry record, read a
  second time by DOI.
  - 22 citations that no registry holds stay as written and are listed for review.
  - Three statements that the cited papers' abstracts do not show are marked in place. Nothing unfound was
    removed.
- **Claims without a record.** Pages that claimed hardware results on silicon, billion-neuron runs, or power
  proportional to spike rate for the event-driven neuron no longer do. Each states the evidence that exists.

## Removed

- The internal Rust safety crate (`publish = false`) no longer exports seven placeholder codec modules that returned
  constant values. The production codecs are those of the engine crate.

## Known limits

- Linux engine wheels are tagged `manylinux_2_39`. Systems with an older glibc (for example Ubuntu 22.04,
  Debian 12 or RHEL 9) get no prebuilt engine wheel; build the engine locally with the Rust toolchain
  (`python -m maturin develop --release` in `bridge/`).
- The conda recipe in `conda/` is a draft. Its source digest is filled in only after the sdist is published.
- The iCE40 bound covers only the calibrated blocks inside their measured domain, measured with the recorded Yosys
  and nextpnr versions.
- Benchmark timings are recorded on named hosts and dates and are not portable speed claims. A record the census
  calls `outdated` describes earlier source bytes; it does not say that the measured value changed.

## Artifacts

The SHA-256 of every file attached to this release follows below, written by the release workflow.
