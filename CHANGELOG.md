<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Changelog -->

# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

## [Unreleased]

### Security

- The release-only Semgrep environment takes wcmatch 11.2 instead of
  8.5.2, which matches multi-star patterns against untrusted names in
  polynomial time (GHSA-53pm-wgxf-c6qf). Semgrep 1.185.0 still declares
  wcmatch~=8.3, so `requirements/semgrep-overrides.txt` carries a reviewed
  pin, as it does for click and PyJWT. The repository's own Semgrep scan
  of `src` and `tools` gives the same findings and scans the same files
  with either version.

- The dependency lock audit records CVE-2026-16052 (http.cookies) for the
  Mojo build environment's locked Python 3.15.2 as a reviewed exception
  until a 3.15 release carries the fix. The environment sets or parses no
  HTTP cookie, the installed `http/cookies.py` must equal the CPython 3.15.2
  release file, and the exception lapses on 2026-12-01 or with a newer
  release.

- The hypervisor keeps each tenant to the region it holds. A second
  `allocate` for a tenant that already holds a region returns that region
  instead of taking another; before, the first region stayed allocated with
  no owner after the tenant was released. The new `Hypervisor.preempt`
  moves the firewall rule with the region, and `PreemptionManager.preempt`
  does the same when it is given the firewall, so a preempted tenant can no
  longer reach the addresses it lost; a victim rule that reaches across the
  region's edge is removed whole. `verify_isolation` now reads the
  region model: overlapping or empty regions, and a rule that does not lie
  in a region its tenant holds, are reported. The unwired Julia and Mojo
  copies of `verify_isolation` are not updated.

- The streaming inference server (`sc-neurocore serve`) no longer answers
  with exception text. A fault inside a step returns 500 with a fixed
  sentence; malformed JSON or inputs return 400 with an authored one. The
  request body is bounded (1 MiB by default; 413 unread above it) and must
  declare its length. `/reset` takes the same lock as `/step`, and a
  population network with projections but no `step` is refused instead
  of being run with its projections ignored.

- The engine API compatibility gate reads the crates.io sparse index only
  over HTTPS from index.crates.io, and only for a name in Cargo's crate-name
  alphabet. Both are checked before the request opens, so no other scheme,
  host or path can be read; the security scan's bandit check had flagged
  the open (B310).

- The Mojo lock audit waits up to 180 seconds between bytes for the
  conda-forge channel index, which the service builds for each request;
  vendor source files keep 30 seconds. Uncached, the index once took 74
  seconds to its first byte, so all three 30-second attempts timed out and
  the audit stopped, as it did in half of the CI shards on 18 November.

- The Mojo lock audit accepts four more CPython advisories against conda-forge
  Python 3.15.2 for two weeks at a time, under the rules of the two already
  accepted: CVE-2026-124249 (`http.cookiejar`), CVE-2026-115559 (`imaplib`
  STARTTLS), CVE-2026-115791 (the `gzip`, `tarfile` and `zipfile` command
  lines) and CVE-2026-19552 (`ssl.get_server_certificate`). Grype's database
  listed them on 18 November and every audit of the lock failed. CPython has
  fixed each for 3.16, but no 3.15 release carries a fix, and conda-forge
  lists nothing newer than 3.15.2. Each record binds the installed package
  digest and one CPython 3.15.2 release file, byte for byte, and lapses on its
  review date, once Grype stops reporting it, or as soon as the channel lists a
  newer final release. A proof file may now be up to 128 KiB, since
  `http/cookiejar.py` and `imaplib.py` exceed the former 64 KiB.

- The security packet's Gitleaks lane runs with `.gitleaks.toml`. It keeps
  every default rule and allowlists only the 33 reviewed false positives of
  the generic API key rule at the 4.2.0 release: SHA-256 content digests
  keyed by source paths whose names contain "lapicque" (the rule reads the
  "api" in it as a key name), a doc comment, an engine binding name and a
  formal module name. Each allowlist names the rule, the files and the exact
  shape of the value, so a key of another shape in those files, a
  digest-shaped key in any other file and every other rule are still
  reported; a check with planted keys shows each still found. The lane's
  summary records the configuration's SHA-256.

- The dependency lock audit accepts a named npm advisory on a development-only
  package for at most two weeks per review, while the npm registry lists no
  release outside the advisory's vulnerable range. Every installation of the
  package must be the recorded version and a development dependency; every
  other advisory, a stale record and an npm error still block. The OSV lane
  of the security packet applies the same records and proof, and blocks any
  entry of its report that it cannot read. The first
  record is GHSA-fc5q-39xc-vjpf (picomatch <= 4.0.7, no fix released), a
  development dependency of the Studio frontend's build and test tools.

### Added

- `sc_neurocore.accel.contrastive_ssl` runs the contrastive SSL kernels (the
  InfoNCE loss, one CSDP phase and the goodness score) on a chosen lane:
  Python, the original and the floor; Julia (`accel/julia/contrastive/ssl.jl`,
  now loaded through a facade); or a new Mojo C-ABI library
  (`accel/mojo/contrastive_ssl`, built by `tools/build_accel_backends.py`).
  Every input is validated with the original's checks before a lane runs.
  `backend="auto"` follows the host's measured order, which is Python alone
  until a measurement is committed. The parity tests state each lane's bound:
  - the loss agrees within a relative `1e-9`;
  - the Julia CSDP phases are exact;
  - the Mojo CSDP phases are within four rounding steps of the operands;
  - goodness agrees within a relative `1e-12`.
  `ssl.jl` now accepts a zero-width weight matrix as the original does, and
  adds in the original's order. The contrastive benchmark times every lane
  through the dispatcher on its record host. The Julia lane inventory is
  regenerated, which also records an earlier change to `edge/bitstream.jl`.

- The coverage union of the main CI workflow runs on every run that is not
  cancelled, and the test shards keep their coverage data after a failure,
  so the union's verdict is visible even when a leg fails first. On every
  main run from 2026-11-17 a failed shard had skipped it and hidden a
  coverage gap.

  Its 100 % gate is now tools/coverage_exception_gate.py, which reads the
  union's coverage json report and the owner-approved register
  tools/coverage_platform_exceptions.json. The register excepts one line,
  the success return of the GPU waveform encoder in
  spike_codec/_waveform_gpu.py, which runs only on a physical GPU adapter.
  The gate fails on any other missing line or branch, on a registered line
  whose text changed, and on a GPU source that differs from its physical-GPU
  evidence. No omit entry or pragma is used.

  The evidence, tools/gpu_waveform_encoder_evidence.json, records one run on
  a physical GPU: the adapter, the GPU encoder test passing with GPU packets
  equal to CPU packets, the excepted line run, and the digest of each GPU
  source. tools/gpu_waveform_encoder_evidence.py writes it from that run's
  reports and refuses a run that does not prove the line.

  A new job, "GPU encoder refusal on a software adapter", builds the engine
  with its gpu feature on Mesa's software Vulkan driver at a pinned version
  and holds the refusal of that adapter, with its reason, through the public
  encoder; its coverage joins the union. It makes no hardware or performance
  claim.

- A split import gate (`tools/sc_nc_import_gate.py`, with its plan in
  `tools/sc_nc_split_map.json`) measures how far the integrated package is from
  the planned SC-NC extractions. The map gives the order of the destinations,
  the destinations each one may import, and the destination of every top-level
  package and module. The gate resolves every import in the tracked sources and
  reports each one that a destination may not make, and each import of
  SC-NeuroCore code from a package that leaves for the circuit platform. The
  main CI workflow runs it in report mode, which never fails a build. A
  destination becomes blocking with `--blocking` once its count reaches zero.

- The accelerated-lane stub triage records a seat's reading of each of the
  83 originals its rule left unclear, with the reason: 20 are compute (the
  holonomic adapters' JAX numerics behind the package's shim, bit-level SC
  kernels, neuron and synapse updates, Huffman coding) and 63 application
  (interfaces, re-exports, exporters and generators, Studio, servers,
  demonstrations, utilities). Of 406 originals, 339 are compute and 67
  application, whose 120 stubs form the deletion list for the owner. A call
  of a name imported from a numeric library now counts as numeric work, a
  stub is found in a test or benchmark only at a path boundary, and the
  unmapped count is split by lane.

- `tools/accel_stub_triage.py` and `docs/_generated/accel_stub_triage.json` class the
  Python original of every Mojo and Julia stub as a compute kernel or as application
  code, by a stated rule read from the original's syntax tree: numeric work and no
  application import is compute, an application import and no numeric work is
  application, and anything else stays unclear until a seat's recorded reading decides
  it. The record gives the deletion list for the owner (each application stub with its
  loaders, the tests and benchmarks naming it, and its declaration) and the order in
  which compute kernels are replaced by real code, ported from today's Python module.
  Of 406 originals, 319 are compute, 4 application and 83 unclear.

- The Julia lane inventory names, for each of the 340 files Julia cannot parse, the Python module it
  copies and how that module was found: 284 by the module their header names and 56 by reading the
  header's joined directory (`adapters_holonomic` for `adapters/holonomic`); none is ambiguous or
  unfound. Each records the share of its lines found verbatim in that module. The record is the
  worklist for replacing transcripts of compute kernels with real code.

- The Mojo lane inventory names, for each machine-translated stub, the
  Python module it copies, so the record is the worklist for replacing
  stubs with real code: `transcribed_module`, the `transcription_rule`
  that names it (a source of truth in the header, the header's stem, the
  file's stem, or the module holding most of the stub's lines when
  several share the stem), and `verbatim_line_share`, the stub's line
  strings found verbatim in that module as `n/d`. A tie between modules is
  listed as `ambiguous` with its candidates and never chosen. Of the 363
  stubs, 361 name one module and 2 are ambiguous.

- Import legacy BCI spike recordings through the codec console. Supply the
  original time and channel geometry and convert validated RLE runs to
  framed raster packets for verification and NumPy export.

- Add installed recording commands under `sc-neurocore codec` to discover
  and inspect codecs, encode NPY, PCM16 WAV or event CSV, verify packet files,
  and decode to NumPy arrays. The benchmark command compares installed
  Python and Rust routes with packet and output parity checks.

- Qualify the installed spike-codec catalogue with the base wheel alone and
  with the compiled Rust engine. The wheel test job verifies all 18 variants,
  records module provenance, and compares Python and Rust packets and decoded
  output. Waveform and predictive instances expose their selected native scope
  and prediction stage, respectively.

- Export the event and band-power codec from `sc_neurocore.spike_codec`.
  The Python and Rust encoder and decoder can reset for an independent stream.
  Paired measurements cover both timing modes and preserve legacy event packets.


- The Rust waveform codec reads and writes complete WFC2 packets in all
  five modes. The installed Python API selects the whole-container backend.
  Both implementations validate the declared geometry and compressed sections
  and can read legacy WFCX spike packets.


- The lossless waveform stream exposes a GPU encoder in wheels built with
  the engine's \`gpu\` feature. The encoder reports the physical adapter and
  graphics API, writes one group per channel and uses a CPU decoder.
  Python, Rust and GPU waveform endpoints can reset for a new stream.
  Python and Rust reject invalid range-coder tails and ADC level values.


- `AERSpikeCodec` supports matching AERX and AERI packets in Python and
  Rust. The UDP adapter translates sample positions to absolute ticks,
  accepts reordering within a transfer and reports missing or invalid
  datagrams.

- `StreamingSpikeCodec` supports Python and Rust SSCF containers and
  standalone frames. Pull iterators conserve the short final frame and
  provide cancellation at the consumer boundary. Decoders check geometry,
  mask lengths and padding before reconstructing samples.

- The inter-channel delta codec supports complete Python and Rust DSCX paths.
  Both preserve valid legacy packets and check the group layout and local
  references before decoding the embedded ISI body.

- `PredictiveSpikeCodec` supports explicit Python and Rust backends for all
  four predictors. The Rust engine now encodes and recovers context
  residuals and updates world-model weights. World-model probability
  evaluation still uses NumPy. The base wheel includes the predictor, so
  the experimental world-model package is optional.

- Predictive lossy packets merge complete timing bins before prediction.
  The decoder recovers the compact or original-sized bin raster from
  the timing information in the packet. New world-model packets retain
  nondefault thresholds. Existing default-threshold lossless packets
  preserve their bytes and decode contract.

- `tools/julia_lane_inventory.py` and `docs/_generated/julia_lane_inventory.json` classify every
  tracked Julia source: whether the package declares it authoritative or a mirror, which Python
  modules build its exact path, and whether Julia's parser accepts it (nothing is evaluated). Of
  638 files, 298 parse and 340 are not Julia; every declared or loaded file parses, and a test keeps
  it so. Without Julia, `--check` carries each unchanged file's verdict by sha256.

- `sc_neurocore.accel.host_isa` reads the host's x86-64 psABI level
  (`x86-64-v1` to `x86-64-v4`, or `unknown`) from the CPU flags Linux prints.
  A measured order whose record states its host's flags applies only on a
  host at that level or above; one that does not applies on the exact CPU
  model only.

- The polyglot dispatch order no longer follows measurements whose sources
  have changed, and it is the same in an installed package as in a source
  checkout. The selector read every benchmark record of its shape from
  `benchmarks/results`, found through its own file's location, so an
  installed wheel, which has no such directory, always used the static
  order, and a source checkout promoted lanes on measurements the coverage
  census calls outdated. It now reads
  `sc_neurocore/accel/measured_backend_orders.json`, which ships in the
  package and is written by `tools/generate_backend_speed_table.py` from
  current or reviewed records only, with every other record listed with
  its reason. No record the selector can use is current today (16 are
  outdated and 7 bind no source), so every kernel dispatches by its static
  order until its record is measured again. Only runs from a source
  checkout change order; installed packages already used the static order.

- `tools/mojo_lane_inventory.py` classifies every Mojo source of the
  acceleration package by stated rules: comment-only, a machine-translated
  stub (Python text in string literals and `return 0`), an implementation
  that still uses `fn`, which Mojo 1.0.0 and later refuse, or an
  implementation with `def`. It records which sources the backend builder
  builds, which Python modules name each library, and the package's own
  authoritative and mirror declarations. The generated
  `docs/_generated/mojo_lane_inventory.json` is checked for drift by its test,
  and every translated stub must lie under a declared mirror glob. It also
  classes every neuron model by how it chooses a backend for `auto`: through
  the backend selector, by an order fixed in code, with one backend only, by
  loading a backend directly, or with no accelerated backend. Only tracked
  files are read.

- The compiler can emit clocked circular CORDIC units for sine, cosine,
  arctangent and two-argument arctangent. Each unit has fixed latency,
  ready/valid handshakes and a signed fixed-point error bound. The Python
  reference and generated C11, Rust and Verilog return the same codes.

- Out-of-context Vivado v2025.2 synthesis and placed-and-routed implementation
  artefacts (`hdl/reports/vivado_sc_shd_event_axi_*_xc7z020_100mhz.rpt`),
  reproducible Tcl script (`hdl/pynq/synth_shd_event_xc7z020.tcl`), and `100 MHz`
  constraints (`hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc`) for
  `sc_shd_event_axi` (`P=8`, full trained SHD weights and 32-bucket time wheels
  in `21` XC7Z020 Block RAM Tiles plus delay tables in `40` `RAMD64E` LUTRAMs:
  synthesised `12,549` LUT / `14,068` FF / `0` DSP, routed `12,605` LUT /
  `14,075` FF / `0` DSP / setup `WNS -4.901 ns` at `100 MHz`, with `67.11 MHz`
  `Fmax` estimated from the 100 MHz run's WNS and hold not closed out of context
  (`WHS -0.852 ns`), plus routed DRC `0` errors / `44` warnings including `RAMB`
  async-control checks), bound to public documentation by
  `tests/test_hardware_number_provenance.py`, alongside Git-tracked source
  enumeration in `tests/test_shd_event_core.py`.

- `hdl/pynq/sc_shd_driver.py` adds integrity-checked weight and delay loading
  (`load_network` over `hdl/fixtures/shd_int_network.json`, matching
  `zlib.crc32`, validating per-selector `LOAD_TARGET` bounds before bus writes,
  and checking `LOAD_STATUS.sealed` / `refused_range` without claiming
  authentication) and bounded-poll event-mode inference (`run_event`,
  `classify_event`, and `classify(..., mode="event")`) for `sc_shd_event_axi`,
  leaving reserved `0x08..0x10` untouched, refusing unsealed starts and counts
  outside `0..15` before bus writes, routing boolean rasters to `SPIKE_IN_0..4`
  and integer rasters to `FRAME_COUNT_0..17` without tail padding, and aborting
  with the hardware status word on `error`, `overflow`, `commit_refused`, or
  `!busy && !done` (`tests/test_shd_driver.py`).

- `hdl/shd_event/sc_shd_event_axi.v` adds the 9-bit AXI4-Lite slave wrapper and
  integrity-checked group-word loader (`0x90..0xAC`, reflected IEEE CRC-32 in
  little-endian byte order matching `zlib.crc32`, with `SLVERR` refusal of
  out-of-range `LOAD_TARGET` commits) plus the write-only 140-nibble input
  count window (`FRAME_COUNT_0..17` at `0xB8..0xFC`) alongside `sc_shd_event_p`,
  keeping `0x00..0x8C` offsets compatible with `hdl/pynq/sc_shd_driver.py`
  except `0x08..0x10` (reserved on the event bitstream, read `0`, writes
  refused with `SLVERR`) and driving core `rst_n` from `S_AXI_ARESETN` alone.
  `hdl/formal/sc_shd_event_axi.sby` proves and covers the seal, target-range,
  reserved-offset, start-admission, `T_MAX` (`1..1024`), and
  `frame_valid`/`frame_ready` invariants, and `tools/cosim_shd_event_axi.py`
  with `tests/test_shd_event_axi.py` verifies bit-exact count and binary
  co-simulation plus negative `SLVERR` refusals across Icarus Verilog and
  Verilator.

- The event-driven integer SHD core `hdl/shd_event/sc_shd_event_p.v` and its
  shared `P`-lane LIF datapath `sc_lanes.v` run the integer axonal-delay
  speech network through per-layer 32-bucket time wheels and an idle
  group-word weight and delay load port, matching `tools/shd_int_reference.py`
  bit for bit on every accumulator. Target leaves are chosen solely by
  filelist (`generic.f`, `ihp_sg13g2.f` and `xilinx_7series.f`) with no
  target preprocessor conditionals, `tools/cosim_shd_event_core.py` runs
  `tb_sc_shd_event_p.v` in Icarus Verilog and Verilator, and
  `hdl/reports/sc_shd_event_p_xc7z020_synth_xilinx.txt` records the Yosys
  0.70 `synth_xilinx` block-RAM inference and flattened resource statistics
  for XC7Z020.

- The SC non-resetting adaptive integrate-and-fire neuron states two
  behaviour cases. At a constant 20.0, the main drive of its 256-step
  reference receipt, it fires once in those 256 steps, at step 139; its
  voltage is never reset and the event raises the threshold by 5. At
  1000.0 the voltage outruns the rising threshold and the neuron fires on
  every step from the second of 12 on, 11 times, which a count of
  crossings would miss: its event is a level. The window at 20.0 breaks
  all 32 sign variants of its three equations. Twenty-seven of the 185
  catalogue models now state cases.

- The equation compiler can emit a standalone combinational function unit for
  either lowering. The source-checkout synthesis command records Xilinx
  7-series cell and DSP counts, verifies the entire mapped interface against
  the emitted RTL, and reports unsupported function and format combinations
  explicitly.

- The escape-rate neuron states two behaviour cases. Its reference test's
  seed corpus runs it with the hazard held at rho_0 0.25 and no current,
  where the stream depends on the seed alone; the row of the default seed
  fires 943 times in 4096 steps, and an LFSR written out in the test gives
  the same. A case cannot set the voltage, so the row is stated with
  v_threshold at -70, the default resting and reset voltage, which holds
  the hazard at rho_0 again. With the defaults at a current of 40.0 it
  fires 49 times in 1024 steps. The corpus row breaks the two sign
  variants inside the leak, and the driven window the other three.
  Twenty-six of the 185 catalogue models now state cases.

- The ISI spike raster codec can encode and decode varint and Huffman packets
  through the Rust engine. `SpikeCodec` selects the installed native engine
  by default and can use either backend explicitly. Both backends preserve
  the existing format and validate framing and timestamps during decoding.
  Both readers accept unambiguous legacy varint packets with at least
  33,554,432 time steps. Legacy varint packets with 16,777,216 through
  33,554,431 time steps collide with the Huffman marker and require
  re-encoding from the original source.
  For a quantised packet, callers can request the number of complete bins
  with the matching timing precision.

- The Nagumo-Sato neuron map states three behaviour cases. At a current
  of 0.0 it fires once in 3 steps, the steps its model test writes out
  with the value of y after each. At 0.05, a drive value of its reference
  trace and of its Q16.16 co-simulation, it fires 4 times in 12 steps. At
  1.0 it fires on each of 12 steps without crossing the threshold, which
  a count of crossings would miss: its event is a level. The window at
  0.0 breaks two of the four sign variants of its equation; the input term
  is zero there, and the window at 0.05 breaks it and the memory term.
  Twenty-five of the 185 catalogue models now state cases.

- The Aihara chaotic neuron map states three behaviour cases. Over 12
  steps, the window its Q8.24 co-simulation test runs for the hand model,
  the schema and the RTL, a current of 0.0 fires on every second step, 6
  times, and -0.03, one of the drive values of its reference trace, 4
  times. With the parameters of its own level test (k 0.0, alpha 0.01,
  bias 0.2) it fires on both of 2 steps without crossing the threshold,
  which a count of crossings would miss: its event is a level. The window
  at 0.0 breaks two of the four sign variants of its equation; the input
  term is zero there, and the window at -0.03 breaks it and the memory
  term. Twenty-four of the 185 catalogue models now state cases.

- The stochastic-computing divider `hdl/sc_cordiv.v` and the MUX scaled
  adder `hdl/sc_mux_add.v` have their first tests. An Icarus co-simulation
  checks each core bit for bit against the package's own model,
  `sc_neurocore.utils.bitstreams.sc_divide` and
  `sc_neurocore.edge.bitstream.sc_mux`. The divider runs 4,096 random input
  pairs that reach all eight combinations of held output and inputs, 4,096
  pairs of streams compared against one shared random word, and a reset in
  the middle of a stream; the adder runs all eight inputs and 1,024 random
  ones. A divider that copies its numerator only when the denominator bit
  is 1, or an adder with its inputs swapped, fails the tests.

- The adaptive-threshold integrate-and-fire neuron states two behaviour
  cases at a current of 20.0: it fires once in 160 steps, the window its
  reference test checks against the two exact relaxations written out in
  the test, and 3 times in 1000 steps, with the intervals lengthening as
  the raised threshold relaxes. The first case breaks 9 of the 10 sign
  variants of its two equations; the threshold's relaxation term changes
  nothing before the first event, and the longer window breaks it too.
  Twenty-three of the 185 catalogue models now state cases.

- The sigma-delta neuron states a behaviour case: a current of 2.0 fires 42
  times in 128 steps, the window its reference test checks against
  equations written out independently of the class. Each of the 11 sign
  variants of its three equations breaks the case, and so does a change of
  any of its parameters. Twenty-two of the 185 catalogue models now state
  cases.

- The exponential integrate-and-fire neuron states two behaviour cases, two
  of the 1,000-step windows its Q32.32 test runs for the hand model, the
  schema and the RTL, which must give the same event counts: a current of
  20.0 fires 2 times, and a current of 100.0 fires 9 times. Each of the 8
  sign variants of its membrane equation breaks the first case; the second
  depends on the reset voltage, which leaves the first unchanged.
  Twenty-one of the 185 catalogue models now state cases.

- The Chialvo map neuron states two behaviour cases, two of the 100-step
  windows its Q16.16 test runs for the hand model and the RTL, which must
  give the same event counts: a current of 0.01 fires 3 times, and a current
  of 1.0 fires once, crossing the threshold and staying above it, so a count
  of downward crossings would give 0. Each of the 8 sign variants of its two
  equations breaks the first case. Twenty of the 185 catalogue models now
  state cases.

- The benchmark evidence gate checks the source digests a record holds once
  per variant, `variants.<i>.source_hashes`, which it could not address
  before: 72 digests in the autonomous-learning, Bioware and NIR graph
  modularisation records. A gate opts in with `"variant_source_hashes":
  "variant_revision"`, and each variant is checked at its own revision: a
  clean one at its `git_head`, a parent at its `source_ref`, and one
  measured on a dirty tree at the record's commit, the only commit that
  can hold the bytes it hashed. A digest that differs is
  `stale_variant_source_hash`, or `dirty_variant_not_committed` for a dirty
  variant; a revision the history lacks is `unresolvable_variant_revision`.
  The three gates opt in, all 72 digests match, and the committed gate
  report is unchanged.

### Changed

- The docstring scope guard lets an enrolled file's policy row leave in
  the commit that deletes the file from the index. A file removed only
  from the working tree still counts as present, a rename is a deletion
  and an addition whose new path must be enrolled, and every other
  omission from the baseline cohort is still refused. Before this, no
  change could delete an enrolled file at all.

- The quantum-annealing bridge validates its user-supplied scalars in one
  module, `sc_neurocore.bridges.annealing_validation`, instead of six
  private copies. Behaviour changes:
  - Booleans and strings are refused for every real-valued input. Model
    biases, couplings, QUBO entries and offsets, compiler scales and
    penalties, solver temperatures, chain strength and anneal time, and
    schedule durations and encoder values used to accept `True` as 1.0
    and `"1.5"` as 1.5.
  - NumPy integer and float scalars and `Fraction` are accepted wherever
    a real number is expected, and NumPy integers wherever an integer
    count or index is expected. The analysis inputs, counts and indices
    refused them before.
  - A refused type reads "<name> must be a real number" instead of
    "<name> must be numeric"; non-finite values still read "must be
    finite", and an integer too large for a float is reported as not
    finite.
- The quantum-annealing API page lists the new module and describes each
  module by its responsibility rather than its line count.

- Models can resolve their simulator from the installed package instead of
  importing their executor. `sc_neurocore.core.backends.resolve_simulator`
  reads the `sc_neurocore.simulators` entry points, and the alpha model is the
  first to use it: `AlphaNeuron.simulate` resolves `alpha` to the engine's batch
  simulator, with an unchanged result. The registry refuses, with a typed error
  that names the cause, two registrations of one model id, a registration from a
  distribution other than sc-neurocore, a registration that fails to load and a
  checkout without installed metadata; it never skips one for another backend.
  Refresh the editable install after a change to that entry point group.

- `sc_neurocore.energy.folded_estimator` no longer imports
  `FoldedResourceMetrics`, the summary of the NIR bridge's FPGA compiler,
  not even for type checking. It names the seven counts the estimate reads
  in a structural `FoldedMetrics` protocol, which that dataclass satisfies
  unchanged, so the energy package needs no bridge module. A test keeps the
  protocol, the attributes the estimator reads and the dataclass fields in
  step. Behaviour is unchanged.

- The copyright notices state the project's two dates, concepts from 1996
  and code from 2020, instead of a third range starting in 1998. This
  covers COPYRIGHT, the NOTICE files, CONTRIBUTING, the README, the REUSE
  annotations and binary-artefact sidecars, the documentation site footer,
  the Sphinx configuration, examples, notebooks and reports. Where a file
  already carried both standard lines, the extra line is removed. The three
  Python files this touches, the Sphinx configuration and two examples, join
  the docstring policy with the docstrings it requires.

- `sc_neurocore.safety_cert.evidence_packet` held two responsibilities: the
  packet's format, manifest and verification, and running the external
  toolchains it records. The runners and their provenance (`run_formal_job`
  with its job helpers, `synthesize`, `place_and_route`, `tool_versions`,
  `source_identity`, `checkout_root`, `FormalTarget` and `TOOLS`) now live in
  `sc_neurocore.safety_cert.evidence_toolchain`; the code moved unchanged. The
  packet module re-exports the public names as the same objects, so historical
  imports and pickles keep working, and the compiler guide names the new
  module.

- `sc_neurocore.bridges.dna_io` held three responsibilities: interchange
  exports, synthesis-order and assembly planning, and text views. It keeps the
  exports (GenBank, FASTA, NUPACK input, JSON); the oligo cost estimate, the
  assembly protocol and the plate layout moved unchanged to
  `sc_neurocore.bridges.dna_lab`, and the circuit diagram and kinetics chart to
  `sc_neurocore.bridges.dna_views`. The `dna_mapper` façade exposes every name
  as the same object under its historical module, so imports and pickles
  through it are unchanged; the architecture contract and the responsibility
  table list the two new modules.

- `sc_neurocore.studio.synthesis` held two responsibilities: running the
  bounded FPGA toolchains, and judging whether a synthesised design fits its
  target device. The judgement (`capacity_verdict`, `capacity_device_name`,
  `capacity_sentence` and the judged resource keys) now lives in
  `sc_neurocore.studio.synthesis_capacity`; the code moved unchanged. The
  toolchain module re-exports those names as the same objects, so historical
  imports and pickles keep working, and the Studio pipeline imports them from
  their own module.

- `sc_neurocore.studio.hardware_path` held two responsibilities: running one
  catalogue model through the stages to synthesis, and classifying what its
  emitted RTL is guaranteed by. The classification
  (`model_guarantee_summary`, `formal_guarantee`, `readiness_labels`,
  `render_model_guarantee_text`, `MODEL_GUARANTEE_SCHEMA_VERSION`) now lives in
  `sc_neurocore.studio.model_guarantee`; the code moved unchanged. The hardware
  path re-exports those names as the same objects, so historical imports and
  pickles keep working, and its guarantee stage records what the new module
  returns.

- `sc_neurocore.studio.project` held two responsibilities, project storage
  and the graph-to-target pipeline. The pipeline (`run_pipeline`,
  `PIPELINE_ROUTE`, `PIPELINE_COSIM_STEP_LIMIT`, `PIPELINE_Q_FORMATS`) now
  lives in `sc_neurocore.studio.project_pipeline`, and the storage module
  re-exports those names as the same objects, so historical imports and
  pickles keep working. The pipeline process task imports the pipeline from
  its own module. The code moved unchanged.

- The package version is stated once, as `VERSION` in
  `src/sc_neurocore/_version.py`, a module that imports nothing from the
  package. The package root re-exports it as `__version__`, so
  `sc_neurocore.__version__` is unchanged. Modules that stamp the version
  into a manifest, a fit, a certificate packet or a Studio record now import
  `_version` instead of the package root, so they depend on the module the
  split map places in SC-NC-CORE rather than on the facade; the split import
  gate reports 12 fewer findings. The version surface audit reads
  `_version.py`, and refuses a package root that states a version of its own
  or does not re-export `VERSION`.

- The network runner in `engine/src/network_runner` is documented apart
  from `create_neuron`: its network, populations, projections, results and
  the TC-LIF and McCulloch-Pitts input adapters, each read against its code.
  Stated by reading: a projection with a delay of `d` steps delivers a
  spike `d + 1` runner steps after it is emitted; `add_projection` checks
  nothing, so a projection that names a missing population makes `run`
  panic, while `admit_projection` refuses it; `step_population_with_currents`
  applies no projection; and the spike-event packing refuses a timestep
  above `u32::MAX` but does not check the neuron index, whose bits above the
  32nd would be lost.

- The engine's four RK4 neuron ports in `engine/src/rk4_neurons.rs`
  (Izhikevich, Izhikevich 2007, AdEx, Hodgkin-Huxley) and its
  `py_rk4_neuron_simulate` function are documented in full, each read
  against its code. Stated by reading: a Hodgkin-Huxley step takes
  `(1 / dt).round()` RK4 sub-steps, about 1 ms, with a half rounded away
  from zero, so a `dt` above 2 leaves the state unchanged, and its spike
  is the upward crossing between the start and the end of that step; the
  Izhikevich, AdEx and Hodgkin-Huxley ports validate nothing; and the
  Izhikevich 2007 batch keeps the steps before a refused one. The Python
  function's docstring now names the models it accepts, their default
  steps, the arrays it returns and its errors.

### Fixed

- The six SHD weight-extraction tool test modules skip again without
  the torch extra. Since they import their names explicitly, they took
  them straight from the tool, which exits without torch, before their
  support module's torch skip could run; they now take them from the
  support module, so the skip runs first.

- The quantum-cognition Rust compilation test builds its test binaries
  in the test's own temporary directory instead of fixed names in the
  shared `/tmp`, so concurrent runs cannot collide or run each other's
  binary.

- The notebooks `06_network_engine` and `07_identity_substrate` held
  their header and some text as CP1250 mojibake, so they still showed a
  `1998–2026` copyright line and broken dashes and arrows. Their text is
  decoded back to UTF-8, and the copyright line is the same two-line
  concepts and code pair as in the other notebooks. The earlier search for
  the old range missed both, because it matched only correctly encoded
  text.

- Tracked files no longer name development-host paths, the private monorepo
  or the group's private record tree. 56 facet receipts recorded between
  2026-09-05 and 2026-10-05 named the absolute interpreter in `command[0]`;
  each now holds `python`, the value later receipts record, carries a note
  with its original digest, and is re-sealed with the receipt module's own
  seal. Benchmark records and the Posner external-data files use
  repository-relative paths or neutral tokens such as `<run-root>`, and the
  checksum list of the Posner research lane is recomputed for the files that
  changed. `bench_balanced_resonate_and_fire.py` reports an absent Julia or
  Mojo instead of falling back to a fixed home directory. A test refuses such
  paths in every tracked file, text or binary, except six files whose lines
  redact, refuse or audit them, each with its reason.

- `SHDAccelerator.classify(mode="legacy")` waited for `CTRL.done` in an
  unbounded loop, so a hung legacy wrapper hung the caller. It now reads
  `CTRL` at most `max_poll_iters` times, raises `SHDDriverError` with the
  last status word on a timeout or when the wrapper drops `busy` without
  `done`, and refuses `max_poll_iters < 1` before any bus write, as the
  event path does.

- The integrity-checked loader of `sc_shd_event_axi` seals a network only
  after exactly `LOAD_WORDS` committed words (`4,940` for `P=8`) with a
  matching CRC-32. It had sealed any non-empty stream whose CRC matched, so
  a partial reload after a reset, or a driver given a larger `lanes` than
  the bitstream's `P`, sealed over tables the stream never wrote.
  `LOAD_STATUS` bit 4 (`count_short`) reports a seal refused for the count,
  and `load_network` raises on it. The testbench covers both refusals and a
  partial reload, and the formal proof asserts that a seal or start follows
  a complete load, with a cover of the refused case; the formal inventory
  the pages state grows to `591` statements (`403` assert, `55` cover).
  The Vivado reports are
  re-measured, and the published figures follow them: routed `12,605` LUT /
  `14,075` FF (were `12,957` / `14,082`), setup `WNS -4.901 ns` at
  `100 MHz` (was `-3.343 ns`), so the `Fmax` estimated from it falls from
  `74.95` to `67.11 MHz`. The critical path stays inside the core
  (`20` logic levels, was `17`); the base design rebuilt with the same
  tool gives the committed figures again.

- The version surface audit reads the package root's syntax tree instead of
  its lines. The root must bind `__version__` exactly once, at the top
  level, by `__version__: str = _version.VERSION`, and import `_version`
  from the package. Any other binding of the name, at any depth and by any
  form (an indented, augmented or walrus assignment, a loop, comprehension,
  `with` or `except` target, a `match` capture, star or mapping rest, a
  `type` alias, a deletion, an import alias, a `global` declaration, a
  function or class of that name), and any `"__version__"` string outside
  `__all__`, such as `globals()["__version__"]` or `setattr`, is refused as
  a version of the root's own. The line-based check had passed an indented
  rebinding and a `globals()` one.

- The coverage union gate of the main CI workflow had gone unmeasured: every
  main run since 2026-11-17 whose state could be read had skipped it,
  because a test shard had failed first, and the first run to reach it
  missed 26 lines in 14 files. Of those, the 22 outside the GPU encoder's
  success path are now reached through their public entry, or removed where
  an earlier check makes them unreachable.

  Tests now ask the spike codecs for backend="python", so the Python
  parsers' header checks and the varint refusal at T = 2**24 run on every
  host, not only where the engine is missing. A child process with an engine
  that lacks a codec, as an older or partial engine build does, holds that
  backend="rust" refuses with the reason and backend="auto" falls back to
  Python and round-trips. The fixed-point function units are compared with
  their reference at format (31, 30), the widest one the format check
  admits, and a bandwidth window narrowed to zero cycles after recording
  reads 0.0.

  scc in sc_neurocore.edge.bitstream no longer has a zero-denominator check
  after its zero-numerator one. By the Fréchet bounds the denominator is at
  least the numerator's magnitude, so the check could fire only within
  rounding of the 1e-7 threshold, where it returned 0.0 for a coefficient
  near -1; the function now returns the coefficient there. The Julia and
  Mojo mirror stubs of scc lose the same check. bridges.photonic_codesign
  refuses a wheel without sc_neurocore.optics before its import, as the
  other wheel-excluded imports do, instead of in a handler whose re-raise
  could not run.

  The ISI header dimensions and the Huffman bit count go through one public
  check, sc_neurocore.spike_codec.wire_fields.uint32_field, which refuses an
  out-of-range value by name; the ISI dimension refusal keeps its message,
  and the Huffman bit count, in Python and in the engine, now reads "Huffman
  bit count must fit uint32". The lossless waveform stream's GPU encoder is
  opened in its own private module, spike_codec._waveform_gpu.

- The split map of the split import gate, tools/sc_nc_split_map.json, has
  its REUSE ownership sidecar, tools/sc_nc_split_map.json.license, with the
  licence and both copyright records. Every JSON file added since the
  2026-10-01 ownership contract must carry one, and the map was added
  without it, so the JSON ownership test failed.

- The Conda audit scanner installer (`tools/install_conda_audit_scanners.sh`)
  retries every failed download of its Syft and Grype archives, up to five
  times, 5 seconds apart. GitHub's release storage answered one hosted test
  shard with a 403, which `curl --retry` treats as final, and the shard failed
  before its tests ran. The pinned SHA-256 of each archive is unchanged and
  still decides whether a download is accepted. The dependency-audit guide
  names the pinned scanner versions, Syft 1.57.0 and Grype 0.121.2.

- The large-checkpoint serialisation test
  (`tests/test_tools/test_function_synthesis_checkpoint_serialization.py`)
  takes the prior arcsin input on every Yosys version. On Yosys 0.70 it used
  the prior arctan input, which exceeds the 64 MiB native parser budget only
  where the SDK is installed at a long path: Yosys writes the techlib paths
  into cell names, so the raw checkpoint grows with the install path. At the
  short path of hosted CI the arctan checkpoint stayed under the budget
  (53.5 MB at a 29-character path), no normalisation took place and the
  fixture failed in every shard that ran it. The arcsin checkpoint is 113.8 MB
  there. The unused arctan fixture is removed.

- The advisory-exception reads (`tools/security_scan/advisory_exceptions.py`)
  repeat Cloudflare's origin statuses 520 to 524 as temporary, like 502 to
  504. The conda-forge channel index is served through Cloudflare, and a
  hosted runner's live read met a 524 (origin timeout) that failed the test
  at once instead of being repeated after 2 and 8 seconds.

- The docstrings, comments, messages and documentation of the new codec,
  function-synthesis and chiplet work use British spelling
  ("normalisation", "serialise", "quantiser", "synchroniser").
  Identifiers, file names and other tools' terms keep their spelling.

- Correct the Rust PyO3 model inventory to 208 wrappers by excluding five
  codec bindings for lossless waveform and spike-event streams.

- The Mojo lane inventory classes 50 more sources as machine-translated
  stubs: 39 implementation sources with `fn` remain, not 89. Its stub rule
  read a single-quoted string as ending at the first quote, so a line
  string holding an escaped quote (`'… \'{name}\' …'`) was not a stub line,
  and it did not count a string assignment parenthesised across lines
  (`var name = (` … `)`); all 50 are `kernels/` mirrors the builder does
  not build. A parenthesised value holding anything but strings is still
  code. The inventory's neuron-model reader also counts the guarded
  `load_mojo_library` as opening a library, beside `ctypes.CDLL`, so
  `connor_stevens` and `hodgkin_huxley` record that they do.

- The backend builder builds the two self-contained Mojo fidelity libraries again
  (`benda_herz`, `sc_stochastic_rate_adaptation`). Their loaders now call the guarded
  `load_mojo_library` instead of `ctypes.CDLL`, and the builder recognised only the latter, so
  it stopped producing libraries that CI requires. It now recognises both, and the Mojo lane
  inventory is regenerated (50 built libraries, as before).

- The chip compiler's precision-loss warning reads "Weight quantisation to
  N-bit introduces …" (it said "quantization"), and its module docstring and
  comment use the same British spelling.

- Every JSON file added since the 2026-10-01 ownership contract now carries
  its `.license` sidecar, and a test holds every tracked JSON to it. Four
  had none: the behaviour report and the equation-variants report in
  `docs/_generated`, `.github/python_reference_inventory.json` and
  `hdl/fixtures/shd_int_network.json`. The two report generators now write
  the sidecar with the JSON and fail `--check` when it is missing or
  differs, through the new `tools/json_sidecar.py`. The policy test, which
  read only `tests/fixtures`, now reads every tracked JSON. The 616 files
  that predate the contract are named in
  `tools/json_ownership_legacy_paths.txt`, a list that may only shrink, and
  vendored upstream files keep their REUSE annotation.

- The generated chiplet package now runs each die on its own clock.
  `sc_chiplet_top` had one `clk` driving every die and both sides of every
  bridge, while `chiplet_constraints.xdc` created one clock per die, at that
  die's period, on the die instance's `clk` pin: dies with different
  `clock_mhz` could not both be right on one net, the bridge registers sat
  under no created clock, and the link constraints joined clock domains the
  RTL did not have. The top now has one input per die, `clk_die_<d>`, which
  drives that die and the matching side of each of its bridges. The XDC
  creates each die's clock on that port and bounds each link between its two
  dies' clocks at the shorter of the two periods; the link latency is the
  bridge's own pipeline. The one `rst_n` reaches each die's domain through
  that die's two-stage synchroniser, and the XDC declares it a false path.
  A bridge counts the link latency in cycles of its destination die's clock
  and sizes its FIFO for the jitter in cycles of the faster clock; at
  200 MHz on both sides nothing changes. Tests check that
  the XDC names only clocks that exist and co-simulate a ring of dies at 200,
  125 and 80 MHz. `rtl.py` keeps validation and generation and moves its
  emitters into `_rtl_die`, `_rtl_links` and `_rtl_top`, written and
  emitting at most 100 columns.

- `BandwidthMeter` no longer keeps every sample. A tenant's cycles
  increase, so no later window reads a sample before the current window's
  base; `record` drops those, and the base lookup is a bisection instead of
  a scan of the whole history. The history stays within one window and its
  base, and every rate equals the whole history's: compared after each of
  400 random samples over 20 seeds and five windows. Fifty thousand samples
  keep at most one window of them. On the former meter that long-run bound
  fails while every rate test passes.

- Tutorial 60 cites "Huang et al. (2022). Formal Verification of Neural
  Networks: A Survey. IEEE Access 10:125603-125618", which neither Crossref
  nor OpenAlex holds, by title or in the journal's 2022 volume. The entry is
  kept as written and marked in place as under review, with nothing unfound
  removed. A verified survey of the field follows it: Huang et al. (2020),
  Computer Science Review 37:100270, read in Crossref, OpenAlex and Semantic
  Scholar. The registry-corrections test pins both.

- On a 64-bit bus, the AXI4-Lite and Wishbone B4 neuron wrappers now place
  their registers 8 bytes apart, and `generate_register_map` takes the same
  `bus_data_width` and publishes those addresses. At the former 4-byte
  stride the register at `0x04` (`I_T`, and every register at an address
  4 modulo 8) sits on byte lanes `[63:32]`, where a master that follows the
  bus's lane rules writes it with `WSTRB 0xF0`. The wrapper stored the value
  in the register's upper half while the neuron reads the lower half, and a
  read-back returned it, so a driver saw success and the neuron never got
  its input. Now each register owns a whole bus word and its data travels
  from lane 0. Buses of 8, 16 and 32 bits keep the 4-byte stride and
  byte-identical wrappers. A simulated master that places data by lane
  drives the neuron on 32- and 64-bit buses, and on the former wrapper it
  fails on 64 bits.

- Twin-sync names now say what the code does. `TimeWarpEngine` takes
  `checkpoint_interval_events` (still the second positional argument);
  `checkpoint_interval_ns`, which always counted events, is a deprecated
  keyword and attribute that warns, and giving both is an error.
  `Checkpoint.compute_checksum` covers every stored field, prefixing each
  array with its dtype and shape and marking None, so checksums and audit
  chains are comparable only within one version.
  `DeltaCheckpoint.compression_ratio` is the full state's bytes over the
  diff's (0.0 for an empty diff), and `state_shape` is a required
  keyword-only field, so an older positional call fails instead of shifting
  into it. A session's `max_divergence` is the tolerance of its divergence
  metric (`DivergenceMetric.tolerance`, default 1.0).
  `TwinSession.digital_events_out`, never counted because nothing leaves the
  twin, is removed from the status and the session snapshot and kept one
  release as a deprecated attribute that returns 0. The class texts no
  longer say that the checkpoint manager keeps identity across a rollback
  (the engine does) or that the global virtual time never falls. The unwired
  Julia, Mojo, Go and Rust copies of the module are not updated.

- A deployment package no longer contradicts itself: its metadata
  `n_synapses` now counts the synapses its blob codes (the explicit weights'
  non-zeros), not the adjacency's; explicit weights must have the
  adjacency's shape, which gives the neuron count; and validation now
  requires every neuron of the network to be placed exactly once, refusing a
  package with an unplaced neuron or a neuron id outside the network, or
  with a neuron count that is not an integer.

- The time-warp engine's anti-messages now cancel what they name: a rollback
  cancels the queued events the node sent after the rollback time at once,
  and sends anti-messages for the processed ones, which remove the event and
  roll its receiver back, or cancel it if the receiver has queued it again;
  an anti-message no longer advances a clock or counts as processed. A
  rollback now returns the node's events processed after the restored
  checkpoint to the queue and runs them again, and recounts its processed
  events. Starvation is measured from the furthest-ahead node, since no node
  can lag behind the global virtual time, which is the minimum. A state
  delta of a two-dimensional array stores changed elements by flat index
  instead of repeating whole rows, records the state shape, and rebuilds the
  state with `DeltaCheckpoint.apply`. Events carry an `event_id`. The engine
  models no node logic, so a cancelled send is not regenerated: a caller
  that models senders injects their valid sends again.

- The Go Chialvo map library computes its exponential with the C library's
  `exp`, the one the Python reference and the Rust and Julia lanes use. Go's
  own `math.Exp` differs from it by one ulp on about one argument in nine, and
  over 1,000 iterations the recurrence carried that difference to `9.8e-14`,
  past the cross-language trace envelope of `5e-14`. The Go trace now equals
  the Python trace bit for bit wherever NumPy's `exp` is the C library's. The
  model's benchmark record is outdated until it is measured again.

- The hardware deployment package no longer codes every weight of a 1-bit
  device as zero: one bit now stores the sign, +1 or -1, and the package
  metadata records `weight_scale`, the factor that turns any code back into
  a weight (code / weight_scale), which was never stored before. A weight
  precision below one bit or above the blob's 16-bit weight field,
  non-finite weights, and placement ids that are negative or too wide for
  their unsigned 32-bit or 16-bit fields are refused with an error instead
  of an overflow or a conversion failure. An empty placement list packages
  with no core used, and validation refuses negative neuron, core and local
  ids.

- The temporal property checks no longer verify what a trace cannot show. A
  window longer than the trace gave a pass even for a neuron firing at every
  step; it now gives `unknown`, or a violation when the whole trace already
  exceeds the bound. A response exactly at the latency deadline now counts,
  latency 0 accepts a spike at the stimulus step, and a deadline past the
  trace end without a response gives `unknown`, as does a causal-order
  window that starts before the trace with no cause recorded. Negative
  neuron ids and stimulus times, which wrapped to the end of the trace, are
  refused, as are ids past the raster, windows below one step and negative
  latencies or delays. The temporal verification tutorial's examples now run
  as printed, and its text no longer claims a check over all inputs.

- The interval checker no longer says a property holds for inputs outside
  its domain: two negative "probability" intervals whose product lies in
  [0, 1], or a negative energy above a more negative cost, now show nothing:
  the check returns false and logs why. An interval must have finite ends in
  order, so a reversed, NaN or infinite end, or an overflowing sum, is
  refused when it is built.

- The software safety monitor no longer reports a negative current, voltage,
  popcount, SC sum or membrane as safe. Every probe takes only values within
  its 16-bit hardware probe range (unsigned, the SCC numerator signed); any
  other value, NaN and infinity included, raises before the monitor changes
  state. Integrality is not checked. Its docstrings now say that four of its
  six properties are proved in the Lean safety bounds and two are axioms,
  instead of calling all six formally proven.

- The chip compiler no longer returns NaN weights with a successful result
  on a 1-bit target: one bit now codes a sign, each weight becoming plus or
  minus the mean magnitude. A precision below one bit, or a weight matrix
  that is empty or not finite, is a violation and quantises no layer. The
  report's violation and warning lines and the precision-loss warning are
  tested instead of excluded from coverage.

- A Mojo library is no longer loaded on a host that cannot run it. Every
  Mojo library is built for x86-64-v3, and on a lower host (or one whose
  level is unknown) loading or calling it could end the process with an
  illegal instruction. All 78 Mojo loads now go through one loader that
  reads the host's level first and refuses the lane with a typed
  `MojoIsaRefusal` (an `OSError`) before `ctypes.CDLL` is called, so the
  dispatch falls through to the next lane and finally the Python floor.
  Shared Go and Mojo loaders guard only their Mojo branch.

- Backend dispatch, the dense IF comparison and Studio's benchmark
  contribution read the host CPU model by one rule
  (`sc_neurocore._host_identity.cpu_model`): the `model name` line of
  `/proc/cpuinfo`, else `platform.processor`, else `"unknown"`. The dense IF
  comparison raised `StopIteration` on a host without a `model name` line
  (some Arm and non-Linux hosts) instead of keeping the static order, and the
  three copies of the reading differed in how they decoded the file. The rule
  imports only the standard library, so the conversion module still loads no
  acceleration package or native runtime.

- The hypervisor places tenants by their whole QoS and meters bandwidth in
  real units. A region fits a tenant only with enough neurons and synapses
  (`HWRegion.fits`), for allocation and migration alike; a migration to a
  region too small, or to the region the tenant holds, is refused before
  anything moves. `BandwidthMeter.throughput` counts the spikes after the
  window's first sample over the cycles they span (two samples of 10 spikes
  ten cycles apart give 1.0, not 2.0), reads `window_cycles`, and gives 0.0
  for one sample or no span instead of the raw count. Mbps come from the new
  `throughput_mbps`, and `exceeds_quota`, with the bits per spike and the
  clock as required keywords, both checked; neither those nor the limit
  accepts a bool. `record` refuses a cycle that is not later than the
  tenant's last, since the window runs along increasing cycles. Several
  realtime tenants share one half of a priority plan, so a third one no
  longer gets nothing. The Go and Rust stubs and the unwired Julia and Mojo
  copies are not updated.

- `generate_bus_wrapper` (`sc_neurocore.hdl_gen.bus_interface` /
  `_bus_wrappers`) now connects `u_neuron.spike_out` in both AXI4-Lite and
  Wishbone B4 wrappers so `irq_spike` and `reg_spike_count` track neuron
  spikes instead of floating undriven, preserves `addr[ADDR_WIDTH-1:2]`
  4-byte word decoding matching `generate_register_map` across 8/16/32/64-bit
  buses (ignoring unaligned low address bits `addr[1:0]` within each 4-byte
  slot), slices each parameter register to its declared width in `params`,
  latches decoupled `AW`-first and `W`-first AXI4-Lite writes with
  `S_AXI_WSTRB` byte-lane enables and `BREADY`/`RREADY` stall protection
  instead of requiring simultaneous `AWREADY && WREADY`, and rejects invalid
  identifiers, register collisions, or out-of-range bus/data widths with
  `ValueError`.

- The stochastic doctor (`compute_scc`, `estimate_precision`,
  `compute_histogram`, `audit_layer`) now accepts only bitstreams of zeros and
  ones and checks them before choosing the Rust core or the Python fallback.
  Other values used to be counted by value in Python and by set bits in Rust,
  could divide by zero in the Python SCC or index past the histogram, and
  floats or out-of-range integers were truncated or wrapped into a byte first;
  they now raise `ValueError`. Empty or unequal SCC streams, a non-positive
  histogram `word_size` and an audit of a layer with no neuron or no bit, which
  produced NaN fields, are refused too, and `BitstreamAuditReport.to_json`
  raises instead of writing NaN tokens. Complex values are refused, and a layer
  in Fortran or transposed order is audited alike on both backends instead of
  being refused for its layout on the Rust core only. Results for valid input
  are unchanged on both backends.

- `EventDrivenSimulator` now leaks every neuron exactly from its own last
  update, `v_rest + (v - v_rest) * exp(-dt / tau_mem)`, instead of never
  leaking a neuron that had not yet spiked and decaying later inputs from the
  last spike time again and again. An event due after a run's `duration`
  stays queued for the next run instead of being dropped, and the
  clock-driven estimate counts a started millisecond (a 0.5 ms run no longer
  reports zero operations). Non-positive sizes and time constants, negative
  delays, non-finite values, unknown neuron ids, and inputs earlier than the
  last processed event are refused with `ValueError`, and so is, with a zero
  refractory period, a cycle of zero-delay connections, which used to keep
  `run` firing at one instant without end. A new `membrane_potential`
  property returns a copy of each neuron's potential as of its last update.

- `ChipletGenerator.generate` now wires functional multi-die AER routing and
  AXI-Stream interposer transport instead of leaving `sc_aer_router` fanout
  tables unconnected, dropping packets across `delay_pipe` backpressure stalls,
  or corrupting binary AER headers with unmatched link LFSR XOR. Die wrappers
  bind `(dst_die, dst_neuron, weight_q88)` fanout tables, shortest-path
  next-hop routing, and lossless fixed-priority arbitration across local and
  incoming links; bridges stall cleanly on `m_tready` deassertion with optional
  `decorrelate_links=True` matched TX/RX LFSR link scrambling across the CDC
  FIFO and latency pipe; standalone `sc_chiplet_rtable_<d>` modules size arrays
  by `die.n_neurons`, widen `target_neuron` to `PKG_AER_W` for heterogeneous
  dies, and emit valid negative signed fixed-point literals; `sc_chiplet_top`
  exposes per-die AER ingress/egress ports; XDC link delays carry
  `-datapath_only`; and invalid topologies or routes fail closed with
  `ValueError`.

- `RTLModule.from_verilog_source` and `UVMGenerator` now parse parameterised
  packed and unpacked Verilog/SystemVerilog port declarations, conditional
  compilation directives (`ifdef`, `ifndef`, `elsif`, `else`, `endif` against an
  optional `defines` set, empty by default), ANSI port initialisers, and
  non-ANSI module headers without silently dropping ports or turning `wire`/`reg`
  keywords into port names. Grouped header `parameter` and body `localparam`
  integer constant expressions (`$clog2`, IEEE 1800 left-associative `**` below
  unary operators, exact integer `/` and `%` truncation, bitwise, shift,
  comparison, `signed`/`unsigned` qualifiers with or without whitespace before
  `[`, and ternary operators bounded within `2**64`) are evaluated fail-closed,
  preserving parameter evaluation errors when referenced, refusing unrecognised
  or interface port items, constant function calls, and unsupported compiler
  directives (`define`, `include`, and macro uses) inside the parsed parameter
  and port declarations with `ValueError`, propagating AMBA clock/reset names
  (`*_aclk`, `*_aresetn`, `pclk`, `hclk`, `ap_clk`) across driver, monitor,
  assertion-bind, and formal-link templates, and emitting valid unpacked array
  declarations and >32-bit sized hexadecimal bounds.

- `SDCGenerator.generate` in `src/sc_neurocore/asic_flow/decks.py` excludes the
  primary clock port from `set_input_delay` and `set_driving_cell` via
  `[delete_from_list [all_inputs] [get_ports <clock_name>]]`. Verified open-PDK
  Liberty suffix tables and PVT corner definitions live in
  `src/sc_neurocore/asic_flow/liberty_corners.py`, matching real Sky130
  (`skywater-pdk-libs-sky130_fd_sc_hd/timing` @ `5a42fd01a0ee`, collated by
  `open_pdks` @ `0c1f82326278`) and GF180MCU (`gf180mcu_fd_sc_mcu7t5v0/liberty`
  @ `43beb45e4d32`, `openroad/orfs` @ `86dfae2d567b`) Liberty filenames.
  `MultiCornerAnalysis` in `signoff.py` checks explicit corner suffixes and
  GF180MCU nominal tokens against those tables without silent fallback, records
  `TSMC28` honestly as its `0.81 V`, `125 °C` slow library without a nominal
  suffix, and refuses default corner calls or unsuffixed Liberty paths
  (`TSMC28`, `INTEL16`, `CUSTOM`) with `ValueError`.

- Edge inference preserves each input's packed stream and connection mask
  across layers. Packed probability and correlation count only logical bits
  in Python and on the Rust edge runtime.
- The Go AER service honours its listen port and forwards registered UDP
  events with per-delivery acknowledgement sequences. It checks the sender
  of each acknowledgement, gives restarted processes independent sequence
  offsets, and expires unresolved deliveries. The Python simulation mirrors
  these rules, and its supervisor reaps stalled children.

- Eleven citations on the model pages and in two engine sources are corrected
  to their works' records. Each record is in Crossref and was read a second
  time by DOI in PubMed or Semantic Scholar:
  - Vervaeke et al. 2010 is *Neuron* 67:435–451;
  - Hirst & Edwards 2001 is *J Physiol* 535:165–180;
  - Lopez-Poveda & Eustaquio-Martín 2006 is in *J Assoc Res Otolaryngol*
    7(3):218–235, cited three times on the inner hair cell page, with its DOI;
  - Korenbrot 2012 is in *Prog Retin Eye Res* 31(5):442–466;
  - Pillow et al. in *J Neurosci* 25:11003 is from 2005;
  - Eshraghian et al. in *Proc IEEE* 111:1016 is from 2023;
  - Tsodyks et al. 2000 is *J Neurosci* 20(1), RC50.

  The other entries of those pages' reference lists were checked the same
  way. Entries that no registry holds, or whose record does not identify
  the work, are kept as written and listed for review; nothing is removed.
  A test holds the corrections and the one form kept for review in the
  engine source.

- The ADC-to-spike sensor quantiser RTL (`hdl/sensors/adc_to_spike_quantiser.v`)
  now preserves signed fixed-point bounds and arithmetic rounding shifts across
  nominal and down-scaled input widths instead of saturating every decimated
  window to `Q_MIN` or `Q_MAX` through unsigned concatenations, and computes
  exact `|Q_MIN|` magnitude when budgeting negative-polarity spikes.

- Populations of one neuron type with different parameters or timesteps now
  compile to separate FPGA neuron modules instead of raising a parameter
  conflict error. The folded interconnect accepts cross-population parameter
  differences when timesteps match and refuses distinct timesteps within one
  neuron type.

- `VerilogGenerator`, `AEREmitter` (`sc_neurocore.hdl_gen`), and the Rust
  safety mirror (`src/sc_neurocore/accel/rust/safety/verilog_generator.rs`)
  now bind `Dense` layers to the `sc_dense_layer_core` module contract
  (`N_INPUTS`, `N_NEURONS`, `DATA_WIDTH`, `start_pulse`, `stream_len`,
  bit-expanded `x_input_fp`, `weight_fp`, `y_min_fp`, `y_max_fp`, `cfg_leak`,
  `cfg_gain`, `I_t`, `spikes`, `step_valid`, `run_done`, and `running`)
  instead of non-existent `NUM_NEURONS`, `input_bus`, and `output_bus` ports,
  forward `bus_width` from `VerilogGenerator.emit_async_aer` into
  `AEREmitter`, drive `output_bus` to zero when no `Dense` layers are
  declared, reject duplicate layer identifiers or `output_width != n_neurons`
  mismatches fail-closed, and remove all type-checker and coverage suppression
  pragmas from `verilog_generator.py` and its test suite. The core holds one
  weight per input and one dot product, so all `n_neurons` outputs of a
  `Dense` layer carry the same value; the API page and `add_layer` say so.

- The Z3 safety supervisor no longer reports an unknown run as clean. When
  its verifier thread ended without a verdict (a panic), the run returned
  its step count; it now ends with `VerifierEnded`. A forced shutdown is
  told apart from a failed safety check: `PySpikingControllerPool.start`
  raises "force_shutdown was called" instead of reporting a safety
  violation, and `is_safety_tripped` is set only by the verifier.

- `FaultResilienceSuite` declares its evaluation function as a callable from
  the list of weight matrices to an accuracy, so strict type checkers check
  callers, and the module no longer silences the checker on its constructor.

- The safety supervisor no longer calls a negative marking safe.
  `supervisor::verify_bounds_at_depth` asserted the snapshot's marking
  into Z3, so a negative one made every query unsatisfiable. That read
  as "safe". It now returns false before the solver runs.
  `PySpikingControllerPool.start` runs its control loop without the
  Python GIL, so `force_shutdown` from another Python thread stops it.
  With the GIL held and no step limit, no other Python thread ran.

- The timeout case of the Studio event-stop test no longer races the
  worker's start-up. The job's execution limit starts when the supervisor
  begins to watch the worker, so it also covers the worker's start-up and
  imports. On a loaded host those took longer than the case's 10-second
  limit, the job timed out before its first training event, and the case
  failed without saying why. The case now submits with the same
  120-second limit as cancellation and, once the worker is training,
  moves the supervisor's clock past it, so the limit lapses while the
  worker trains. A job that ends before its first event fails with a
  message. The product's limit is unchanged.

- The Studio HTTP case of two clients saving one new workspace no longer
  depends on how fast the host is. It started both saves at once with a
  0.3-second lock timeout, so on a loaded host the second save waited
  longer than that for the first and got 503 instead of 409. The second
  save is now sent once the first holds the workspace, both wait with the
  product's own lock timeout, and the case expects the first save to
  succeed, the second to be told of revision 1, and the first's state to
  be kept.

- Three imports of packages the wheel leaves out now refuse with the
  source-checkout reason instead of a bare `ModuleNotFoundError` in an
  installed wheel: `sc_neurocore.bridges.photonic_codesign`, which needs
  `sc_neurocore.optics` as a whole; `SpikeMonitor.cross_correlation`
  (`analysis.spike_stats`); and `BitstreamEncoder(mode="chaotic")`
  (`chaos.rng`). A new test reads every shipped module's syntax tree and
  fails on any import of an excluded package that the refusal does not
  precede, and requires each known excluded import by its module and
  target. The blocked-package subprocess calls the three features and
  round-trips the world-model predictor, which the wheel now ships.

- A panic in the engine no longer ends the Python process. The release
  profile the wheels are built with sets `panic = "unwind"`, where it had
  `"abort"`, so a panic reaches Python as `PanicException`. The two
  documented panics Python could reach are refused first:
  `RustHybridLinearAttentionNeuron(dim=0)` raises `ValueError` before its
  step divides by zero, and `DifferentiableDenseLayer.get_weights` copies
  through the fallible path and raises `MemoryError` on an allocation
  failure. Integer overflow still wraps in release builds; the engine page
  states both as its contract.

- The event training browser step installs its Chromium before it runs. It
  gives Playwright its own cache directory, where no browser had been
  installed, so every run failed at browser launch. A test now holds every
  workflow step that moves the cache and runs Playwright to install its
  browser first.

- CI runs the three Studio training browser specs no hosted workflow had
  run. The live browser step of the test job now lists the training
  preregistration and training conversion specs, which its configuration
  already selected but its file list left out, and a new step runs the
  event training suite with the two settings it needs: a fresh absolute
  SC_NEUROCORE_STUDIO_EVENT_TEST_ROOT and the repository root on
  PYTHONPATH, since its server imports a test helper. CONTRIBUTING.md and
  the spec's header document them. A new contract test reads every
  workflow's Playwright invocations and fails when a live spec runs in
  none of them against a backend, or when the mocked default run selects
  one.

- The public pages state one count of the spike train analysis toolkit's
  functions, 127, the number of functions `sc_neurocore.analysis` exports.
  They said 124, 125 or 132. `docs/api/analysis.md` no longer says it covers
  every metric of Elephant, PySpike, SpikeInterface and NeuroTools: no test
  compares the toolkit with them, and the page now says so. A test binds
  each page that states the count to the exports and refuses an unlisted
  one. The v3.12 milestone list and the v3.13 report keep the counts of
  their time.

- CI installs the ECP5 router, nextpnr-ecp5 0.6-3build5 with its chip
  database, on the leg that runs the live browser journeys. The
  catalogue-to-silicon journeys route the linear and the polynomial IF
  neuron on the Studio's ECP5 target, and that leg had only the iCE40
  router, so the first of them stopped at place and route ("nextpnr-ecp5
  not found") once the live suite reached it. The toolchain manifest pins
  both packages and gives the router's line to the host check. A new
  contract test requires the router of every target a live journey routes
  to be installed, at the manifest's version, before the journeys run.

- The function synthesis guide excludes the legacy ABC `&cec -y` executable
  that reports false equivalence for complemented outputs. Every use of this
  solver path requires first, middle and last output phase-flip controls
  before a positive result receives proof credit.

- The notebook guide's test table names the test files that exist. It
  named 14 files that were split into one file per tested class, and its
  counts and its total of 340 described that older layout. Each row now
  gives the number of files split from its original, and the counts are
  what pytest collects from them: 435 tests in all.

- The Halton source's formal job `hdl/formal/halton16.sby` asserts the two
  properties its harness listed and did not check: the counter steps by
  one on every enabled clock, and the output is the bit reverse of the
  count before that clock, one register stage behind, held while enable is
  low. A FORMAL-only port `formal_counter` gives the harness the counter. A
  counter that skips a value fails the proof. The formal inventory is 91
  jobs and 510 statements (340 assert, 123 assume, 47 cover).

- The signal decorrelator `hdl/sc_decorrelator.v` rotates each stream by
  its shift amount modulo the word width. The 4-bit shift amount reaches
  15, so for words narrower than 16 bits, such as the 8-bit words of its
  testbench and of the Verilog generator's
  `emit_decorrelator(stream_width=8)`, a shift of the width or more pushed
  every bit out and the stream became zero instead of a rotation. Words of
  16 bits behave as before. Its formal job `hdl/formal/decorrelator.sby`
  held no assertion, and formal-all on v4.1.0 refused it ("Prepared proof
  has zero assertions"). It now proves by induction, for the default
  instance, the 8-bit one and a 12-bit one, that each output word is the
  previous source word rotated by (stream index XOR phase XOR seed nibble)
  modulo the width (so it keeps the source word's population count), that
  reset clears it, and that the eight default streams give distinct words
  for a one-hot source word; a cover reaches a shift of 8 or more on the
  8-bit instance. A FORMAL-only port `formal_phase_counter` gives the
  harness the phase counter. The module comment no longer calls the
  registered output zero-latency. The formal inventory is 91 jobs and 508
  statements (338 assert, 123 assume, 47 cover).

- The internal `core_engine` crate's `cordiv` and its C function
  `sc_cordiv_packed` follow the package's CORDIV contract. They copied the
  numerator only when the denominator bit was 1, so on a numerator bit of 1
  over a denominator bit of 0 they held the previous output where
  `sc_divide`, the engine's `cordiv` and `hdl/sc_cordiv.v` output 1. The
  crate's test claimed convergence for decorrelated streams and checked
  only that the output lay between 0.1 and the denominator's density. Three
  tests now pin the eight-row table, the quotient from one LFSR read once
  every 16 steps, and the numerator returned bit for bit by one LFSR read
  every clock. No Python code calls the C function.

- The CORDIV divider's documentation states its operating condition and
  its source. `sc_divide`, `hdl/sc_cordiv.v`, the engine's `cordiv`, the
  hardware contract page and the utilities page said the output converges
  to P(x) / P(y) whenever P(x) <= P(y), and the pages added that correlated
  streams bias it. The reverse holds: the quotient needs both streams
  compared against one shared random sequence, and independent streams
  converge to P(x) / (P(x) + P(y) - P(x) P(y)), about 0.417 instead of 0.5
  for 0.3 and 0.6. The contract page's example now draws both streams from
  one sequence. The citation, given as Li, Qian, Riedel and Bazargan in
  IEEE Trans. Signal Process. 62(9), 2014, a record we could not find, is
  now Chen and Hayes, IEEE ISVLSI 2016 (doi:10.1109/ISVLSI.2016.48), the
  CORDIV paper. The quotient also needs a sequence whose successive values
  are independent: with both streams compared against one 16-bit LFSR read
  every clock, as `hdl/sc_bitstream_encoder.v` is, the output misses the
  quotient and equals the numerator bit for bit whenever 2 P(x) <= P(y)
  (0.303 instead of 0.502 for 0.3 and 0.6); the contract page says so.
  Two tests pin both regimes. The convergence test, which measured
  independent streams against the quotient, now measures them against
  their limit and requires an error below 0.02 at 4,096 bits (0.008
  measured; against the quotient it stays near 0.08). The pages cite the
  test files that exist since the CORDIV tests were split. The circuits
  are unchanged.

- SNN Studio: two populations added before the server answered the first
  could get the same label and position, because each add read the number
  of populations when it was made (two "+ Exc" could both give "Exc 0").
  Adds now run one after another, each reading the count once the earlier
  ones are applied, and an add the server refuses spends no label.

- A surrogate's parameter must be finite and above 0. The Python classes
  SurrogateLif and DifferentiableDenseLayer, native and wrapped, raise
  ValueError for a steepness k of 0, a negative k, NaN or infinity, which
  gave an infinite, negative or NaN gradient. In Rust, SurrogateType gains
  constructors that refuse such a k or width (fast_sigmoid, super_spike,
  arc_tan, triangular, piecewise_linear) and a check; SurrogateLif::try_new
  and the new DifferentiableDenseLayer::try_new refuse a surrogate built
  directly, and SurrogateLif::new and DifferentiableDenseLayer::new panic
  with the refusal's words, as documented, where 4.1.0 built them.
  DifferentiableDenseLayer::try_update_weights returns an error for
  gradients of the wrong shape, which update_weights ignored silently;
  update_weights keeps its 4.1.0 signature and panics with those words,
  and the Python binding's messages are unchanged.

- The hardware-in-the-loop debugger refuses inputs it used to misread
  silently, in its Python client and its Go daemon alike. A trigger
  threshold that is negative or not finite, which disabled its test as
  0 does, is refused (at construction in Python, by Validate in Go); 0
  stays the one value that disables a test. A negative refill, which
  could leave the token bucket below zero, is refused and changes
  nothing; Refill in Go now returns an error.

### Removed

- `crates/neuro_symbolic/` drops seven Python files, deleted on the owner's
  approval: an older copy of `sc_neurocore.neuro_symbolic.predictive_coding`
  with the same logic but no docstrings or types, its support module and
  five test files. Each test file has the same test names as its
  counterpart in `tests/test_neuro_symbolic/`, and pytest never collected
  them. The Rust crate is unchanged.

- The accelerated lanes drop 120 transcripts of application code, deleted on
  the owner's approval of the listed set: 54 Mojo and 66 Julia files that
  copied Python modules which are not compute kernels (web and command-line
  applications, network I/O, records, re-exports and demonstrations) into
  functions returning zero. No loader read any of them. The stub triage and
  the Mojo and Julia lane inventories now list the 610 transcripts that remain,
  608 of compute kernels and 2 of the one original still unclassified.
- The mirror-authority declarations no longer name the deleted `app.mojo`, and
  a test requires every declared mirror pattern to name files of the tree.
  The safety-certificate and evolution-substrate architecture tests no longer
  expect the deleted monitor and substrate transcripts; the Rust and Go
  monitors remain.
- The evolution-substrate multi-language benchmark no longer binds the
  deleted Mojo transcript, which it never ran, and was measured again in all
  five lanes; the API page shows the new medians.

### Repository hygiene

- The 54 top-level test modules of the AdEx, AER, alternative-path,
  astrocyte, augmentation, auto-fit, benchmark, bio and biophysical
  suites import their names explicitly instead of through a
  suppressed star import, gain their missing docstrings and type
  annotations and are enrolled in the docstring policy. Seven support
  modules left without an importer are deleted; three that keep
  importers gain types and module docstrings.

- The 29 safety-certification test modules that star-imported a
  support module (`tests/test_safety_cert`) import their names
  explicitly instead, gain their missing test and module docstrings
  and are enrolled in the docstring policy. The seven supports, whose
  `_unsafe` helpers are still imported, stay as they are.

- The 30 bio-hybrid test modules that star-imported a support module
  (`tests/test_bioware`) import their names explicitly instead, gain
  their missing test docstrings and are enrolled in the docstring
  policy. The encoding, experiment, plasticity and validation-contract
  supports, left without an importer, are deleted; the five supports
  whose helpers are still imported stay as they are.

- The 29 evolutionary-substrate test modules that star-imported a
  support module (`tests/test_evo_substrate`) import their names
  explicitly instead, gain their missing test docstrings and are
  enrolled in the docstring policy. Their nine support modules, left
  without an importer, are deleted.

- The 33 debug and runtime test modules that star-imported a support
  module (`tests/test_debug`) import their names explicitly instead,
  gain their missing test docstrings and type annotations and are
  enrolled in the docstring policy. The photonic-emitter graph
  stand-ins gain types and docstrings, and the runtime and wave-4
  supports, left without an importer, are deleted.

- The 42 bridge test modules that star-imported a support module
  (`tests/test_bridges`: DNA mapper, photonic NoC and annealing
  backends) import their names explicitly instead, gain their missing
  test docstrings and are enrolled in the docstring policy. The seven
  fixtures the DNA-mapper and photonic supports handed out through the
  star import move to `tests/test_bridges/conftest.py`, the one helper
  of the annealing support moves to its only user, and the three
  supports, left without an importer, are deleted.

- The 13 ASIC-flow test modules that star-imported a support module
  (`tests/test_asic_flow`) import their names explicitly instead,
  gain their missing test docstrings and are enrolled in the docstring
  policy. Their four support modules, left without an importer, are
  deleted.

- The 22 federated stochastic-computing test modules
  (`tests/test_federated`) import their names explicitly instead of
  through a suppressed star import, gain their missing test docstrings
  and type annotations and are enrolled in the docstring policy. Their
  support module, left without an importer, is deleted.

- The 21 twin-synchronisation test modules (`tests/test_twinsync`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. Their support module, left
  without an importer, is deleted.

- The 22 spintronic-mapper test modules (`tests/test_spintronic`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. Their support module, left
  without an importer, is deleted.

- The 21 surrogate-gradient training test modules (`tests/test_training`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. They take every name their
  support module re-exports from that module, so its torch skip still
  runs before anything that needs torch.

- The 18 meta-plasticity test modules (`tests/test_meta_plasticity`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. Their support module, left
  without an importer, is deleted.

- The 18 sustainability-profiler test modules
  (`tests/test_energy_accounting`) import their names explicitly
  instead of through a suppressed star import, gain their missing test
  docstrings and return annotations and are enrolled in the docstring
  policy. Their support module, left without an importer, is deleted.

- The 17 explainability test modules (`tests/test_explainability`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. Their support module, left
  without an importer, is deleted.

- The 16 edge-deployment and tiny-SC-port test modules (`tests/test_edge`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and type annotations and
  are enrolled in the docstring policy. Both support modules are left
  without an importer and are deleted.

- The 16 memristor-mapper test modules (`tests/test_memristor`) import
  their names explicitly instead of through a suppressed star import,
  gain their missing test docstrings and are enrolled in the docstring
  policy. Their support module, left without an importer, is deleted.

- The 16 FPGA-hypervisor, isolation, QoS and region test modules
  (`tests/test_hypervisor`) import their names explicitly instead of
  through a suppressed star import, gain their missing test docstrings
  and type annotations and are enrolled in the docstring policy. The
  isolation support module, left without an importer, is deleted.

- The 12 UVM testbench-generator test modules (`tests/test_uvm_gen`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and return annotations and
  are enrolled in the docstring policy.

- The 11 optimizer and hardware-observation test modules
  (`tests/test_optimizer`) import their names explicitly instead of
  through a suppressed star import, gain their missing module and test
  docstrings and type annotations and are enrolled in the docstring
  policy. Optimizer results the tests read are narrowed with
  `assert ... is not None`, the fake Rust engine is built without type
  suppressions, and the optimizer support's helpers are typed.

- The 12 Arcane-Zenith cognitive-core test modules
  (`tests/test_arcane_zenith`) import their names explicitly instead of
  through a suppressed star import, gain their missing test docstrings
  and type annotations, including typed fixture parameters, and are
  enrolled in the docstring policy. Their support module, left without
  an importer, is deleted.

- The ten SC neural-architecture-search test modules (`tests/test_nas`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings and are enrolled in the
  docstring policy.

- The 12 sensor-fusion test modules (`tests/test_fusion`) import their
  names explicitly instead of through a suppressed star import, gain
  their missing test docstrings and return annotations and are enrolled
  in the docstring policy.

- The 11 compiler-export, ONNX-export and TVM-lowering test modules
  (`tests/test_export`) import their names explicitly instead of
  through a suppressed star import, gain their missing test docstrings
  and type annotations and are enrolled in the docstring policy. The
  TVM lowering support module's mock nodes and graphs are typed and
  documented like the ONNX export support's.

- The eight photonic crosstalk and FDTD test modules
  (`tests/test_optics`) import their names explicitly instead of
  through a suppressed star import, gain their missing test docstrings
  and type annotations, including typed fixture parameters and typed
  Meep fakes, and are enrolled in the docstring policy. Both support
  modules are left without an importer and are deleted.

- The ten native array-guard and core-engine-bridge test modules
  (`tests/test_native`) import their names explicitly instead of
  through a suppressed star import, gain their missing module and test
  docstrings and type annotations and are enrolled in the docstring
  policy. The array-guard support module, left without an importer, is
  deleted.

- The 11 BCI-studio and predictive-coding test modules
  (`tests/test_bci`, `tests/test_neuro_symbolic`) import their names
  explicitly instead of through a suppressed star import, gain their
  missing test docstrings and type annotations and are enrolled in the
  docstring policy. Both support modules are left without an importer
  and are deleted.

- The 16 fault-injection and model-zoo test modules
  (`tests/test_fault_injection`, `tests/test_model_zoo`) import their
  names explicitly instead of through a suppressed star import, gain
  their missing test docstrings and type annotations and are enrolled
  in the docstring policy. The fault-injection validation tests pass
  their deliberately wrong arguments through `cast` instead of under
  type suppressions, and the model-zoo support module, left without an
  importer, is deleted.

- The 13 HDL-generation and world-model test modules
  (`tests/test_hdl_gen`, `tests/test_world_model`) import their names
  explicitly instead of through a suppressed star import, gain their
  missing module and test docstrings and type annotations and are
  enrolled in the docstring policy. The Kuramoto configuration test
  checks its refusal with `pytest.raises` instead of an excluded
  branch, and the Kuramoto RTL support module, left without an
  importer, is deleted.

- The 19 star-import test modules of five small suites
  (`tests/test_analog_bridge`, `test_integration`, `test_interfaces`,
  `test_learning`, `test_stochastic_doctor`) import their names
  explicitly, gain their missing test docstrings and type annotations
  and are enrolled in the docstring policy. The analog-bridge export
  test names the objects it checks instead of reading them from the
  module globals, and the four support modules left without an
  importer are deleted.

- The 15 tool test modules under `tests/test_tools` that still took
  their names from a suppressed star import (build accel backends,
  criterion to bencher, extract ORCA params, readiness evidence index,
  SNN memory-discipline audit) import them explicitly, gain their
  missing module and test docstrings and are enrolled in the docstring
  policy. The readiness evidence index tests define their module-scoped
  `tool` fixture themselves instead of receiving it through the star
  import.

- The behavioural-equivalence support module
  (`tests/behavioral_equivalence_support.py`), whose four importers
  now import their names explicitly, is deleted.

- The 13 hierarchical-partitioner graph and reporting test modules
  (`tests/test_chiplet/test_hierarchical_partitioner_*`) import their
  names explicitly instead of through a suppressed star import, gain
  their missing test docstrings and are enrolled in the docstring
  policy. The graph and reporting support modules, which no test
  imports any more, are deleted.

- The 15 hierarchical-partitioner core, backend, runtime-cache and
  metrics test modules (`tests/test_chiplet/test_hierarchical_partitioner_*`)
  import their names explicitly instead of through a suppressed star
  import, gain their missing test docstrings, drop their remaining lint
  and type suppressions and are enrolled in the docstring policy. The
  backend, metrics and runtime-cache support modules, which no test
  imports any more, are deleted.

- `engine/tests/test_hdc.rs` regains the layout rustfmt requires: the
  header edit had left two blank lines before its module comment, which
  the engine's `cargo fmt --check` step refuses.

- The 17 chiplet generator, routing and topology test modules
  (`tests/test_chiplet/test_chiplet_*`) import their names explicitly
  instead of through a suppressed star import, gain their missing test
  docstrings and are enrolled in the docstring policy. The routing and
  topology support modules, which no test imports any more, are
  deleted.

- The nine vmin LIF LUT generator test modules
  (`tools/test_gen_vmin_lif_lut_*`) import their names explicitly
  instead of through a suppressed star import, gain their missing test
  docstrings and are enrolled in the docstring policy. Their support
  module, which no test imports any more, is deleted. With every
  `tools/test_*.py` module now importing explicitly, the ruff
  per-file exemption from F403 and F405 for `tools/test_*.py` is
  removed from `pyproject.toml`.

- The six SHD weight-extraction tool test modules
  (`tools/test_extract_shd_weights_*`) import their names explicitly
  instead of through a suppressed star import, gain their missing test
  docstrings and are enrolled in the docstring policy.

- The 15 equivalence test modules (`tests/equivalence/test_encoder_equiv_*`,
  `tests/test_behavioral_equivalence_*`, `tests/test_equivalence_check_*`
  and `tests/test_equivalence_miter_*`) import their names explicitly
  instead of through a suppressed star import, and the encoder modules
  import their support module as a package instead of through a
  `sys.path` entry. They gain the missing test docstrings and return
  annotations and are enrolled in the docstring policy.

- The 20 end-to-end test modules (`tests/e2e/test_e2e_pipeline_*` and
  `tests/test_qc_e2e_*`) import their names explicitly instead of
  through a suppressed star import, so ruff checks them without a
  `noqa: F403`. They gain the missing test docstrings and are enrolled
  in the docstring policy.

- The engine monolith guard is replaced by a rule for where Python
  bindings live. `tools/engine_bindings_location.py --check` refuses any
  `#[pyfunction]` in a tracked `engine/src` Rust file outside a binding
  module (a `bindings` directory or a `bindings.rs` file), for every
  engine file rather than a fixed list. The DOWN-only line ceilings,
  `tools/engine_monolith_ceiling.toml` and their tests are removed; files
  are divided by responsibility, not by size. The CI step and the
  pre-commit hook run the new rule.

- The RK4 neuron simulator's Python binding, `py_rk4_neuron_simulate`
  and its trace helpers, moved out of `engine/src/rk4_neurons.rs` into its
  binding module `engine/src/rk4_neurons/bindings.rs`, which the extension
  registers as before. The model file now holds only the integrators.
  Behaviour and the Python name are unchanged.

- The three co-simulation ownership test modules import their names
  explicitly instead of through a suppressed star import, so ruff checks
  them without a `noqa: F403`. They gain docstrings and are enrolled in
  the docstring policy.

- The ASIC-flow architecture test no longer caps the package modules at
  350 lines or its focused tests at 250. In its place,
  `test_each_module_is_declared_and_defines_exactly_what_it_owns`
  requires every package module to be declared and each to define only
  its owned symbols and declared private helpers, where the ownership
  test before checked only the historical public symbols.

- The safety-certification architecture test no longer caps the
  compatibility module at 120 lines, the ten domain modules at 600 or
  the safety-certification tests at 500. In its place,
  `test_each_module_defines_exactly_its_owned_classes_and_the_facade_none`
  holds each domain module to its owned classes and declared private
  helpers and the compatibility module to no definitions, where the
  ownership test before checked object identity only.

- The hardware neuron-graph architecture test no longer caps the
  neuron-graph modules at 330 lines, the NIR bridge tests at 300 or the
  graph benchmark and its probe at 380 and 240. In its place,
  `test_each_responsibility_module_defines_exactly_its_owned_symbols`
  requires every `neuron_graph*.py` module to be declared and each to
  define exactly the symbols it owns, where the ownership test before
  checked only the established symbols.

- The hypervisor isolation test module no longer caps the isolation
  module at 120 lines or the historical `hypervisor` module at 900. Its
  ownership check stays: the isolation module defines exactly its two
  classes and one function, and the historical module none of them. The
  module imports its names directly instead of a suppressed star import
  through a `sys.path` entry, gains docstrings and is enrolled in the
  docstring policy.

- The hypervisor tenant test no longer caps the tenant module at 90
  lines or the historical `hypervisor` module at 785. Its ownership
  check stays: the tenant module defines exactly its four classes and
  the historical module none of them. The test file gains docstrings
  and is enrolled in the docstring policy.

- The hypervisor scheduler test no longer caps the scheduler module at
  125 lines or the historical `hypervisor` module at 689. Its ownership
  check stays: the scheduler module defines exactly its three classes
  and the historical module none of them. The test file gains
  docstrings and is enrolled in the docstring policy.

- The hypervisor compliance-audit test no longer caps the audit module
  at 80 lines or the historical `hypervisor` module at 850. Its ownership
  check stays: the audit module defines exactly its three classes and
  the historical module none of them. The test file gains docstrings and
  is enrolled in the docstring policy.

- The quantum-annealing architecture test no longer caps the facade at
  120 lines, the responsibility modules at 400 or the focused tests at
  500. In its place,
  `test_modules_own_their_declared_definitions_and_the_facade_defines_none`
  holds each of the ten modules to its declared top-level classes and
  functions, with each public name defined in one module and none in
  the facade.

- The co-simulation support architecture test no longer caps the facade
  and its 29 owner modules by length. In its place,
  `test_owners_are_facade_independent_and_the_facade_defines_nothing`
  checks every owner, the statistics module included, for a facade
  import, and holds the facade to a docstring and imports. Each owner's
  exact definitions stay pinned by the ownership tests. The test also
  replaces its suppressed star import with explicit imports.

- The Studio jobs architecture test no longer caps the jobs modules and
  their focused tests at 300 lines. In its place,
  `test_studio_jobs_modules_own_exactly_their_declared_definitions` holds
  each of the 34 modules to its declared top-level classes and functions,
  with no name defined twice and none in the facade. The unused
  `JOBS_TEST_PATHS` list is removed from the test support module.

- The photonic emitter architecture test no longer caps the private modules
  at 350 lines or the facade at 100. It keeps its contract, renamed
  `test_responsibility_modules_have_one_way_imports_and_a_definition_free_facade`:
  each private module imports only its declared siblings, and the facade
  defines no class or function.

- The photonic crosstalk lane test no longer caps each lane source at 350
  lines. It keeps what it is for, renamed
  `test_polyglot_sources_are_real_numeric_kernels`: no lane source may carry
  a transcript or no-op marker.

- The neuron-module architecture tests no longer cap file lengths: the Rust
  map models and per-model contract tests at 300 lines, and the Aihara,
  adaptive-threshold and shared-runtime Julia files at 500, 150, 200 and 100.
  Each test already holds its module's ownership: one struct per Rust map
  file, one test class per contract file, and which module defines each
  executor. That ownership is what remains.

- The autonomous-learning bridge's architecture tests no longer cap file
  lengths (300 lines per module, 120 for the facade), following the owner's
  rule that files are divided by responsibility, never by size. They now
  hold what the caps stood for. Each divided module defines exactly the
  top-level names of its declared concern, and no name is defined twice.
  The facade defines no class and only its three delegating functions.
  A stray definition in a module, which the line caps did not catch, now
  fails.

- The evolution-substrate facade test no longer caps the facade at 180
  lines. The facade already defines no class or function and exports its 56
  historical names, so that contract is what remains.

- The hierarchical partitioner's architecture tests no longer cap line
  counts (120 for the facade, 300 per split module and per test file). They
  keep the facade's single `__getattr__`, one owner per public symbol and the
  documented dependency graph. They now also hold:
  - each split module to the top-level names of its declared concern;
  - every `test_hierarchical_partitioner_*` file to the concern it covers.

- The chiplet architecture tests no longer cap line counts: 150 for the
  facade, 500 per responsibility module and test file, 100 for `_sv` and 300
  for each RTL emitter. They keep the existing ownership of every public
  symbol and the fixed import graph, and now also hold:
  - each private RTL helper to the top-level names of its declared concern;
  - every `test_chiplet_*` file to the concern it covers.
  The emitters' 100-character line width, a formatting rule, is kept.

- The DNA mapper's architecture tests no longer cap line counts (250 for the
  facade, 650 per responsibility module). They now hold what the caps stood
  for:
  - each of the eleven responsibility modules defines exactly the top-level
    names of its declared concern, and no name is defined twice;
  - the facade defines nothing at all, private names included.
  The responsibility import graph stays acyclic.

- The Bioware architecture tests no longer cap line counts: 300 per source
  and test file, 350 for the benchmark and 200 for its probe. They now hold
  the responsibility the caps stood for:
  - each responsibility module defines exactly the top-level names of its
    declared concern, and no name is defined twice;
  - every test and support file names the module whose concern it covers,
    except the two package-wide ones.
  The benchmark and probe caps are removed. The existing source-binding test
  already holds them to their sources.

- Tests that hold a contract about a CI job read it through
  `tools/ci_workflow_inventory.py`, which finds the job, its workflow's text
  and its workflow-level environment in whichever CI workflow declares it, and
  refuses a job declared twice. Thirty test files no longer open
  `.github/workflows/ci.yml` themselves, so the workflow can be divided by
  responsibility without rewriting each contract. Contracts on the
  workflow-level `SC_NEUROCORE_REQUIRE_*` flags now require them in every CI
  workflow, since a called workflow does not inherit its caller's
  environment.

- The engine takes z3 0.23 (from 0.22); z3-sys and its bundled solver stay
  at 0.14.2. The fuzz workspace's lock follows the engine to z3 0.23 and
  gains the zstd, flate2 and libz-sys packages the engine has needed since
  its waveform containers, at the engine lock's versions.

- `reuse lint` passes again (REUSE 3.3): two assertions in
  `tests/test_shd_event_core.py` held the license tag text in a string,
  which REUSE parsed as two invalid license expressions. The tag is now
  joined at run time. The same test lists the event core's files from Git,
  so an ignored simulation or synthesis output in `hdl/shd_event/` no longer
  fails it in a working checkout.

- Every module that loads a Mojo shared library is now held to the public
  docstring policy (66 more files, 2141 in all). Ten neuron-model modules
  and one Mojo package gain module docstrings, and 51 constructors, steps,
  resets, accessors and Mojo batch functions gain docstrings. No code
  changes. Two benchmark records whose bound sources changed only in
  docstrings are kept with a recorded review.

- Every JSON fixture under `tests/fixtures/` carries its ownership in a
  `.license` sidecar with the licence and both copyright records. Four
  recorded Grype advisory matches and the NIR corpus manifest had none, and
  five sidecars gave one copyright record instead of two. A policy test
  now holds every fixture JSON to the sidecar.

- `engine/.cargo/config.toml` no longer says that its `x86-64-v2` target is
  for portable wheels. Cargo reads it only for builds started in `engine/`
  (benchmarks and fuzzing). Every wheel build starts in `bridge/`, where no
  configuration sets a target, so the published wheels have always been
  compiled for baseline x86-64, and they stay so.

- SNN Studio's store builds its graph-editing actions (populations,
  projections, their undo history, validation, simulation, and the NIR and
  notebook files) in a module of their own, studioGraphActions.ts, instead
  of in the factory that also holds sign-in, audit, identity, training and
  project actions. The code moves unchanged, and new tests cover it fully;
  further tests cover the server project and share-link actions that stay.
- The operator administration actions of SNN Studio's store (the audit
  log and its quarantine archives, evidence bundles, job and operator
  status, and identity accounts) move unchanged to studioAdminActions.ts,
  covered fully by new tests.
- The training actions of SNN Studio's store (surrogates, target
  profiles, the job list, selecting, starting and stopping a run,
  following its event stream, and its checkpoints and weights) move
  unchanged to studioTrainingActions.ts, with the stream and the counters
  that outdate late replies. Signing out reaches them through one release
  call. New tests cover the module fully.
- The sign-in actions of SNN Studio's store (the stored token's session
  check, signing in and signing out) move unchanged to
  studioAuthActions.ts. Sign-out receives the training module's release
  call as a parameter instead of reading it from the store factory. New
  tests cover the module fully.
- SNN Studio's store moves its server project actions (saving against a
  revision, keeping a refused edit, loading, listing, deleting and
  restoring) unchanged to studioProjectActions.ts, and its saved session
  and share link actions unchanged to studioSessionAndShareActions.ts.
  Both receive the store's restored-experiment step as a parameter. New
  tests cover both modules fully, and further tests cover the workspace
  setters that stay.
- The hardware actions of SNN Studio's store (compiling the model to RTL,
  co-simulating it, choosing the synthesis target, single- and
  multi-target synthesis, the resource estimate and the tool check) move
  unchanged to studioHardwareActions.ts. New tests cover the module
  fully, including each refusal to route a catalogue model whose RTL
  lacks current bit-exact co-simulation parity.
- The simulation and analysis runs of SNN Studio's store (the run and
  its automatic repeat after an edit, the f-I curve, bifurcation,
  sensitivity, precision, heatmap, characterisation, several models side
  by side, the network, an imported trace, comparison, nullclines,
  frequency response and the spike-triggered average) move unchanged to
  studioRunActions.ts, covered fully by new tests.
- The experiment exports of SNN Studio's store (the Python script with
  its one-liner, replay script and digest, the replay pack and notebook,
  and the trace as JSON, CSV or SVG) move unchanged to
  studioExportActions.ts, covered fully by new tests. Repeated blank lines
  the earlier moves left in the store's factory file are removed.
- The network pipeline run of SNN Studio's store moves unchanged to the
  graph module, studioGraphActions.ts, beside the other network actions,
  and new tests cover it.
- The experiment workspace of SNN Studio's store (the inputs, the
  template, model and preset catalogues and the choices among them, and
  the return to a model's defaults), with the counters that outdate late
  replies, moves unchanged to studioWorkspaceActions.ts. The store's
  factory now only composes its action modules, apart from loading the
  host's capabilities and recording how each stage ended. An empty import
  it had kept since July is removed. New tests cover what nothing tested
  before: loading the template catalogue, a library failure that is not
  an Error, and a preset of each kind, so the new module is covered fully.

- SNN Studio's live browser test of constructor types moved AdEx's initial
  membrane voltage by ten slider steps, but a floating-point slider has
  had no numeric step since October ("any"), so the move came to NaN, the
  slider stayed put and the test waited four minutes for a run that was
  never sent. It now moves the value a tenth of the way toward the
  slider's farther bound and checks that the move is finite and real. The
  continuous-integration browser step now runs the whole
  catalogue-to-silicon live suite, so this test, and the one after it that
  never ran once it failed, can no longer go stale unseen.

- SNN Studio's audited TypeScript project, the typed program its linter
  reads for every file outside `src`, did not type-check: the graph jobs
  panel it reaches imports a stylesheet, and without Vite's types, which
  only the build's project included, TypeScript refused that import. The
  audited project now includes them, and the continuous-integration
  frontend job type-checks it on its own, which no step did before.

- A test of SNN Studio's store no longer spells out a password: the
  rotated password it sends is built at run time, so the secret scan has
  no literal in it to report as a credential.

- Every Rust crate under `crates/` now has a formatting gate:
  `tools/crate_format_check.py` runs `cargo fmt --check` on each crate, in
  the CI Rust job and in the preflight for a change under `crates/`. The
  engine and the documentation symbol producer had gates of their own; the
  seven crates had none, and two had drifted. `core_engine` (two asserts and
  its arithmetic benchmark) and `neuro_symbolic` are formatted, with no change
  of behaviour.

- The engine crate's public API is compared with its release on crates.io in
  a new CI job, `engine-semver`. `tools/engine_semver_check.py` runs
  `cargo semver-checks` 0.51.0 and fails on an incompatible change outside a
  major release:
  - **Between releases:** the published version named in
    `engine/Cargo.toml` is the baseline, and the change is classed as minor.
  - **At the release bump:** the highest published version below the new one
    is the baseline, and the version numbers set the release type.
  - **Never a pass:** a check that cannot run fails.

  The 4.2.0 release gate compared only public functions and type shapes, and
  missed traits, impls, attributes and re-exports. The `gpu` feature is
  compared. The `supervisor` module behind the default `z3` feature is not,
  because the bundled Z3 build would dominate the job, and each run says so.

## [4.2.0] - 2026-11-17

The curated summary of this release is `docs/development/RELEASE_NOTES_v4.2.0.md`, which is also the GitHub
Release text.

### Security

- The toolchain manifest's `apt install` pattern now reads each option one
  way only (`-o` with its value, or another word that starts with a dash).
  An option two branches could both match made a line of options that
  never reaches `install` take exponential time to refuse (CodeQL
  `py/redos`, error); 22 copies of ` -o -!` took 0.9 s, each 2 more four
  times as long. The matches over every workflow are unchanged, and a test
  bounds the refusal of 40 copies in a child process with a time limit.
- Move tornado in the optics and workstation dependency locks from 6.6 to
  6.6.1 for CVE-2026-126347 (GHSA-5cjw-f85j-cv6r): tornado 6.6.0's fix for an
  open redirect in `RedirectHandler` missed URLs that begin with three or more
  slashes. Only tornado moves; each lock is regenerated with the command its
  header records, with the 43 file hashes PyPI lists for 6.6.1.
- Move Go from 1.27.3 to 1.27.4 in every module and workflow for
  GO-2026-6859 (CVE-2026-97054), an HTTP/1 server crash in `net/http` that
  the locked-dependency audit reported for all five Go modules.
- The Rust documentation ratchet refuses an original-image folder named
  through a link, and reads the folder's owner and mode from the folder it has
  opened, so the folder it checks is the folder it locks and uses.

### Added

- The benchmark coverage census accepts a recorded reason for not
  measuring a record again after a bound source changed, in
  `benchmarks/source_change_reviews.toml`, only when the change touched
  documentation or comments. The source must still have the reviewed
  bytes, and its code, compared without docstrings and comments
  (`tools/source_code_equivalence.py`, Python and Rust), must equal the code
  of the bytes the record measured, found in the repository's history. Such
  a record counts as `reviewed`, not `current`; a review of a code change
  is rejected and shown, and a review that applies to nothing is listed.
  The KL refinement record, whose partitioner changed only in Rustdoc
  comments, is the first one reviewed.

- The Mihalas-Niebur neuron states two behaviour cases, the first that set
  parameters of their model: with panel M's current jumps (0.01 and
  -0.0006, the ones its reference receipt and RTL test run), a current of
  0.002 fires 8 times in 400 steps, its first burst, and 14 times in 2000
  steps, in two bursts, measured with its public class and matched by the
  schema. The class as built by default leaves both jumps at 0 and fires 2
  and 10 times over those windows. Each of the 14 sign variants of its four
  equations breaks the first case; without the jumps the four terms that act
  through the internal currents would hold every case. Nineteen of the 185
  catalogue models now state cases.
- A model descriptor's behaviour case may set parameters of the model it
  runs, in a `[behaviour.parameters]` table. Each must be a declared
  parameter with a declared range, set to a finite number within that range,
  both ends included. A state variable, an undeclared name, a parameter
  without a range and a value outside it are refused, naming the value, the
  range and the parameter. The behaviour report builds the model with them,
  for both drives of a contrast, and shows them next to the inputs; the
  variants report sets them on the schema and on every variant, and a schema
  that does not declare one differs from the model. A case that sets none
  keeps its digest, so the reports were unchanged by the format.
- The Wilson-HR neuron states a sixth behaviour case: 1 upward crossing in
  20 steps at a current of 0.1, measured with its public class and matched
  by the schema. The window ends after v's first upward crossing of v_peak
  and before it falls back, so a step that counted downward crossings would
  give none. Each of the five windows it stated before holds as many
  downward crossings as upward ones.
- The Pernarowski neuron states a sixth behaviour case: 1 upward crossing
  in 160 steps at a current of 0.2, measured with its public class and
  matched by the schema. The window ends after v's first upward crossing of
  the threshold and before it falls back, so a step that counted downward
  crossings would give none. Each of the five windows it stated before
  holds as many downward crossings as upward ones.
- The Hindmarsh-Rose neuron states a sixth behaviour case: 1 upward
  crossing in 2000 steps at a current of 20.0, measured with its public
  class and matched by the schema. At that drive x crosses the threshold
  once and is held above it, so a step that counted downward crossings
  would give none. Each of the five windows it stated before holds as many
  downward crossings as upward ones, and so does each window its Q16.16
  test enrols.
- The FitzHugh-Rinzel neuron states a fifth behaviour case: 1 event in
  10000 steps at a current of 2.0, the window its dynamics test enrols for
  the public class; the schema matches. v crosses the threshold upward once
  and stays above it, so a step that counted downward crossings would give
  none. Each of the four windows it stated before holds as many downward
  crossings as upward ones.
- The SC adaptive-threshold map neuron states two behaviour cases: 3 events
  over the 32-step window at a current of 0.4 that its Q8.24 test runs for
  the hand model and the RTL, which must give the same events, and 4 events
  over 38 steps at 0.4, measured with its public class and matched by the
  schema, a window that ends on an upward crossing. Each of the 9 sign
  variants of its two equations breaks a case. Eighteen of the 185 catalogue
  models now state cases.
- The SC clipped-logistic bursting map neuron states four behaviour cases:
  the event counts over the 30-step windows its Q16.16 test runs for the
  hand model, the schema and the RTL (2, 1 and 1 at currents 0.5, 1.0 and
  2.0), and the 7 events in 30 steps at 0.05 that its sensitive-boundary
  test enrols for the hand model. Each of the 10 sign variants of its two
  equations breaks a case. Seventeen of the 185 catalogue models now state
  cases.
- The upward-crossing SC Rulkov map neuron states five behaviour cases: the
  event counts over the 30-step windows its Q16.16 test runs for the hand
  model, the schema and the RTL (none, 7 and 10 at currents 0, 0.5 and
  1.5), 11 events in 200 steps at 0.2, and 11 events in 31 steps at 1.5, a
  window that ends on an upward crossing; the last two are measured with
  its public class and matched by the schema. Each of the 17 sign variants
  of its two equations breaks a case. Sixteen of the 185 catalogue models
  now state cases.
- The Ermentrout-Kopell map neuron states four behaviour cases: the event
  counts its Q16.16 test enrols for the hand schema and the RTL (none, 45
  and 64 in 2000 steps at currents -0.5, 0.5 and 1.0), and 5 events in 1111
  steps at 0.02, measured with its public class and matched by the schema.
  Each of the 8 sign variants of its phase map breaks a case. Fifteen of the
  185 catalogue models now state cases.
- The inclusive-threshold perfect integrator
  (`SCInclusivePerfectIntegratorNeuron`) states two behaviour cases: 500
  events in 1000 steps at a current of 5, the count its source receipt test
  enrols for this class, where every second candidate equals the threshold
  exactly and an inclusive comparison fires (a strict one fires 333); and 500
  events in 2000 steps at 3, measured with its public class and matched by
  the schema, with every candidate at least 0.1 from the threshold. Both
  sign variants of its step break a case. Fourteen of the 185 catalogue
  models now state cases.
- The Pernarowski neuron states five behaviour cases: the crossing counts its
  Q16.16 test enrols for the hand model, the schema and the RTL (17 in 5000
  steps at each of the currents -0.1, 0, 0.1 and 0.2), and 54 crossings in
  16049 steps at -0.1, measured with its public class and matched by the
  schema. Each of the 14 sign variants of its three equations breaks a case.
  Thirteen of the 185 catalogue models now state cases.
- The FitzHugh-Rinzel neuron states four behaviour cases: the event counts its
  Q16.16 test enrols for the hand model, the schema and the RTL (7, 8 and 8 in
  3000 steps at currents 0.4, 0.5 and 0.6), and 83 events in 39600 steps at
  0.4, measured with its public class and matched by the schema. Each of the
  13 sign variants of its three equations breaks a case. Twelve of the 185
  catalogue models now state cases.
- The Courbage-Nekorkin map neuron (`CourageNekorkinMapNeuron`) states six
  behaviour cases: the four event counts its Q16.16 and Q32.32 tests enrol for
  the hand model, both schema files and the RTL (none in 128 steps at a current
  of -0.3, none in 512 and one in 620 without a current, one in 128 at 0.3);
  3 events in 41 steps at 0.1, measured with its public class and matched by
  the schema; and its fast variable after those 41 steps on the upper branch
  of the map, between 0.5 and 0.625. Each of the 19 sign variants of its
  equations breaks a case. Eleven of the 185 catalogue models now state cases.
- The Rulkov map neuron states six behaviour cases: 0, 7 and 10 events in 30
  steps at currents 0, 0.5 and 1.5, the counts its Q16.16 test enrols for the
  hand model, the schema and the RTL; and 0, 34 and 130 events in 1000 steps
  at the same currents, measured with its public class and matched by the
  schema. Each of the 17 sign variants of its equations breaks a case. Ten of
  the 185 catalogue models now state cases.
- The Ibarz-Tanaka map neuron states four behaviour cases: none in 4 steps at
  a current of -0.5 and 3 events in 30 steps at 0.2, the counts its Q16.16 test
  enrols for the hand model, the schema and the RTL; and 7 events in 1583
  steps at -0.5 and 18 in 1582 steps at 0, measured with its public class and
  matched by the schema, at horizons at least 17 steps from the nearest event
  of any perturbed run. Each of the 26 sign variants of its map breaks a
  case. Nine of the 185 catalogue models now state cases.
- The SC four-state GLIF neuron states six behaviour cases, the event counts
  its Q16.16 test enrols for the hand model, the schema and the RTL: none at
  rest and at a current of 15, then 23, 54, 86 and 95 events in 1000 steps at
  currents 22, 30, 45 and 50. Its public class holds them, and each of the 15
  sign variants of its four equations breaks one. Eight of the 185 catalogue
  models now state cases.
- The integer quadratic integrate-and-fire neuron states one behaviour case,
  the count its RTL test enrols for the source, both RTL forms and the schema:
  26 events in 400 steps at an integer current of 10. Its public class holds
  it, and each of the 8 sign variants of its map breaks it. A case's drive
  keeps the type it is written in: a whole number written without a point is
  an integer, so a model whose step takes an int32 current can be driven.
  Every drive stated before is written with a point and is unchanged. Seven of
  the 185 catalogue models now state cases.
- The SC resetting Wilson neuron states three behaviour cases, the event
  counts its Q16.16 test enrols for the hand model, the schema and the RTL:
  none without a current, then 1 and 4 events in 5000 steps at currents 2 and
  10. Its public class holds them, and each of the 14 sign variants of its two
  equations breaks one. Six of the 185 catalogue models now state cases.
- The symmetric SC quadratic integrate-and-fire neuron states eight behaviour
  cases, the event counts its Q16.16 test enrols for the hand model, the
  schemas and the RTL: none without a current, then 2, 3, 6, 11, 26, 100 and
  250 events in 1000 steps at currents 0.333, 0.5, 1, 2, 5, 20 and 50. Its
  public class holds them, and each of the two sign variants of its equation
  breaks one. Five of the 185 catalogue models now state cases.
- The Wilson cortical neuron states five behaviour cases, the counts its
  Q16.16 parity test enrols for the hand model, the schema and the RTL: none
  without a current, then 44, 46, 49 and 52 upward crossings in 5000 steps at
  currents 0.08, 0.1, 0.14 and 0.2. Its public class holds them, and each of
  the 15 sign variants of its two equations breaks one. Four of the 185
  catalogue models now state cases.
- The Hindmarsh-Rose neuron states five behaviour cases, the counts its Q16.16
  parity test enrols for the hand model, the schema and the RTL: none at
  currents 0 and 2, then 26, 40 and 52 upward crossings in 2000 steps at
  currents 3, 4 and 5. Its public class holds them, and each of the 13 sign
  variants of its three equations breaks one. Three of the 185 catalogue
  models now state cases.
- `tools/model_behaviour_variants.py` asks whether a model's behaviour cases
  depend on the signs of its equations. For a catalogue model that states
  cases and has a bound schema, it builds every variant of the schema with the
  sign of one term of one equation changed (on the equation's syntax tree,
  sums inside a factor or a function call included), runs the unchanged schema
  first, which must hold every case, and runs each variant on the cases up to
  the first one it breaks. `--check` fails on a variant that holds every case,
  on a schema that does not hold its model's cases, and on a stale
  `docs/_generated/model_behaviour_variants.json`. A schema runs through the
  Universal DSL as the Studio runs a candidate, with a case's current as the
  one input; a model without a bound schema, with a schema the engine refuses,
  or with cases of another shape is reported as such. In the behaviour report
  a step that returns a truth value counts as a spike or none, and an error a
  step raises makes the case invalid with the error's name. The
  FitzHugh-Nagumo descriptor states five cases from the model's band test:
  none without a current, 26, 28 and 28 spikes at currents 0.5, 0.8 and 1.0,
  and one spike at 2.0, in 10000 steps; its class and its schema hold them,
  and each of the eight variants of its two equations breaks one.
- `tools/model_behaviour_report.py` runs the behaviour cases that model
  descriptors state and writes `docs/_generated/model_behaviour_report.json`.
  Each case runs through the model's public class, on a newly built model,
  with its constant inputs passed by name, and is written as stated with what
  was observed, whether it held and whether it was exercised (a spike, a state
  variable leaving its starting value, or a contrast held). A model whose
  cases all hold while none is exercised is `held_vacuously`. `--check` fails
  on that, on a failed case, on a case that cannot be run (at most 100000
  steps) and on a stale report. Each model that states cases carries
  `cases_sha256`, the digest of its cases as parsed. The dendritic NMDA
  descriptor states eight cases and holds them: neither input alone fires the
  soma, both together give 59 spikes in 20000 steps, glutamate adds 17 spikes
  at a soma current of 40, glutamate alone moves the dendrite to its balance
  point near -59.52, and the lane anchor is 266 spikes. The cases are the
  author's statement of what the model does, not an independent reference. A
  drive is constant finite numbers, and a spike count needs a step that
  returns 0 or 1; a model whose step returns a rate or several values can
  state cases on its state variables only.
- A model descriptor may state behaviour cases in an optional `[[behaviour]]`
  list. An `interval` bounds one observable (the spike count, or the final
  value of a declared state variable as `state:<name>`) after a number of
  steps at constant named inputs, with both ends; equal ends state an exact
  value. A `contrast` states that the observable at one drive exceeds the
  observable at a second drive by at least a positive margin. A bound with one
  end, an order without a margin, an undeclared observable, an empty drive, a
  key the kind does not take, spike bounds spanning or exceeding the whole
  range of a run, and a fractional spike margin or one above the number of
  steps are refused when the descriptor is parsed. A drive is a table of finite
  numbers. The cases take no part in a readiness tier or in the contract
  digest of a receipt, and the descriptor generator keeps them when it
  refreshes a descriptor from the model code.
- The Studio worker launcher checks once, at its start, that its configured
  interpreter imports the worker bootstrap as a worker runs it: with `-s`, the
  worker's environment and, for a privileged launcher, the worker's identity.
  If it cannot, the launcher refuses to start and names the interpreter and
  the reason: it cannot be run, the import's last error line, or no answer
  within 60 s.
- One toolchain manifest, `tools/toolchain_manifest.toml`, gives every tool
  version the CI declares, and `python -m tools.toolchain_manifest --check`,
  also a pre-commit hook, refuses:
  - a declaration that differs from it;
  - a floating runner, an undated nightly, or an unpinned `apt`, `apt-get` or
    `choco` install of a listed tool;
  - a setup version it cannot read as a literal, such as an `env`, `vars` or
    `inputs` reference;
  - a setup action it does not know, in a workflow or a composite action.

  The hosted jobs name their images: `ubuntu-24.04` in place of
  `ubuntu-latest`, `windows-2025-vs2026` and `macos-26`. Other pins: the
  nightly Rust of the docs, fuzz, release and scanner jobs is
  `nightly-2026-11-13`, which the cargo-fuzz runner reads from the manifest;
  Julia 1.13 is 1.13.3; the Icarus Verilog, Verilator, cvc5 and ShellCheck
  installs name their Ubuntu 24.04 versions. The engine wheel artefacts follow
  the image names (`engine-wheel-ubuntu-24.04-py…`). `--check-host` asks the
  tools on `PATH` for their versions.
- The Studio marks on the browser's performance timeline when it draws a
  catalogue, a trace, a chosen view, a network and a training epoch, and when
  an epoch arrives from its stream; the marks change nothing on screen.
  `npm run test:e2e:timing` measures the built bundle against a real backend:
  ten samples of each measure across declared workloads (a catalogue load,
  simulations of about 1 000, 10 000 and 100 000 steps, model and view
  switches, networks of 2, 10 and 50 populations, a 50-epoch training stream),
  written as a `studio.render-timing.v1` artefact with every sample, p50, p95,
  host, load, browser and commit, labelled `functional-non-isolated` unless it
  ran on reserved cores. `docs/studio/performance-measurement.md` explains it.
  No budget is checked.
- A backend lane's receipt carries what its run showed: the recorder reads the
  lane and the library digests each test case wrote into the retained JUnit
  report, records the libraries as `native-binary` artefacts, and marks the run
  `error` when a case ran another lane, named no lane, or named no library on a
  lane that builds one. Its claim scope is the parity word the descriptor gives
  the lane (`reference` for the Python lane); the recorder refuses a lane
  without one, and the verifier reads a receipt `invalid` when the
  descriptor's word has changed since. The receipt also holds the version lines
  of the lane's toolchain. No shipped descriptor names a lane's evidence yet,
  so no status changes.
- A complete run of `LapicqueNeuron` records the backend lane that executed and
  the native library files it loaded, outside the model's attributes
  (`sc_neurocore.neurons.lane_runs`). The Lapicque accelerator names a loaded
  lane's libraries and answers which lanes can run (`backend_available`), so
  the lane gate skips an unbuilt Go or Mojo lane locally and fails it where the
  lane is required. A test fixture writes the lane and the library digests into
  a test's JUnit report, and `junit_lane_evidence` returns them per executed
  test case with every inconsistency.
- The readiness facet of a native backend lane binds the lane's sources and
  build inputs as its `native-backend` subjects; the Rust lane also binds every
  tracked file of the engine's Python package and of the vendored rayon-core
  crate its build compiles. A source counts only when it is a tracked file
  spelled as Git spells it (in the index of a Git checkout, compared as
  written; elsewhere in normal form; the file must exist). A declared lane that
  lists any other source reads `unavailable`, and the problem names the path
  or says why a checkout's index cannot be read; `lane_source_problems`
  reports them. A descriptor's `python` lane takes no sources. No shipped
  descriptor lists a source yet, so no status changes.
- Every backend lane a descriptor may declare has a readiness facet: sixteen
  facets, with `backend:rust_safety` new. A lane's facet reads its evidence
  from the `evidence` text of the lane's `[backends.<lane>]` table and requires
  the lane's sources as subjects. The library a run loaded is an artefact of
  the receipt, never a subject: a receipt of `backend:rust`,
  `backend:rust_safety`, `backend:go` or `backend:mojo` credits only when it
  carries it, and a receipt of any other facet only without an artefact.
  `LANE_BUILD_INPUTS` names the tracked manifests, lock files, build
  configurations and tool pins that fix each lane's build, and the readiness
  ledger's facet block lists the required artefacts. A declared lane without a
  bound receipt is listed on its profile's row and does not decide the
  profile's admission. No lane has a receipt yet.
- A model descriptor's `[backends.<lane>]` table takes its lane name from a
  closed list (`python`, `rust`, `rust_safety`, `julia`, `go`, `mojo`; the
  earlier `rust_engine` is still read) and may state `evidence`, the tests that
  are the lane's evidence, and `sources`, the tracked files that implement the
  model in that lane. A table under another name, a table without a status or
  with another key, a lane that is neither a table nor a status word, evidence
  that is not text, and a source that is absolute, passes through a parent
  folder, is empty or is listed twice are refused. No shipped descriptor
  changes and every model keeps its tier.
- The benchmark coverage census says whether each record still describes the
  sources in the tree: `current` (every bound source has the recorded digest),
  `outdated` (a bound source changed or is gone; the changed paths are listed)
  or `unbound` (no source is named by a path). An outdated record describes
  earlier bytes; it does not say that the measured value changed. The census
  lists this for the records a catalogue model cites and, separately, for every
  other record under `benchmarks/results` that binds sources; records that key
  digests by lane are bound through the evidence gate's manifest where it
  names their paths. An edit of any bound source regenerates the census in the
  same commit. The evidence gate gains `current_source_hash_failures`. Census
  schema version 2; the three model states and their totals are unchanged.
- `mux_dot_product_variance` gives the variance of one output stream behind a
  uniformly random multiplexer (`q_mean (1 - q_mean) / N`), and
  `interleaved_dot_product_variance` that of a round-robin multiplexer
  (`sum q_k (1 - q_k) / (K N)`; a length that is not a multiple of the number
  of terms is refused with `GuaranteeNotDetermined`). Each states its
  assumptions on the select input and on the independence of the streams.
  `dot_product_variance` is now described as what it computes, the variance of
  averaging every product stream in full (`sum q_k (1 - q_k) / (K^2 N)`), and
  returns the same values. For two constant terms 0 and 1 and `N = 100` the
  random multiplexer has variance 0.0025, where the averaged form gives 0.
- The wheel ships the core spike-train statistics as
  `sc_neurocore.spike_train_statistics`: spike times, inter-spike intervals,
  rate, count, binning, power spectrum, ISI coefficient of variation, Fano
  factor and permutation entropy. NumPy is its only dependency, and
  `sc_neurocore.analysis` re-exports the nine functions under their existing
  names. The identity health check and director monitor run from an installed
  wheel; state extraction and the decoder still need a source checkout for PCA
  and functional connectivity, and say so. Permutation entropy is computed
  with NumPy array operations, ties ranked by position as in the Rust helper,
  and an order or delay below 1 raises `ValueError`.
- A documentation page on what each kind of evidence establishes and what it
  does not: records of a run, agreement between implementations,
  co-simulation, formal results, reports of design tools, benchmark artefacts
  and measurements on a device. The architecture overview names, for each of
  its five contracts, the test or hosted step that checks it and how far that
  check reaches.
- The standalone Linux desktop runtime is assembled from the complete selected
  original wheel set, with installed records and wheel payloads validated and
  relocatable console launchers generated; original commands, diagnostics and
  process receipts are retained. A hash-locked CI input producer and a
  consumer check source binding, types, documentation, formatting and complete
  statement and branch coverage of the six assembly tools.
- `benchmarks/capture_lapicque_samples.py` retains all seven original timing
  samples for each of the five Lapicque backends, and
  `benchmarks/validate_lapicque_samples.py` rejects damaged, instrumented or
  source-mismatched evidence. Both run in CI with complete statement and
  branch coverage, and three guides document the commands, provenance checks
  and external performance-isolation boundary. No performance result or model
  readiness changes.

### Changed

- Truncating integer division has one home,
  `sc_neurocore.utils.numerics.trunc_div`. The compiler's function units
  re-export it instead of keeping their own copy, the IQIF model's branch
  point divides through it behind its own refusal of a denominator below
  one, and the NIR and NMDA co-simulation test helpers import it. A test
  refuses any other definition. The four co-simulation receipts that bind
  the compiler sources are recorded again, so their silicon labels stay.
  The Verilog compiler, function-call domain, equation-IR and IQIF
  benchmark records are measured again on the workstation they were taken
  on: every output, contract and trajectory is unchanged, the documents
  quote the new medians, and the IQIF lanes' same-host order is now Rust
  before Mojo. The two equation-IR gates' source maps also name the three
  sources the new measurement hashes, so the gate checks their digests.

- The Cazelles map descriptor states where its class and its bound schemas
  part. The class and its native runtimes snap a candidate near an edge onto
  the edge, within 8 ulp at the default domain, and refuse one outside the
  map's domain. The schemas,
  and the RTL compiled from them, have no domain guard and continue by the
  last linear piece. A current of 0.5 leaves the domain at the third step. A
  test holds the descriptor's notes to that behaviour.
- The benchmark evidence gate binds four more records whose digests name no
  path it reads. Three key their source digests by runtime lane, as the
  Benda-Herz record does: the Bertram phantom burster
  (`bench_bertram_phantom.json`, event parity across five runtimes) and the
  two GLIF simulations (`bench_glif_simulate.json`,
  `bench_sc_four_state_glif.json`, their workloads); each lane's digest is
  mapped to the file its benchmark script hashed. The MNIST accuracy record
  (`mnist_conv_accuracy_reproducibility.json`) is bound to the checkpoint and
  training log it names by path. At each record's commit every digest equals
  its file; the tracked gate report counts 89 gates. The benchmark coverage
  census reads the same maps: the Bertram and MNIST records are current, and
  each GLIF record is outdated by its Rust source.

- Move the Mojo environment lock to the newest compatible builds of click
  (8.5.1), platformdirs (4.14.5), python-dateutil (4.0.0, which no longer
  needs six), pyzmq (27.3.0), tornado (6.6.1) and libuuid (build 1); Mojo
  stays at 1.0.0.

- Move the three stages of `deploy/Dockerfile` to the current digest of
  `python:3.12.16-slim` (`sha256:5e2bdb1b…`), the tag's rebuilt image.

- Key the NIR interoperability benchmark's producers by their repository paths
  (`src/sc_neurocore/...`) and measure the record again with the documented
  command: its nine producer digests are unchanged, but keyed relative to the
  package they named no tracked file, so the benchmark coverage census called
  the record outdated. Its tests resolve each key in the checkout or in the
  installed package and require the repository path.

- A generated equation-neuron testbench writes its VCD waveform only when the
  simulation is run with the plusarg `+vcd` (`vvp sim +vcd`); without it the
  run writes no file. `compile --testbench` says how to ask for the waveform.

- Move soupsieve to 4.1.1 in the documentation, workstation and desktop
  assembly tools locks, and bracex to 3.2.3 in the semgrep lock; only those
  pins move.

- Measure the Verilog compiler and function-call-domain benchmark records
  again on the current sources: they bound earlier bytes of 8 of their 43
  and 9 of their 146 sources. The function-call gate also binds the two
  sources that benchmark now reads. Every generated output and serialised
  contract is unchanged. The timings come from a loaded workstation; old
  and new code run alternately there differ by less than the load's
  spread, so they show no change in speed either way. The compiler
  reference, the equation-to-Verilog tutorial and the function-call guide
  quote the new medians and how the run was made, and two tests hold the
  compiler reference's and the tutorial's tables to the record.

- Move the toolchains to their newest releases in the toolchain manifest
  and every declaration it checks: Node 26.13.1; npm 12.2.2, installed by
  the npm setup action from its verified registry archive and named by
  the frontend's packageManager and the contributing guide; pixi 0.84.2;
  and Julia 1.13.4 for the native-parity lanes.

- Move every dependency declaration that no recorded hold keeps back to
  its newest compatible release: ruff 0.17.5 in every declaration and
  the pre-commit hook, the typos hook 1.52.2, semgrep 1.185.0,
  pymdown-extensions 13.1 and uv 0.13.8; the hash-locked Python profiles
  regenerated with the commands their headers record, with coverage
  7.17.3, fastapi 0.162.7, filelock 4.2.10, markdown-it-py 4.3.0,
  stevedore 5.10.0 and further patch releases; the Rust locks at their
  current patch releases, the bundled Z3 504.1.0 among them; and the
  Studio frontend's @module-federation/vite 1.24.0, Playwright 1.65.1,
  ESLint 10.15.0, eslint-plugin-jsdoc 65.4.7, DOMPurify 3.4.18 and
  minimatch 11.0.8.

- The written C ABI headers of three Go adapters, adaptive_threshold_if, alpha
  and resonate_and_fire, are named `abi.h` in their packages, as
  conversion/cshared's already was, apart from the `lib<name>.h` that a
  c-shared build writes beside the library. The wheel ships
  `accel/go/adaptive_threshold_if/abi.h` in place of
  `libadaptive_threshold_if.h`, and conversion/cshared ships its `abi.h` only.
- The version audit (`tools/version_surface_audit.py`) holds the release
  metadata to the project version: `.zenodo.json`, `CITATION.cff` and
  `conda/meta.yaml`. A file among them that states no version is refused by
  name, and a stale checked-in file is reported as one to edit, not to
  reinstall.
- The benchmark coverage census counts a model as measured when a committed
  record measures it, also where the record does not hold the model's class
  name: a record that binds by path the benchmark scripts that produced it is
  attributed to the model those scripts name when they name exactly one, and
  to neither when they name two. Each cited record states how it is attributed
  (`class_name` or `bound_script`, with the script). A script names a model
  only by its class name as a whole identifier, so `NMDANeuron` is no longer
  named by a script that names `DendriticNMDANeuron`. Measured models go from
  28 to 84 of 185, and models with a script and no record from 73 to 17; 84
  have neither, as before. No record was measured again: 61 records listed as
  naming no model are now attributed, with their currency unchanged. Census
  schema version 3.
- Readiness evidence is recorded from the validator's actual interpreter, and
  exact raw execution artefacts are retained privately. Public receipts keep
  repository-relative commands and projected output without host identity.
- Verified silicon readiness is derived through the descriptor's canonical
  tier rules, and symbolic differentiation is typed through the existing SymPy
  interfaces.
- The README states at its top that this repository is a public mirror of
  ANULUM's canonical source, with the development history before publication
  held privately.
- The architecture overview page is replaced by an entry point to the
  architecture: what the programme is for, the loop from model to evidence,
  the three tiers that exist today, the contracts between them, which
  established tools are integrated and what is built here, what ships and what
  is research, and what is planned. It states no counts and links to the
  generated capability snapshot; the navigation lists it first. The system map
  and two pages that described the old content are reworded, and two stale
  counts are removed from `ARCHITECTURE.md`.
- The integration and performance-tuning guides are rewritten from the code
  and records. The integration guide covers the SHD accelerator: what is
  co-simulated and synthesised, that a bitstream from the block-design script
  carries no trained model, how to build it, the driver and its register map,
  and the steps before a first board run. The performance guide covers
  bitstream length and precision, with lengths computed by `adaptive_length`,
  the packed, CuPy, sparse, Rust and GPU paths, the one committed timing
  result, and the absence of any power measurement. The PYNQ demo and driver
  use the block-design instance name `sc_shd_0`; the demo no longer prints a
  throughput from core cycles alone, and gives the 75.2 % accuracy as the
  floating-point model's. The hardware-claims test reads both guides.
- Seven docs of the engine's population and map models now say exactly
  what the code does, after a second reading: the clamps come first, so
  an infinite rate or state is clamped and only a NaN is replaced by its
  initial value; the Tsodyks-Uziel-Markram rate uses the new resources
  and utilisation before their clamp; the Montbrió-Pazó-Roxin event is
  an upward crossing and the Aihara event a level, not a crossing; the
  Aihara batch checks its step count, configuration and currents before
  the first step but can still stop part-way; and the Ibarz-Tanaka reset
  branch needs a positive `v` as well as `v >= top`.
- The engine's Wong-Wang decision circuit (`engine/src/wong_wang.rs`) is
  documented in full: its eleven described fields from the model
  descriptor, and the rest from the code. The step's equations, the
  transfer function's guarded branches, the refusals that leave the state
  unchanged, the batch's sample layout and its error behaviour are stated.
  The Ornstein-Uhlenbeck update makes the noise's stationary standard
  deviation `sigma / sqrt(2 - dt / tau_ampa)`, close to `sigma / sqrt(2)`
  for a small step, which the struct doc now says, and `step` draws its two
  samples before the checks, so a refused step still advances the random
  stream.
- The engine's six synapse models (`engine/src/synapses/mod.rs`: STDP,
  reward STDP, static, triplet STDP, short-term plasticity, dopamine STDP)
  are documented in full, each read against its code. Corrected by
  reading: the fixed-point STDP step's doc gave a pre spike a weight
  increase, but the code depresses there (by `trace_post * |a_minus|`,
  floored at `w_min`) and potentiates on a post spike, so simultaneous
  spikes potentiate; the triplet rule's formulas paired the pair amplitude
  with the slow trace, where the code adds `A3 * o2` (and `A3 * r2`) to the
  pair amplitude. The static synapse's `delay` is not applied by
  `transmit`, which its doc now says.
- The engine's photonic network-on-chip results (`engine/src/photonic.rs`:
  waveguide routing, WDM crosstalk, power budget and the evanescent
  crosstalk of a waveguide bank or of given pairs) are documented in full:
  each field says how its value is computed, such as the
  `ceil(sqrt(n))` routing grid with 0.08 dB per crossing, the three
  bandwidths that count a channel as adjacent, the MZI losses a power
  budget adds, and the 300 dB isolation that stands for no coupling.
  `route_waveguides` was documented as returning tuples; it returns one
  `WaveguideResult` per connected pair.
- The network runner's neuron dispatch (`NeuronVariant`, 179 variants) is
  documented: each variant links the model or adapter it wraps, and
  `soma_voltage` states what it returns for models without a membrane
  potential (a map's `x` or output, a phase, a population rate, an
  adaptation, accumulator or current variable, a wrapper's `v()`, the
  retinal ganglion cell's constant baseline, or 0 for the point processes
  and the continuous-attractor and compositional-binding models).
- The engine's spike train analysis modules (`engine/src/analysis`) each
  carry a doc naming what they hold, the GPFA result's size fields are
  documented, and `spike_directionality`'s doc, which sat on the `use`
  statement above it, is back on the function and now says which spikes
  it counts.
- The engine's DNA circuit module (`engine/src/dna.rs`) documents its
  gate kinds, each with the rate law it gives (the NAND and XOR gates
  have none and run on their leak alone), and the fields of a gate and
  of the kinetic configuration, with the units the rates are scaled in.
- The engine's DCLS Q8.8 reference (`engine/src/scpn/dcls.rs`) documents
  its format configuration, which accepts only Q8.8 data with a Q16.16
  accumulator, and each error it returns, including that the tap-index
  error covers only an index above `i64::MAX`.
- The engine's stochastic graph layer (`engine/src/graph.rs`) documents
  its CSR matrix, its adjacency storage and the layer itself: the
  degree-normalised aggregation, the layout of the transform weights
  (which the bitstream pass clamps to [0, 1]), and the checks and panics
  of its constructors.
- The engine's fusion, memristive and learning layers
  (`engine/src/fusion.rs`) document their fields and constructors: the
  fusion weights' normalisation, the memristive write noise and stuck-at
  faults, and the learning layer's update, which uses the spikes of the
  same step only.
- The engine's SC optimizer (`engine/src/optimizer.rs`) documents its
  candidate estimates (mode and decorrelator codes, power, accuracy and
  latency) and its annealing result, including that the three `pareto_*`
  vectors hold every accepted move, which `extract_pareto` then filters.
- The engine's supervised control loop (`engine/src/supervisor.rs`)
  documents its snapshot, shared state and errors, the Z3 check (what it
  proves; an undecided solver counts as a failure, and a snapshot with a
  negative marking is reported safe), the worker, and the run: when it
  ends, including with a clean result once the verifier thread has ended
  without failing a snapshot, and that the Python handle's `start` clears
  a shutdown forced before it.
- The engine's surrogate-gradient components
  (`engine/src/grad/surrogate.rs`) document each surrogate's parameter
  (none is checked; a zero or negative one gives an infinite, negative or
  unbounded derivative), the straight-through window, and that the
  surrogate LIF caches the voltage after its step, so a spike step is
  evaluated at the reset voltage.
- The engine's 7 single-neuron models under `engine/src/neuron` (AdEx,
  dendritic XOR neuron, exponential integrate-and-fire, fixed-point LIF,
  homeostatic LIF, Izhikevich, Lapicque) are documented in full, each file
  read against its code. Stated by reading: the fixed-point LIF holds `v`
  at `v_rest`, not `v_reset`, while refractory, and stores `v_rest` and
  `v_reset` unmasked; the exponential integrate-and-fire's two cached
  fields are read by no step and written by no refractory step;
  `Izhikevich::new` takes every parameter, and only `regular_spiking`
  fixes the regular-spiking values; the public `mask` helper panics for a
  width of 0 or above 32. To keep the files under their frozen line
  ceilings, three files' unit tests move to a sibling `<name>_tests.rs`
  and six methods or helpers move to child modules, with every public
  path unchanged; the seven ceilings fall to the new lengths. The AdEx
  and exponential integrate-and-fire benchmarks list the new files among
  their sources, and the DPI and Neurogrid benchmarks list the two files
  the hardware emulators' earlier split created. The Lapicque benchmark
  keeps its source list: the sample capture binds that harness byte for
  byte as reviewed source, so its new child module joins the list only
  together with a new review of the capture.
- The engine's 12 top-level map and neural-mass models (Aihara, Cazelles,
  Chialvo, Courbage-Nekorkin-Vdovin, Ermentrout-Kopell, Ibarz-Tanaka,
  Jansen-Rit, Medvedev, Rulkov, and the SC clipped-logistic,
  clipped-rational and upward-crossing Rulkov maps) are documented in full,
  each file read against its code. Each batch helper's doc says whether it
  is failure-atomic (Chialvo, Ibarz-Tanaka, Rulkov and the upward-crossing
  Rulkov map run on a copy) or keeps the steps before an error (Cazelles,
  Courbage-Nekorkin, Ermentrout-Kopell, Medvedev and the clipped-rational
  map). Stated by reading: the Cazelles snap tolerance, the Ibarz-Tanaka
  lowest branch, which depends on the input and the slow state, the
  Ermentrout-Kopell crossing on the phase before it is wrapped, and the
  Jansen-Rit argument order `y0, y3, y1, y4, y2, y5`.
- The engine's 4 population models (Brunel balanced network, El Boustani
  network, Montbrió-Pazó-Roxin mean field, Tsodyks-Uziel-Markram network)
  are documented in full, each file read against its code: which update
  uses the old and which the new state, the bounds each step clamps to, and
  the value a non-finite state is replaced by. The Montbrió-Pazó-Roxin
  equation now shows the coupling term `J * tau * r` the code adds, and its
  rate is in units of 1/`tau`, not Hz. To keep the four files under their
  frozen line ceilings, each one's unit tests move to a sibling
  `<name>_tests.rs`, and the ceilings fall to the new lengths.
- The engine's 10 undocumented rate-model files (Li-Rinzel astrocyte,
  fractional-order LIF, leaky compete-and-fire, liquid time-constant,
  McCulloch-Pitts, parallel spiking, Siegert transfer function, sigmoid
  rate, threshold-linear rate, Tsodyks-Markram) are documented in full, each
  file read against its code; the Amari field and the Compte working-memory
  model, already documented, deny missing docs too. Stated by reading: the
  astrocyte's `k_er` is both its ER leak rate and its SERCA half-activation,
  and a NaN current sets its calcium to 0; the liquid time-constant neuron's
  `bias` enters only its time-constant sigmoid (its descriptor now says so),
  and a threshold of 1 is reached only by overshooting, when `dt / tau`
  exceeds 1, or by rounding onto a target `tanh` has saturated to exactly 1;
  the Siegert noise amplitude is a tenth of the input's magnitude, not a
  parameter; the leaky compete-and-fire units inhibit each other in index
  order.
- The engine's 5 motor-system models (alpha and gamma motor neurons, motor
  unit, Renshaw cell, upper motor neuron) are documented in full, each file
  read against its code: the alpha neuron's sub-steps with its persistent
  inward current, calcium buffering and afterhyperpolarisation, the exact
  relaxations of the Renshaw cell, the upper motor neuron's four exponential
  sub-steps and its clamp of `v` to [-150, 100] mV, and what each reset
  restores. Corrected by reading: the motor unit's twitch is first-order in
  both runtimes, not a critically damped second-order response (the Rust
  and Python docstrings now say so); its descriptor's `a_adapt` is a
  subthreshold coupling, not a per-spike increment in mV, and its
  `tau_twitch` is the decay time constant of the force; the alpha neuron's
  `buf_ratio` is the free fraction of entering calcium, its
  afterhyperpolarisation conductance is a potassium one, and its persistent
  inward current is centred at -40 mV.
- The engine's 9 hardware-emulator models (Akida, BrainScaleS AdEx, DPI,
  Loihi 2, Loihi CUBA, Neurogrid, SpiNNaker 2, SpiNNaker LIF, TrueNorth) are
  documented in full, each file read against its code: the integer update
  order of the Loihi and SpiNNaker steps (divisions truncate towards zero, a
  time constant of 0 panics, the fixed-point decay product wraps), the
  rank-order weighting of the Akida neuron, the dendritic input and capped
  soma of the Neurogrid neuron, and what each reset restores. The DPI
  neuron's documentation moves from its trace alias to the struct. The
  TrueNorth descriptor says that the leak is subtracted, as every runtime
  does.
- The engine's 12 AI-oriented neuron models are documented in full, each file
  read against its code: each step's update order and event rule, what reset
  keeps (the Arcane neuron keeps its deep compartment, learned self-model and
  novelty history), and the inputs that make a step panic (an empty
  key-value state or a window longer than its buffer in the hybrid linear
  attention neuron). The differentiable-surrogate descriptor describes
  `alpha` as the per-step membrane decay, not a surrogate-gradient scale.
- The engine's 6 interneuron models (basket, chandelier, PV fast-spiking,
  Martinotti, SST, VIP) are documented in full, each file read against its
  code: the Wang-Buzsáki family's sub-steps of `dt` up to about 0.5 ms per
  call (a `dt` above 0.5 ms advances nothing), the four RK4 sub-steps of the
  Martinotti, SST and VIP models (VIP with its own kinetics), each channel's
  gate and time constant, and the spike as an upward crossing with no reset
  of `v`.
- The engine's 7 special-module files (the Larter-Breakspear and Wendling
  neural masses, the retained decoupled ion mass, the spike response and GLM
  neurons, the stochastic point processes and spiking models, the Wilson-Cowan
  unit) are documented in full, each file read against its code. Corrected by
  reading: the Larter-Breakspear descriptor named `v_ca`, `v_na` and `v_k`
  half-activation thresholds, but they are reversal potentials, and `b` is the
  inhibitory population's rate factor, not a calcium slope; the
  Galves-Löcherbach `threshold_rate` is the centre of the firing sigmoid, not
  its scale; the spike response neuron's reset no longer carries the GLM's
  documentation; the stochastic IF `sigma` is the noise amplitude, not the
  standard deviation of the membrane noise; the Wilson-Cowan states name the
  range of the shifted sigmoid they relax towards.
- The engine's 6 remaining cell and connection models (cardiac Purkinje fibre,
  pancreatic beta cell, gap junction, graded synapse, myelinated axon, smooth
  muscle) are documented in full, each file read against its code: the
  sub-step scheme, the currents, the spike or event rule over the whole outer
  step, the clamps and what reset restores. Corrected by reading: the node of
  Ranvier has no fast potassium current; the Frankenhaeuser-Huxley
  permeabilities are effective current densities, `P * F * [C]o / 1000`, and
  the potassium concentration ratio is 48 by default, not about 30; the
  myelinated-axon step states how it approximates the node-internode coupling.
- The engine's 12 sensory models are documented in full, each file read
  against its code: each transduction stage, its integrator and clamps, and
  what an invalid input or state does. Corrected by reading: the outer hair
  cell's length change is `L_max * (0.5 - charge/Q_max)`, contraction
  positive, and its nonlinear capacitance is not modelled; the retinal
  ganglion cell's rate is per ms in effect, not in Hz; the olfactory receptor's
  two feedback pathways are named by their time constants, and its CaM pathway
  scales cAMP production rather than acting on the CNG channels. The
  documentation records that the direction-selective ganglion cell reads no
  direction, and that the rod's dark calcium settles near 9, not 1.
- The engine's 8 ion-channel models are documented in full, each file read
  against its code: the 50 forward-Euler sub-steps of the six Wang-Buzsaki
  based models (gates from the old potential, currents from the updated
  gates), each channel's gate and current, the midpoint step and magnesium
  block of the NMDA autapse model, and what a refused step returns. Where the
  engine differs from the Python models it says so: the A-type and BK models
  clamp and repair where Python refuses, and their reset also restores the
  parameters; the BK half-activation uses the calcium before its decay. The
  input-driven NMDA project model gains its struct documentation.
- The engine's 17 trivial-module models are documented in full, each file
  read against its code: each update and spike rule, what is validated and what
  a refused step returns, and what reset restores. Where the engine differs
  from the Python model the documentation says so: the nonlinear LIF takes an
  unvalidated forward-Euler step and spikes only on an upward crossing, not the
  validated RK4 step, and the parametric LIF returns the spike of the stored
  potential, one step behind. Its struct names a quadratic, not a cubic,
  nonlinearity. Three descriptors are corrected: the KLIF `k` is the input gain;
  the nonlinear LIF's `a` and `b` are in `uA/(cm^2*mV^2)` and `mS/cm^2`; the SC
  normalised-energy LIF's `alpha` is the dimensionless energy each spike
  removes.
- The engine's 10 multi-compartment models are documented in full, each file
  read against its code: how each step integrates (Euler sub-steps, RK4 or the
  implicit cable solve), which compartment receives the input, where the spike
  is detected, and what a refused step returns. The astrocyte LIF docstrings
  (Rust and Python) now divide the whole membrane sum by `tau_m`, as both codes
  do, and the Marder STG descriptor gives `f_ca` the unit `uM*cm^2/uA`. The
  astrocyte LIF Python module documents its public methods in NumPy form and is
  enrolled in the docstring policy.
- The engine's 22 simple spiking models are documented in full, and the Rust
  documentation ceiling is lowered from 3004 to 2712 undocumented items. The
  documentation states each model's integrator and step size, its spike rule,
  which inputs are refused and what a refusal returns (some models return -1,
  others 0), and whether reset keeps the parameters. The McKean descriptor
  describes `mu` as the height of the Heaviside excitation step added where
  `v` reaches `a`, not a left-branch gain.
- The engine's 24 biophysical conductance models and the shared `safe_rate`
  helper are documented in full, and the Rust documentation ceiling is lowered
  from 3490 to 3004 undocumented items. The documentation states each model's
  integration scheme, spike rule, validation and reset, and each file denies
  missing documentation. Corrected: the Traub-Miles documentation no longer
  claims an M-current the model does not have; the GLIF resistance is in GOhm,
  not MOhm; the GIF adaptation increment moves `eta`, not the threshold; the
  Mainen-Sejnowski coupling and capacitance belong to the axonal compartment;
  the Huber-Braun Rust model's documentation says it adds no membrane noise,
  unlike the Python model.
- The engine's cerebellar models are documented in full, and the Rust
  documentation ceiling is lowered from 3646 to 3490 undocumented items: every
  field, constructor, step and reset of the Golgi, deep cerebellar nuclei,
  granule, Lugaro, stellate and unipolar brush cell models, and each file
  denies missing documentation. `tools/rust_model_field_docs.py` writes a
  field's meaning and unit from the neuron model descriptors where the Rust
  field has no documentation yet, and never replaces existing documentation.
  The Lugaro descriptor describes `a_adapt` as the subthreshold adaptation
  coupling, as both models use it, not a per-spike increment.
- British spelling in the descriptive values of the `LapicqueNeuron`
  descriptor and the `lapicque` schema profile (TOML and JSON): the units
  (`normalised polarisation`, `normalised capacitance`, `normalised
  resistance`), the meanings, display name, summary, scope, notes and
  validation text, the model module's docstrings and its two error messages.
  Keys, the schema's `family` identifier and test identifiers are unchanged,
  and no unit, value or behaviour changes.
- British spelling in the unit strings of the `SCLapicqueLIFNeuron` and
  `SCUnitCapacitanceRespiratoryNeuron` descriptors and of the
  `sc_lapicque_lif` schema profile (`normalised voltage`, `normalised
  resistance`, `normalised`). No unit, value or behaviour changes.
- British spelling in the descriptive text of eight neuron model descriptors
  (`normalised`, `energised`, `specialisation`, `synthesises`,
  `repolarisation`, `behaviour`). Unit strings and the Lapicque descriptor are
  unchanged.
- Move the Studio frontend's Vite pin from 8.3.7 to 8.3.8 in `package.json`
  and the lock.
- Refresh the dependency pins no pending change holds: patch updates of itoa,
  syn and const-oid in the nine separate Rust crate locks; Vite 8.3.7, the
  React plugin for Vite 6.3.0 and typescript-eslint 8.72.2 in the Studio
  frontend; and the typos hook 1.52.0. The engine's lock and the Python, Go and
  Mojo pins are left for the change that fixes them; Rust stays on 1.98.1
  while its recorded migration constraint remains, and TypeScript 7 waits for
  typescript-eslint, which supports TypeScript below 6.1.

### Fixed

- `.zenodo.json` no longer relates SC-NeuroCore as part of
  `anulum/scpn-fusion-core`, a separate repository. The deposit Zenodo
  makes from this release carries only the relation to this repository.
- The DNA kinetic simulators refuse a run they cannot simulate, with
  ValueError and the same words in the engine's py_dna_simulate_kinetics
  and the Python KineticSimulator, before anything is computed. A gate
  without its input names panicked in the engine, which the release build
  turns into SIGABRT; an unknown gate type ran as AND; unequal lists were
  cut to the shortest and a name given twice kept its last concentration;
  NAND, XOR and the Python-only CATALYTIC ran on the leak alone; a gate
  writing "time" or another gate's output replaced that trace; and a step
  of 0, a NaN or a tiny step gave no samples or asked for an endless
  allocation; a NaN or negative concentration ran, and OR read it as 0 in
  the engine but not in Python. Now dt must be finite and above 0,
  duration_s finite and at least dt, every input concentration finite and
  at least 0, each gate must read the input count its type needs (AND and
  OR one or more, MUX three, the others one), and a run may hold at most
  2**25 samples. The Python time axis steps by dt, as the engine's does,
  instead of spreading the samples over duration_s; for an accepted run
  with the same integrator both give the same traces. The engine's
  kinetics moves to engine/src/dna/kinetics.rs
  with check_kinetics, KineticsError and try_simulate_kinetics, which
  returns the refusal; simulate_kinetics keeps its 4.1.0 signature and
  panics with the refusal's words.

- The surrogate LIF's gradient on a spike step is taken at the membrane
  that crossed threshold, before the reset. Its forward pass cached the
  voltage the fixed-point step returns, which is the reset voltage on a
  spike step, so backward evaluated the surrogate there: with the defaults a
  spike got about 1/24 of the gradient of the membrane that fired, and less
  than a near miss. The surrogate stands in for the derivative of
  S[n] = Θ(U[n] − ϑ), with the reset acting on U[n + 1] (Neftci, Mostafa and
  Zenke 2019, arXiv:1901.09948, eq. 5). The fixed-point LIF gains a
  crate-internal step that also returns that membrane, and its public step
  returns the same values as before. A refractory step's gradient is now
  exactly 0: the step returns no spike and the rest voltage whatever the
  membrane is, so a surrogate value there was a gradient the forward pass
  does not have.

- A surrogate LIF built with a configuration the fixed-point LIF refuses
  raises ValueError with the reason, through the extension's class and
  through sc_neurocore_engine.grad.SurrogateLif. The constructor panicked,
  and the engine's release build aborts on a panic, so a data_width
  outside [1, 16], a fraction not below it or a negative refractory period
  ended the Python process with SIGABRT and no exception. The Rust
  SurrogateLif gains try_new; new keeps its documented panic.

- The Go and Rust safety kernels of the Hodgkin-Huxley model now run as
  many explicit-Euler sub-steps as the Python model where 1/dt is a
  half-integer. The Python model runs round(1/dt) sub-steps, and Python
  rounds a half to even; both kernels rounded it away from zero, so at dt
  0.4 they ran 3 sub-steps where the Python model runs 2, and at dt 2 one
  where it runs none. The Julia and Mojo kernels already rounded to even.
  Each of the three kernels' tests now holds the Python model's state after
  one step at dt 0.4, 2/9, 2/13 and 2. The default dt of 0.01 gives 100
  sub-steps either way, so no benchmark number changes.

- The engine's timed motor-system tests (alpha, gamma and upper motor
  neurons, the motor unit and the Renshaw cell) are held to host-derived
  budgets as well, not to a fixed number of milliseconds; the budgeted tests
  number 34. Every ratio is measured again on the reference host in one
  session. The guard that holds each budget to a ratio keys it by the module
  path `module_path!()` gives at it: a file declared `mod tests;`, with or
  without `#[path]`, is that tests module, and a file that `#[path]`
  includes as another module keeps its inline tests below it, through
  declarations with a visibility and chains of included files back to the
  crate root. It read only an inline `mod tests`, so it refused a budget in
  a moved test module or in the AdEx tests file, and keyed others by the
  file's name.

- The benchmark evidence gate now checks every source digest its records
  hold in their tables. In 23 gates the manifest's map named only part of
  its record's path-keyed digests, 243 paths in all, so a record whose
  digest for one of the others was wrong at its own commit would still have
  passed; the two equation-IR records alone held 15 and 14 such paths. Each
  map now names every such path, and every one of them matches the source
  at the record's commit. A test refuses a digest in a record's tables, at
  any depth, that the gate does not check, either at its own place or
  through the same digest of the same path where the map points. Three
  records also hash their sources once per variant inside a list, 72
  digests the gate cannot address yet; the test holds their number per
  gate.
- Corrected 85 citations in 25 model pages, the transcriptomic page, three
  engine docs, two model sources, one descriptor summary and two test
  docstrings, where a recurring false co-author, "Bhatt", stood in place of
  the real authors. Each corrected citation is its DOI's registry record
  (Crossref, or DataCite for one arXiv preprint), read a second time in
  PubMed (60 works) or Semantic Scholar (5), and takes the record's year,
  journal, volume and pages where these were wrong too: El Boustani and
  Destexhe 2009 is in *Neural Computation* 21:46-100, Rasmusson et al. 1990
  in *Am J Physiol* 259:H352, Vervaeke et al. 2010 in *Neuron* 67:435 and
  Berkefeld, Fakler and Schulte in *Physiol Rev* 90:1437 (2010). Five
  entries with a wrong title are the one record that shares their first
  author and year and also the cited place (Boccaccio et al. 2006,
  Santos-Sacchi et al. 2006), the journal and volume (Diana et al. 2007) or
  the title's opening (Carrasquillo et al. 2012, Gerling et al. 2014). The
  Lugaro page and engine doc and the learnable neuron page now cite the
  papers that the July descriptor check named for their fabricated citations
  (Dieudonné and Dumoulin 2000; Jolivet et al. 2006), and the astrocyte
  model cites Li and Rinzel 1994. Twenty-two citations that no registry
  holds, or whose cited place holds another paper, are kept as written and
  listed for review, and so is the El Boustani page's account of the 2009
  paper, whose abstract names no NMDA term. A new test counts the remaining
  "Bhatt" citations per file, so a new one fails. The five Python files the
  corrections touch join the docstring policy (2018 to 2023 files): the
  astrocyte and VIP models gain module docstrings and documented checks, the
  astrocyte reset is documented as returning the class defaults rather than
  the values the model was built with, and the two test support modules have
  their text moved above the future import, where it is the module
  docstring. Two unipolar brush cell statements attributed to Diana et
  al. 2007, and the outer hair cell's prestin form attributed to
  Santos-Sacchi et al. 2006, are marked as not shown by those papers'
  abstracts and are under review.

- The photonic crosstalk benchmark hashes `engine/src/photonic/bindings.rs`
  as well. Its engine lane calls the crosstalk functions defined there since
  the bindings moved out of `engine/src/photonic.rs`, so a change to them
  was not recorded in a new measurement's source digests. The committed
  record is unchanged; its digests are checked at its own commit.

- A Studio event-stop test counted the storage authority's finish requests
  as soon as `wait` returned. The authority applies the first finish before
  its reply is lost, so the record is terminal before the supervising
  generation sends the identical repeat, and on a loaded host `wait` returned
  in between: a hosted coverage shard saw one finish where two were expected.
  The shared helper now waits, up to 60 seconds, for the repeat before it
  counts, and a new test holds the repeat back for half a second, which
  fails against the old helper.

- `HomeostaticLIFNeuron.reset_state` now restores the threshold and the rate
  trace the neuron was built with, as well as returning `v` to rest. It
  inherited the stochastic LIF's reset, which kept both adapted values, so a
  reset neuron did not repeat its first run; the base contract asks a reset
  for every internal state's initial value, and the engine's
  `HomeostaticLif::reset` already restored both. The two steps still differ:
  the engine's is a deterministic leak with a fixed 20-step time constant,
  the Python model's the stochastic LIF, which the Python docstring and
  `engine/PARITY.md` now say.

- The bit-true fixed-point LFSR refuses a register it could never leave. It
  checked only for a seed of 0 before masking, so a non-zero seed whose low
  `width` bits are 0 (0x10000 at 16 bits) loaded a zero register, and
  `reset(seed)` checked nothing; both now refuse a seed with no set bit in
  the register, and a refused reset leaves the state as it was. A width
  below 6, where the feedback's lowest tap (bit width - 6) does not exist,
  is refused when the LFSR or the bitstream encoder is built, instead of
  failing at the first step. The engine's `Lfsr16` is 16 bits wide and
  already refuses a zero seed in both places. The RTL encoder's `SEED_INIT`
  stays a documented parameter contract, with no check.

- `python -m tools.rust_doc_ratchet --update` keeps the ceiling's form. It
  rewrote the engine's aggregate v1 ceiling as the individual v2 record of
  every undocumented case, about 10 MB, while the repository keeps the v1
  form, so each documentation unit lowered the v1 figures by hand. An
  aggregate ceiling is now lowered in place, with the measured revision as
  its `source_sha256`, and a ceiling that lists its cases keeps listing them.
  `--individual` migrates an aggregate ceiling to the v2 form as a deliberate
  step, and `--record <path>` writes the full v2 record outside the
  repository; both refuse to run without `--update`, and the record refuses
  a path inside the repository.

- Thirteen more model descriptors name RK4, the integrator their classes run
  by default and every backend lane uses: DeSchutterPurkinje, Durstewitz
  dopamine, Golomb FS, Hay L5 pyramidal, Izhikevich 2007, Martinotti,
  multicompartment MCN, NeuroGrid, PV fast-spiking, Pospischil, the six-state
  thalamocortical neuron, SST and VIP. They said Euler. The descriptor
  generator now takes a class's integrator default as its method; a test
  holds every class that declares one.
- The Python Loihi CUBA and Loihi 2 models now decay each state by its
  quotient truncated toward zero, as the engine does; they floored it. A
  negative state therefore leaked more than the same positive one, and a
  drive that took a state below zero gave the two runtimes different traces:
  for the drive -7, -3, 5, -9, -1, -4 the Python Loihi CUBA neuron ended at
  v = -33, u = -10 and the engine at v = -48, u = -13. Truncation is the
  rounding of Loihi's sign-magnitude decay, which Lava's bit-accurate LIF
  model also uses, and the one the Loihi CUBA page already stated. The new
  `sc_neurocore.utils.numerics.trunc_div` gives it, and the descriptors'
  time-constant meanings say so.
- The engine's Akida neuron now follows the Python model: a weight of 0 is
  no input event, the next event's weight factor is a running product of
  `modulation` rather than a power, and the accumulator keeps integrating
  after its one spike. The engine counted every call as an event, so in a
  network every step without input advanced the rank, and the weights 3, 0,
  3, 3 gave the Python model v = 6, rank 3 and the engine v = 5, rank 4.
  The power also rounded differently from the product: 160000 at rank 4
  with a `modulation` of 0.95 added 130321 in the engine and 130320 in the
  Python model. A test holds the two runtimes to the same states step by
  step for all three neurons.

- Eleven parameters of the hardware neurons' descriptors, and one of the
  self-referential neuron, named a unit in their meaning while declaring
  the unit "1", so the engine's field docs read two units side by side, for
  example "Resting potential (integer fixed-point units) (dimensionless)".
  Each now declares its unit (integer fixed-point, integer accumulator or
  integer units, or steps) and its meaning no longer repeats it. The Loihi
  CUBA `tau_v` meaning said it decays the membrane current where it decays
  the voltage, and the Akida `modulation` meaning spoke of spike ranks where
  the rank counts input events; both now say what the step does. The engine
  docs are rendered again from the descriptors, and a test refuses a
  dimensionless field whose meaning names a unit.
- The engine's continuous-attractor ring and the Python model now agree bit
  for bit, and both put the bump at the first of equal maxima. The engine
  took the last of them, so at rest it named unit 15 where the Python model
  named unit 0. The Python model summed each unit's recurrent input with the
  built-in `sum`, which compensates rounding from Python 3.12 on, so its last
  bits, and on an even ring the unit holding the bump, depended on the
  Python version: after 50 steps at 2.0 Python 3.12 named unit 0, Python 3.11
  unit 7 and the engine unit 12. It now sums left to right, as the engine
  does, and all three name unit 7. The model page and the class docstring
  also wrote the recurrent input as `f(sum_j w_ij u_j)`; both runtimes sum
  `w_ij f(u_j)`, and the equation now says so.
- The engine's timed Rust tests in the cerebellar, channel, interneuron,
  sensory, simple-spiking and miscellaneous neuron modules are held to a
  budget derived from the host that runs them, not to a fixed number of
  milliseconds. A fixed limit held on the machine it was written on and failed
  on a slower or busier one: during a full `cargo test` on a loaded host the
  Golgi cell's 50 ms limit failed while the test alone passed. A test's budget
  is the policy margin times its reference ratio times a calibration taken in
  the test's own thread just before and just after its work, on the thread's
  CPU clock where the platform has one. The ratios, the reference hosts and
  their runs, and the margin are kept in `engine/test_host_profiles.toml`; a
  test holds every ratio to its rule and refuses a new literal millisecond
  limit in the engine. The six interneuron timing tests, opt-in until now for
  the same reason, run by default again. Eight tests that dropped each step's
  result now keep it, so a release build cannot remove the work they time.
- A formal resource attempt refused for a Python traceback in its solver
  log now names the first exception the traceback reports, not only its
  first line, which names no cause; a traceback cut short by another tool
  line is marked "(no exception line)". A hosted run failed this way once
  with only "Traceback (most recent call last):" to show, and its log was
  gone with the runner.
- The engine's fractional-order LIF neuron now keeps the potential after a
  reset in its memory, as the Python model does. It kept the value from
  before the reset, so the memory term undid every reset: after its first
  spike the engine fired on every step, even under a drive below threshold
  (one step at 1.5, then 0.5: the Python model settles at 0.5, the engine
  fired on each step). `reset` also fills the memory with `v_rest`, as the
  Python model does, rather than with 0. The Go, Julia and Rust safety-crate
  runtimes already stored the reset value, so the engine was the only one
  out of line; a network of the network runner's `FractionalLIF` neurons
  now resets as well. A test compares the engine with the Python model step
  by step through two spikes.
- `POST /api/models/cosim` answers HTTP 422 with its reason, before any job
  runs, when the model offers no bit-exact co-simulation path for the
  integrator or the current lies outside the Q-format's range. Both used to be
  admitted as a job and answered HTTP 500 `studio_job_failed`, with the reason
  kept only in the job ledger. The Studio's own request builder already
  refused the first; a direct caller met both.
- `sc-neurocore studio` stopped with SIGTERM, as a service manager or
  `docker stop` stops it, now ends as a clean return and runs its exit
  handlers. Uvicorn drains the server on the signal, restores the handler it
  found and raises the signal again; under Python's default action the
  process then ended before any exit handler, so an unconfigured Studio left
  its private scratch job root in the temporary folder, and the ledger's
  connections were not closed. The installed-wheel acceptance stops the
  Studio this way, which is why its distribution test left one root behind.
  The Studio holds its SIGTERM handler only while it serves and restores the
  one it found when the server returns, so a Studio run inside another
  process leaves that process's SIGTERM as it was.
- A model descriptor's `[dynamics]` entry must be its expression text, or a
  table that holds only that text under `expr`; anything else is refused,
  naming the entry. A number, a truth value or a list used to be left out of
  the parsed descriptor without a word, a table whose `expr` was not text read
  as an empty equation, and a table's other keys were dropped. Every shipped
  descriptor gives its 420 dynamics entries as text, so none is refused.
- The benchmark coverage census reads the source digests a record keys by
  repository path under `source_sha256` or `source_files_sha256`, where the
  record binds no path under `source_hashes` and the evidence gate's manifest
  does not map it. Thirteen committed records it counted as `unbound` now carry
  a currency: 4 are `current` and 9 `outdated`, each by changed sources; none
  is gone. A field whose keys name no folder still binds nothing:
  `bench_chiplet.json` keys its sources by bare file name, and read from the
  repository root every one of them would look removed. Over all records the
  totals move from 36 current, 91 outdated and 191 unbound to 40, 100 and 178;
  the cited records and the model totals are unchanged.

- The network's Rust dispatch check remembers its state probe for a model
  name together with the class the name resolves to. Remembered by the name
  alone, a class put under a catalogue name kept answering for it until the
  process ended: after a test put a stub under `SCLapicqueLIFNeuron`, a
  default population of that model read "cannot be stepped" and ran in
  Python, on whichever test worker that happened.
- `ARCHITECTURE.md` and `ROADMAP.md` give the Q8.8 SHD reference's accuracy
  as reported on 2026-04-11, 76.06% against PyTorch's 75.66% on the
  2264-sample test set, with its source and no committed record, in place of
  a 4% gap. The directory map and data flow of
  `ARCHITECTURE.md` name directories without counts and point to the
  generated capability snapshot; a test keeps counts out of them. The
  architecture contract table says that the hosted SHD step also runs the
  whole core on synthetic weights against a fixed expectation. The research
  labs guide names all five runtime dependencies, which a test reads from
  `pyproject.toml`.
- The dendritic NMDA neuron's descriptor names its integration method RK4,
  the path its class runs by default and every backend lane uses; it said
  Euler, the generator's default for a model without a v1 schema.
- At interpreter exit the Studio job ledger closes only the connections no
  live thread can be using: those of the exiting thread and of threads that
  have ended. It closed every connection before, also that of a process-job
  supervisor still running in a daemon thread, whose next ledger write then
  raised `sqlite3.ProgrammingError: Cannot operate on a closed database` in
  the thread. A connection a live thread still holds is released by the end of
  the process.
- The Rust safety crate compiles three source-parity files it did not declare:
  `sc_compte_wm_network.rs`, `sc_resetting_mat.rs` and
  `sc_unit_capacitance_respiratory.rs`, so their tests run with the crate's
  tests. `safety/mod.rs`, which no build reads, no longer holds a second module
  list; that list had drifted from `lib.rs` by three names. The file stays, with
  only a comment. A test fails when a `.rs` file in the folder is neither
  compiled nor one of the seven codec transcripts kept as unsupported source,
  when `mod.rs` declares a module, or when a declaration of `lib.rs` depends on
  a `cfg`; comments count for nothing.
- A Studio started without a configured job root removes its private scratch job
  root when its process exits normally; only the process that made the root
  removes it, and a configured root is kept. A process killed outright, as by
  SIGKILL, still leaves its root. Each such start left one
  `sc-neurocore-studio-jobs-*` folder in the temporary directory: one run of the
  112 test files that build the Studio app left 287.
- The RTL compile of an equation neuron refuses in words what the RTL cannot
  bind: an equation that reads a name which is not a state, a parameter or a
  constant (such as `dv/dt = -v / tau` with no `tau`) compiled to RTL that read
  an undeclared net and did not elaborate, and an unknown function or a reset
  of a name that is not a state failed with unmarked faults. The ODE editor's
  'Generate RTL' answers each of these with HTTP 422 and the reason, in place
  of HTTP 500, by running the compile task's own parse and emit before
  admission. One equation compile is bounded before the RTL emitter runs, on
  `/api/compile` and `/api/ir/emit-sv-direct` alike: at most 128 function
  calls and 16,384 characters over its equations, threshold and reset, refused
  in words over either. The co-simulation receipts of the four bound silicon
  facets are recorded again on the new compiler. Notebooks 27 and 29 pass
  `V_th` and `V_reset` as parameters, and notebook 27 is run by a test.
- The ODE editor's 'Generate RTL' answers an equation, threshold or reset it
  cannot parse with HTTP 422 and the parser's reason, for example "Invalid
  equation syntax: '-65 = (-v + 1) / 10'", in place of HTTP 500
  "studio_job_failed". The compile runs the worker's parse before it admits a
  job, so no job is started for such a request; a fault that is not the
  caller's input is still left to the job.
- A catalogue model compile the compiler refuses is answered HTTP 422 with its
  reason, in place of HTTP 500 `studio_job_failed` and "Studio job failed.",
  by 'Generate RTL' and the selected-model co-simulation. Sent as the Studio
  sends them, 10 of the 64 models that offer 'Generate RTL' were refused that
  way: nine by parameter names their schema does not know, one by a default
  Q-format that cannot hold it. A map compiled at a step other than its own is
  refused too. The request is checked before a job is admitted with the
  compiler's own resolver; a fault in the server's own compile configuration
  still goes to the job.
- 'Generate RTL' compiles the nine catalogue models whose parameters the Studio
  names differently from their canonical schema, which were refused as
  "Unknown schema parameter override(s)". The compile translates each as the
  model's class and its schema both state: a rename (AdEx's b, c_m and delta_t
  are the schema's b_adapt, C and delta_T; Brunel-Wang's C_m; the
  Hodgkin-Huxley and Wang-Buzsaki conductances; Resonate-and-Fire's
  threshold), a derivation (Jansen-Rit's four couplings from c; the Amari
  field's sampled kernel weights from its kernel parameters and dx), and the
  seed of the escape-rate and Poisson neurons, which seeds the schema's 16-bit
  LFSR and is recorded with the compile, in its traceability input and in the
  co-simulation cross-check. The Amari field compiles only as the schema's
  fixed field of four sites, so its default of 64 sites is refused with that
  reason, and a request in its catalogue names samples the kernel at the
  request's own dx. A seed the LFSR cannot take, or one parameter sent under
  both of its names, is refused in words.
- 'Generate RTL' offers a model only the Q-formats that hold the neuron the
  Studio compiles, judged on the catalogue's parameters the Studio sends, not
  the schema's defaults. For the scaled-reset adaptive integrate-and-fire
  neuron, Q8.8 was offered as the default and then refused, since it cannot
  hold tau_2 = 200; Q16.16 is now its only offered format and its default.
  Across the catalogue, every offered format compiles the request the Studio
  sends.
- A Studio storage-profile job whose worker exits before asking for its grant
  fails at once as "Studio process worker exited with N before requesting its
  grant.", in place of waiting out the grant deadline (60 s by default) and
  reporting "Studio worker grant failed.". The supervisor asks the launcher,
  while it waits, whether the worker still runs. A grant endpoint path too long
  for a Unix socket is refused with its length.
- A Studio worker that connects for its grant and exits before the grant is
  given is reported as "exited with N after requesting its grant, before it was
  given"; within one poll of the launcher's report it was reported as never
  having asked.
- The Studio's storage service stops on SIGTERM or SIGINT whatever signal mask
  the process that starts it hands down: it unblocks both at start, as the
  worker launcher service does.
- The Studio worker launcher service unblocks SIGTERM and SIGINT when it
  starts, so an inherited blocked-signal mask cannot keep it from stopping in
  order. The launcher test support places its sockets under a folder short
  enough for the Unix socket path limit, kills and reports a launcher that
  does not stop on SIGTERM, and on Linux ends a launcher together with the
  test that started it.
- Screen readers and speech input get the Studio's own names. A model in the
  library is named by the label it shows, such as "AdEx", with its contract
  line (validation, integrator, terminal tier) as its description, in place of
  the label run into that line ("AdExvalidation parity ..."). The RTL panel's
  co-simulation controls are a named group, and its read-only RTL viewer is
  named "Generated RTL" in place of "Editor content".
- The Studio announces what an action did to screen readers. The bottom bar is
  always drawn and its run report is a status: a model run starting and
  finishing, and a finished network canvas run with its neuron, spike and step
  counts, are announced, without the shortcut hint. The bar wraps in a narrow
  window and no longer overflows the page at 320 CSS px. The model list's
  count, the network canvas's population and projection count, the training
  run's state ("Training running", "Training stopped") and the evidence cart's
  count are statuses too. The cart's last export says that it passed its own
  round-trip verification, and an export failure is announced in the same
  place.
- On the network canvas and the E-I network tab, the status bar reports only
  that tab's network, not the last single-model run. A run refused before it
  starts because the network has problems says how many problems were found,
  and a run the server refuses says so, with its reasons.
- The evidence cart says in words why a finished run was not queued, in place
  of passing on a code such as analysis_result_identity_invalid. A run or
  analysis whose result carries no readable digest is named as such, in place
  of being skipped without a word or called unchanged; a failed or unchanged
  run still says nothing. The export's read-back check names a failure in
  words, in place of "payload_sha256 mismatch": which entry, or the bundle, no
  longer matches its SHA-256 digest, and that the cart was not exported.
- Tab and Shift+Tab leave the ODE equation editor, as everywhere else in the
  Studio; the editor kept them, with only its own Ctrl+M toggle as a way out
  (WCAG 2.1.2). The editor moves focus on Tab, and is named "Equations" after
  its visible heading, in place of "Editor content".
- The keyboard focus is visible and lands somewhere named. When the first-run
  tour closes, by Escape, Skip tour or its last step, focus moves to the
  Studio's title, not the page body. The focus ring, which measured 1.44:1
  against the panel behind it, is a 2px band of the accent colour framed by a
  2px gap, and the palette check holds it to 3:1 on every control surface.
  Under forced colours (Windows contrast themes), focused controls draw an
  outline in the system's highlight colour, where they showed no indicator.
- Typing an equation in the ODE editor no longer scrambles it. As soon as a
  typed line read as an equation, such as `dv/dt` after selecting all, the
  Studio's rendering replaced the text under the caret and moved the caret to
  the end, so the rest of the line landed in the reset expression. The editor
  keeps the typed text for as long as the Studio holds exactly what it reads,
  and still shows a change made elsewhere, such as a chosen template.
- The equation editor keeps every `/` and `?` typed into it. In
  Chromium-based browsers the keyboard help took both keys whenever focus was
  not in an input or a text area, which the editor (EditContext) is not, so
  `dv/dt` could only be pasted. The help's `?` and `/` act only when focus is
  outside a text control (fields, selects, editable regions and anything with
  a text-entry role); on buttons and links they still open it, and Escape
  closes it wherever focus is.
- Clearing a training setting no longer blanks the Studio. After a completed or
  failed run, clearing an optional setting (choosing another dataset, clearing
  the seed or the gradient norm limit, switching back to the spiking model,
  withdrawing the preregistered criterion or clearing the event input) left it
  in the configuration as `undefined`, which the configuration seal refuses. A
  cleared setting now leaves the configuration, as a saved project is read
  back, and a test clears each optional setting.
- Exporting a saved project's evidence bundle works after a network canvas
  projection's seed is emptied or its rule is set to all_to_all. Both edits
  left the field as `undefined`, and the browser refused the export with
  "undefined is not a sealable JSON value."; a cleared field now leaves the
  projection, as a saved project is read back.
- The Morris-Lecar Mojo kernel compiles under the pinned Mojo 1.0.0: its
  functions are declared with `def`, which replaced `fn`. Its own check
  reproduces the Python reference's spike counts (0, 3 and 5 over 2000 steps
  at currents 0, 50 and 100), and in the five-runtime Morris-Lecar benchmark,
  run again, every backend counts 476 spikes; the model page shows the new
  timings.
- The sign of the NMDA current in the dendritic NMDA neuron: the current
  `g_NMDA * glutamate * B(V_d) * (V_d - E_NMDA)` was added to the dendrite's
  equation, so below `E_NMDA` glutamate lowered the dendritic potential and
  removed spikes. It now depolarises below `E_NMDA`, as in the library's other
  NMDA models, in every lane: Python, the Rust engine, the Rust safety mirror,
  Go, Julia and Mojo. With the default parameters a somatic current of 20
  alone and glutamate alone stay below threshold, and both together fire. The
  20,000-step anchor at current 50 and glutamate 0.5 moves from 253 to 266
  spikes. Results computed with this model before the fix have the glutamate
  effect inverted. The tests hold two-sided bounds on the spike counts and the
  dendrite's balance point solved from the documented equations. The Mojo
  kernel is rewritten for Mojo 1.0.0, the model's page states the sign
  convention, corrects the coincidence table and names the test files that
  exist, and the benchmark record is measured again.
- `low_discrepancy_error_bound(length)` returns `1/N` only for a power of two,
  where the first `N` radical-inverse points form the grid `j/N`, and refuses
  any other length with the new `GuaranteeNotDetermined` (a `ValueError`); it
  returned `1/N` for every length, although five Halton points encode 0.3 as
  0.6. The new `threshold_encoding_error_bound(points)` returns the exact
  worst-case decoding error of a point set (its one-dimensional star
  discrepancy).
- `bipolar_decode` and `bipolar_multiply` check their bit arguments in the
  representation the caller gave, before converting them to one byte per bit;
  `[0.9, 1.9]` was decoded as the bits 0 and 1, `256` became 0 and `NaN` became
  0 with a warning. Such input, and text, bytes, objects and complex numbers,
  is refused with a `ValueError` that names the argument; exact 0/1 values in
  boolean, integer or real containers of any shape give the same result as
  before. The check is one function, `sc_neurocore.core.bit_input.as_bit_array`,
  shared with the cross-correlation diagnostics, and the two bipolar functions
  are annotated as taking any array-like input.
- The stochastic cross-correlation diagnostics (`estimate_scc`,
  `observed_and_bias`, `correlation_diagnostic`) check their input in its
  original representation before converting it to bits. Fractional values,
  integers other than 0 and 1, `NaN`, infinity, text, bytes, objects and
  complex numbers are refused with a `ValueError`; exact 0/1 input in boolean,
  integer or real form gives the same result as before.
- The benchmark coverage census reads the files Git tracks, staged ones
  included, so a local run left beside the committed records, or a folder Git
  ignores, no longer changes its counts. Outside a repository (a packaged or
  temporary tree, or where no `git` program can be started) the directories
  are read as they are. The evidence gate gains `tracked_files`; a tracked file
  whose name is not UTF-8 is listed in the form a directory walk gives it. The
  committed census is unchanged.
- Every subpackage that the wheel ships imports from an installed wheel:
  `sc_neurocore.identity` and `sc_neurocore.experimental` import
  `sc_neurocore.analysis` and `sc_neurocore.physics`, which the wheel leaves
  out, where they are used rather than at module level. A feature that needs
  one of them, such as the identity substrate's health check or the
  heat-diffusion comparison route, names the missing package and says that it
  ships only in a source checkout; a missing third-party dependency is raised
  unchanged. A test blocks every package the wheel leaves out and imports
  every shipped subpackage.
- Every job-ledger database connection is closed when the Studio application
  shuts down, including the connections opened by worker threads that have
  ended; Python 3.13 and later reported them as unclosed-database warnings in
  unrelated code.
- `compile --synthesize` and `deploy` report a failed open-source synthesis
  flow as a failure: a nextpnr place-and-route error or a bitstream packer
  error exits with status 1, as a Yosys failure already did, in place of
  printing "PnR failed" and exiting with 0. A step that exceeds its time limit
  (300 s for Yosys and nextpnr, 60 s for packing) is reported instead of
  raising a traceback, a missing nextpnr or packer is reported as a skipped
  step, and `deploy` no longer says "Yosys not found" when an installed Yosys
  fails.
- Validator interruptions and bounded process cleanup are preserved when a
  group leader exits or a detached output writer retains its stream.
- The README's MNIST figure for the surrogate-gradient example names ATan, the
  example's default surrogate, and the committed 10-epoch benchmark of the
  same network that reached 95.9 %, in place of ~95 % for FastSigmoid,
  SuperSpike and ATan alike.
- The hand-written test matrix of VALIDATION.md and one workflow row of the
  README. The Python row points to the README's generated test-file count, in
  place of 3,376 tests in 145 files; the formal row gives the 91 SymbiYosys
  proof jobs under `hdl/formal/`, in place of 67 checks in 7 files; the Rust
  engine row is CI-listed; the engine tests run on Ubuntu, Windows and macOS.
  The generated coverage-scope block is regenerated. The README describes
  `benchmark.yml` as a workflow started by hand with a run reason and change
  reference, not as performance regression tracking.
- Statements on the documentation home page and two overview pages that the
  code contradicts or no record supports. The MPI line and the
  getting-started guide no longer claim billion-neuron runs. The analysis
  toolkit has 25 modules and the visualisation package 12 plot functions, both
  marked as source-checkout only. The equation compiler is described as
  handling supported ODE models, with a link to the supported modes. The NIR
  paragraph names the other ecosystem members as such and no longer calls the
  FPGA backend the missing one. The product overview calls the package's entry
  points supported, matching its beta status, and the market page states
  SC-NeuroCore's own focus without an unsourced comparison with other
  frameworks.
- README statements that the code or its records contradict. The Zenith
  quickstart passes the one scalar weight the Rust plasticity layer accepts,
  so the snippet runs, and says that no call yet copies trained Torch weights
  into the Rust layer. The example table describes the Zenith demo by what it
  does. The core tier is the supported core surface of the beta package. The
  Rust engine section gives the largest committed scaling run, 100K neurons,
  says which SIMD paths are chosen at run time and that the SVE and RISC-V V
  paths are compiled only on request. Two crate test counts match
  `cargo test -- --list`: 22 for autonomous_learning and 34 for
  stochastic_doctor_core.
- The SCPN example is described by what it builds: the README row and the
  docstring of `examples/05_scpn_stack.py` call it a 16-layer SCPN (Stochastic
  Coupled Phase-oscillator Network) stack with inter-layer coupling and global
  coherence metrics, in place of seven layers.
- No page claims that the event-driven neuron's power is proportional to spike
  rate; no power measurement of `hdl/sc_event_neuron.v` exists, and its leak
  counter and threshold compare run on every clock. The README, the
  architecture file and the documentation home page say that it integrates on
  input events and decays on leak ticks. The RTL header drops an unmeasured
  saving of more than 99 % and an event-address check the module does not
  have. `examples/power_comparison.py` derives its summary from its own rows
  (both counts fall with activity; a 38-fold reduction), labels its figures as
  toggles, not power savings, and a test keeps the summary bound to the rows.
- No page claims hardware results that no record supports: the system map's
  SHD pipeline proven on silicon, the research-labs guide's spikes identical on
  silicon, and the formal page's safety monitor enforcing its invariants in
  silicon at 100 MHz or more on the PYNQ-Z2. Each page states the
  co-simulation and synthesis evidence and that no board run has validated it
  yet. The formal page lists the monitor's six runtime checks and
  violation-flag bits, and notes that its precision check is a simplified
  range check. A test finds these claims without regard to case, line breaks
  or emphasis.
- The README, the external formal evidence page and the formal workflow's job
  summary say that full formal qualification remains incomplete, or is not
  claimed, until the `formal-all` gate is green, in place of saying that Core
  4.1.0 is blocked by it.
- Documentation errors that a record proves. The delay-training page names
  the published authors of its first two references and the venue of the
  first, the co-simulation guide lists the models and tolerances its tests
  enforce, the Studio guide counts its 16 analysis views, and the README gives
  the firing-rate range of the non-parity Rust rows at 10,000-100,000 neurons
  as the artefact records it (3.2-17x above Brian2). The hardware engineers'
  guide adds current Yosys and nextpnr measurements beside the figures for
  which no source was found; no figure is removed.

### Removed

- The internal Rust safety crate no longer exports seven placeholder codec
  modules that returned constant values (`aer_codec`, `codec`, `delta_codec`,
  `entropy`, `predictive_codec`, `streaming_codec`, `waveform_codec`). Their
  source files stay in the repository as unsupported source and are not
  compiled, and the crate is marked `publish = false`. The production Rust
  codec components are those of the engine crate; the Go, Julia and Mojo codec
  mirrors stay unsupported sources, as the mirror-authority guide says. The
  engine workflow runs the safety crate's tests, among them seven doctests
  that must fail to compile if an export returns.

### Repository hygiene

- A test holds every crossing-count case model to the direction rule: a
  model whose step reports an upward crossing of a threshold with no reset
  must state a spike case that holds more upward crossings than downward
  ones, so a step that counted the other direction would break it. The test
  runs each spike case through the public class, checks that every event it
  returns is an upward crossing of the variable it names, and requires every
  case model to be classified as such a model or as a reset or level one. It
  holds the nine crossing models, among them the four whose direction cases
  were added for it (FitzHugh-Rinzel, Hindmarsh-Rose, Pernarowski,
  Wilson-HR).
- The storage service's stop-on-SIGTERM test runs on every POSIX host. It
  was skipped off Linux because it reads the service's blocked-signal mask
  from `/proc`; that one check now runs where `/proc` has the process, and
  the stop itself is checked everywhere the service runs.
- The configured pre-commit hooks check the benchmark coverage census on every
  commit (`benchmark-coverage-census-drift`), so a commit that changes a source
  a benchmark record binds cannot carry a stale census; the preflight's
  pre-commit range runs it before a push. `CONTRIBUTING.md` says when to write
  the census again.
- The Studio API context builder is split by responsibility: the audit sink,
  the identity store, the browser session guards, the job manager, the
  assembly of the context and its exposure on the app state each come from
  their own function, and the synchronous process-job runner hands its receipt
  and its refusals to two more. No function of the module exceeds 50 lines,
  which a test holds; startup makes the collaborators in the same order as
  before.
- The hosted CI holds the committed benchmark evidence gate report to what the
  gate writes: the lint job's gate step compares its output with
  `benchmarks/results/benchmark_evidence_gate_report.json` and fails, naming the
  file and showing the difference, when they differ. Three tests run the step's
  own script with a stand-in gate.
- The changelog's unreleased entries are sorted into the standard categories,
  Security, Added, Changed, Fixed, Removed and Repository hygiene, and state
  the change without its rationale. A test holds every unreleased entry under
  one of these headings, in that order, with no heading left empty.
- A Go accelerator build puts its library in place as a new file: the builder
  copies it beside its target under a temporary name and renames it over the
  target, in place of writing into the existing file, which changed the mapped
  code of a process with the former library loaded and wrote through a
  symbolic link to the file it named. The library keeps the mode the build
  gave it, and a copy that fails leaves the former library and no partial
  file.
- The Go accelerator builds no longer write over the committed C headers:
  `tools/build_accel_backends.py` and the CI and audit workflows build into a
  temporary folder and copy only the library. The builder compares cgo's
  header with the tracked one and names any that differ, without writing them,
  so in CI the five loading tests compare a fresh build's header with the
  committed one.
- The hosted CI lints the Studio frontend: the frontend job runs `npm run lint`
  after installing and before the tests, and the script fails on a warning
  (`--max-warnings=0`), an unused `eslint-disable` comment included. A
  contract test holds the job's step, the script and the preflight's lint to
  one command.
- Thirteen Studio components get a test of their own: the admin panel and its
  audit-archive section, the capability strip, the cohort panel, the model
  comparison, the custom-equation editor, the model capability list, the
  network controls, the synthesis evidence controls, the template list, the
  text prompt, and a training run's event input and preregistered criterion.
  Each test mounts its component and drives it as a user does, and a
  deliberate fault in each component's source made its test fail. Ten of the
  components are at 100 % of their statements and branches; the admin panel,
  the text prompt and the model comparison stop short only at defensive guards
  no user action reaches. The frontend coverage ledger records the gains, six
  files that earlier tests had raised, and the render-timing module at 100 %.
- The Studio frontend has a coverage ledger:
  `studio/frontend/coverage-ledger.json` records the statement and branch
  coverage of every source file under `src`, and `npm run
  test:coverage-ledger` runs the suite with v8 coverage and refuses a file that
  falls below its recorded value or a new file below 100 %.
  `scripts/coverage-ledger.ts accept` writes the ledger again only when nothing
  fell. The CI frontend job runs the check after Vitest and keeps the measured
  summary as an artefact.
- `.githooks/pre-push` is stored executable, so the hook that
  `make install-hooks` installs runs, and it runs the preflight's default (the
  gates that guard what the push touches, and no test suite) in place of
  `--all-gates`. A test pushes through the stored hook and checks that the
  preflight runs and that a failing preflight stops the push.
- `tools/preflight.py` checks what a change touched and runs no whole test
  suite by default. Each gate declares the kinds of file it guards; a gate the
  change does not touch prints `not applicable to this change: <gate>` and does
  not run. The capability manifest, SPDX header and benchmark evidence gates
  always run, and every gate runs when the touched files cannot be determined
  or the tool itself changed. A `frontend` gate runs, for a change under
  `studio/frontend/`, the frontend's lint, its type check and the Vitest tests
  related to the touched files, and fails when `npm` or `npx` is missing.
  `--all-gates` runs every gate; `--full-suite` adds the whole pytest suite
  (`make preflight-full`); `--no-tests` stays accepted as the default.
  `sc-neurocore preflight` keeps running every gate and the whole suite.
- The preflight checks the whole push range: it finds the commit a push
  starts from, runs the documentation scope guard from it (`--since`), and runs
  the pre-commit hooks over the files changed in the range, or over all files
  when a hook configuration changed, as the hosted pre-commit run does. The
  hooks receive the same commit through `SC_NEUROCORE_POLICY_BASELINE`, so the
  Go and Rust documentation ratchets compare with the push base. The range
  gates refuse a checkout with uncommitted or untracked changes.
- The dependency-lock audit retries a transient OSV failure: a transport
  failure (such as a timeout or a reset connection) or HTTP 429, 500, 502, 503
  or 504 is retried after 2 and then 6 seconds, three attempts in all; a failed
  certificate check and any other status, such as a 400 refusal, are final at
  once. Each page's receipt lists every retried attempt with its exception
  types or its status and body digest, and each retried body is kept beside
  the final one. Exception text and headers are still not kept.
- The shared HDL toolchain workflow restores one published SDK archive in
  place of building the tools on each runner: the `scn-hdl.tar.gz` asset of
  release `hdl-sdk-yosys070-yices270-524357fd`, SHA-256
  `49a65847681b69ae5a42f2801562d464f7d9d52af6c87e7744c180a6a70668c3`. The
  workflow checks the archive's hash, refuses members outside `scn-hdl/` and
  anything but plain files and folders, checks the Yices and Z3 binaries
  against the receipts' digests, runs the same version, licence and smoke
  checks, and hands the archive on unchanged. A build from source runs only
  when a caller passes `rebuild_sdk: true`. The release also carries the Yices
  and GMP sources the binary was built from.
- The Rust documentation ratchet extracts its original revision into one
  persistent folder outside the checkout, `sc-neurocore-rust-doc-original` in
  the temporary directory or the one `SC_NEUROCORE_RUST_DOC_ORIGINAL_DIR`
  names, so Cargo's build outputs persist and Z3 is not compiled from C++
  source on every run; the sources are extracted afresh each time. The folder
  is created for the current account alone (mode 0700); a link, a folder
  another account could write, and a non-empty folder without the ratchet's
  marker are refused. An exclusive lock lets concurrent runs take turns, and a
  waiting run says so.
- The Rust documentation measurement runs the toolchain the measured crate
  pins (`engine/rust-toolchain.toml` for the engine) in every compiler call,
  also for the original revision, whatever toolchain, compiler or compiler
  wrapper the environment or a Cargo configuration file names. A pinned
  toolchain that does not run, or a compiler that does not report exactly the
  pinned release, is refused by name, and rustup may not fetch one.
- The body of the shared `safe_rate` helper of the biophysical neuron models
  moves into its own module, `biophysical/rate_functions.rs`; the facade
  `engine/src/neurons/biophysical.rs` keeps a documented `safe_rate` that calls
  it, and its line ceiling in the engine monolith guard is lowered from 54 to
  50.
- `requirements.txt` and `requirements-dev.txt` install everything the package
  needs: the runtime file lists `defusedxml`, `tomli-w` and
  `typing-extensions`, and the development file the six development
  dependencies it lacked and the exact `mypy` and `ruff` versions of the `dev`
  extra. A test keeps both files in step with `pyproject.toml`.

## [4.1.0] - 2026-11-08

- The curated summary of this release is `docs/development/RELEASE_NOTES_v4.1.0.md`, which is also the GitHub Release text.
- Accept rounding differences between processors in `tools/ice40_up5k_calibration.py --check`.
  NNLS and the BLAS kernels under it round the last digits of a fitted constant
  differently on different CPUs, so a hosted runner found the generated Rust
  constants stale. The check now compares the text outside the floating-point
  literals exactly and each literal within 1e-9; the generated constants are
  unchanged. `tools/export_shd_int_network.py` loads its checkpoint with
  `weights_only=True` and reproduces the committed network byte for byte.

- Align the dependency locks with current upstream releases (2026-11-08).
  Every hash-locked Python profile is regenerated with its own recorded
  command; pinned tools move to ruff 0.17.4, semgrep 1.184.0, cyclonedx-bom
  8.0.1, click 8.5.1, hypothesis 6.170.15, mkdocs-material 9.8.1, jupyterlab
  4.6.6, notebook 7.6.5 and the current setuptools stubs. The Rust locks take
  cc 1.11.1 and smallvec 1.16.5, and the fuzz workspace follows the engine to
  z3 0.22. The Studio frontend takes the current Vite, Vitest, Playwright,
  module-federation, React Flow, Zustand and eslint-plugin-jsdoc releases.
  The release and container workflows pin action-gh-release 3.0.4 and
  login-action 4.8.0, and the scanners move to Syft 1.57.0, Grype 0.121.2
  and Trivy 0.76.2 (checksums and image digest pinned). Recorded holds are
  unchanged: Rust 1.99.1, TypeScript
  7, uv 0.13.6, brian2 2.11, MCP 2, Mojo 1.2, the Sinabs NIR range and
  Semgrep's click range.

- Add the trained integer SHD network for the bit-exact core (2026-11-08).
  `hdl/fixtures/shd_int_network.json` is the integer
  form of a DCLS-max checkpoint trained with sharp delays. It runs bit for
  bit on `hdl/sc_shd_int_core.v`; `shd_int_network.README.md` records the
  checkpoint and export provenance. `tools/export_shd_int_network.py` reproduces
  the file from the checkpoint. `tests/test_shd_int_network.py` pins its
  digest and runs it on the core. The core testbench waits out the vendor
  global set/reset when it drives a synthesised netlist, and the training
  script cites the learnable-delays paper as arXiv 2306.17670.

- Add a bit-exact integer SHD core and its reference (2026-11-07).
  `tools/shd_int_reference.py` defines the integer-delay SHD network in integers:
  - 4-bit counts in, int8 weights;
  - delays 0..30;
  - LIF with an arithmetic-shift leak and integer thresholds;
  - every width bounded for inputs of up to 1024 steps.
  `hdl/sc_shd_int_core.v` computes it bit for bit, in an event mode and a
  full-scan mode with identical results. `tools/cosim_shd_int_core.py` and
  `tests/test_shd_int_core_cosim.py` compare every accumulator with the reference.

- Place and route hierarchical designs from `deploy` (2026-11-07). Yosys 0.70
  leaves `$scopeinfo` debug cells in a flattened hierarchy, which nextpnr 0.6
  cannot place, so such a design printed "PnR failed"; deploy synthesis now
  deletes them. The generated Makefile uses each family's own place and route
  and packer (ECP5: `--textcfg` and `ecppack` to `.bit`), and its Yosys command
  keeps the `$` of `$scopeinfo` from the shell.

- Place and route ECP5 designs in `compile --synthesize` and `deploy`. Both
  passed nextpnr-ecp5 the iCE40 option `--asc`, which it does not accept, so
  every ECP5 run reported "PnR failed" after synthesis. ECP5 now writes the
  Trellis text configuration (`--textcfg`, `MODULE.config`) and `ecppack`
  packs it into `MODULE.bit`; iCE40 keeps `--asc` and `icepack`.

- Emit SystemVerilog for a device with that device's resource report, and judge a
  graph against real Yosys and nextpnr runs (2026-11-07).
  `ScGraph.emit_sv_with_report(target, clock_mhz)` returns, for `ice40:UP5K`, an
  iCE40 UltraPlus UP5K report with budgets and, for one calibrated block inside its
  measured domain and in the measured operand form (dense layer, LIF neuron,
  power-of-two synapse stream), a static bound on routed core logic cells and
  critical path; any other graph, including one with port or other constant
  operands, carries the reason instead of a number. Zynq targets keep the heuristic report, now labelled
  uncalibrated. The bound is generated from
  `benchmarks/results/ice40_up5k_calibration/designs.json` by
  `tools/ice40_up5k_calibration.py`, which refits it, checks every recorded design
  and weight probe against it and prints the figures its documentation cites.
  `sc-neurocore resource-check GRAPH --clock-mhz N --timeout-seconds S` (Python:
  `sc_neurocore.studio.resource_feedback.run_resource_feedback`) synthesises the
  graph, places and routes it out of context in a five-pin harness on the SG48,
  and reports package I/O, logic capacity, timing and the estimate as pass,
  violation or undetermined, exiting 0, 1 or 2. `evidence-packet export` adds the
  place-and-route report and the estimate set against the routed core to the
  packet's manifest (`--clock-mhz`, `--pnr-timeout-seconds`). Studio place and
  route names a design that needs more I/O pins than its package has, or more
  logic than its device has, and Studio synthesis deletes the `$scopeinfo` debug
  cells Yosys 0.70 emits, which nextpnr 0.6 cannot place. CI installs nextpnr-ice40
  0.6-3build5 and runs the real flow.
- Fix `sc-neurocore compile --synthesize` for `ice40` and `ecp5`. It handed
  Yosys the layout that `deploy` writes (`hdl/*.v` and a top `.sv`), which
  `compile` never writes, so it stopped with `FileNotFoundError`. It now
  synthesises the `MODULE.v` it wrote. A Yosys failure is reported as a failure
  with exit status 1; before, any unsuccessful run was reported as a missing
  Yosys.

- Take one catalogue model from its schema to synthesis with one command.
  `sc-neurocore hardware-path MODEL -o DIR` (Python:
  `sc_neurocore.studio.hardware_path.run_hardware_path`) checks the model's
  capability matrix (compile, co-simulation and synthesis required), resolves
  its configuration and writes the RTL with its compile traceability,
  co-simulates the RTL cycle by cycle against the generated bit-true C kernel,
  writes a guarantee summary, and synthesises with Yosys; for `ice40` and
  `ecp5` with nextpnr installed it also places and routes through the
  digest-bound Studio terminal, otherwise place and route is recorded as not
  run with the reason. Each stage refuses by name with a code (unknown model,
  a disabled operation, an undeclared or uncosimulated integrator, a format
  that cannot hold the model, a configuration, current or step count the
  resolver or co-simulation rejects, a missing or failing tool, a design the
  device cannot place); a co-simulation that is not bit-exact writes its report
  and stops before synthesis. The guarantee summary uses the IR summary's
  classes: the co-simulation is `CO-SIMULATED ONLY` with its boundary and
  uncovered cases quoted; the model's catalogue formal job is `FORMAL` only when
  its RTL file holds exactly the bytes the catalogue emitter writes for this
  RTL, `PROVEN AT OTHER PARAMETERS` with both parameter sets otherwise, and
  `NONE` without a job (no job is run); readiness labels come from the sealed
  readiness record. `hardware_path.json` and `hardware_path.txt` record every
  stage, the SHA-256 of every artefact, the tool versions and what the run does
  and does not establish: equality with the bit-true kernel at the chosen
  format for the recorded stimuli, not with the floating-point model; formal
  results only as bounded safety on a job's own RTL; synthesis and timing
  figures as tool estimates; no measured hardware. `sc-neurocore hardware-path
  --list [MODEL ...]` prints the supported set on the host, and
  `tools/hardware_path_support.py --write|--check` keeps
  `docs/_generated/hardware_path_support.md` (41 of 185 catalogue models
  supported) derived from the capability matrix.

- Build each lookup table of the function-accuracy measurement once per format
  instead of once per read. `lut_read_int` rebuilt and saturated the whole table
  for every input code, so measuring a function over many codes rebuilt it as
  many times. The Studio's model detail spent most of its time there: the
  capability matrix of ConnorStevensNeuron took 115.9 s and now takes 1.1 s on
  this host, with an identical result. The function-call-domain and
  equation-IR benchmarks are re-recorded against the changed source.

- Export a hash-bound evidence packet for one generated dense layer design, and
  verify it. `sc-neurocore evidence-packet export GRAPH -o DIR` accepts an IR
  graph of one `sc.input` of `vec<rate,N>`, constants, one `sc.dense_forward`
  and one `sc.output`, and refuses any other operation by name. It writes the
  IR and SystemVerilog, the guarantee summary, the Q formats and every encoder
  seed, a cycle-by-cycle co-simulation of the emitted top level under Icarus
  Verilog against a new bit-true Python reference
  (`sc_neurocore.safety_cert.dense_cosim`, N inputs by M neurons; tested from
  2x2 to 4x3 at state widths 12 to 64), the committed formal jobs of the library modules involved run in a
  scratch copy outside the checkout, a Yosys report, tool and source versions,
  a statement of what the packet does and does not establish, and a manifest of
  SHA-256 digests. A formal job or synthesis whose tools are missing is
  recorded as unavailable with the reason; only a formal `PASS` is listed as
  established, and a job that did not run, did not pass, or proved a source
  other than the simulated primitive is listed as a gap. `sc-neurocore
  evidence-packet verify DIR` refuses a missing, extra, altered or symbolically
  linked member, a directory that holds no member, an entry that is not a
  regular file, and a manifest digest mismatch. Tests export packets for real 2x2 and 3x2 designs,
  find planted faults in the reference configuration at their first differing
  cycle, and refuse every tamper class.

- Refuse an equation that overflows Python's parser instead of failing with an
  internal error. Some texts far below the 16,384-character limit make CPython's
  parser raise `MemoryError` ("Parser stack overflowed"), for example 3,000
  chained powers in 8,998 characters, on Python 3.11 to 3.14. The equation gate
  caught only `RecursionError`, so the Studio candidate routes answered such a
  text with HTTP 500. The gate now refuses it as "Equation expression is too
  deep to parse", and numbering a neuron's random draw sites, which parses the
  raw text before the gate, leaves both parser failures to the gate.

- Write a guarantee summary next to the SystemVerilog emitted from an IR graph.
  The new `sc-neurocore compile-ir` command verifies the graph, emits the top
  level and writes `NAME.guarantees.json` and `.txt`: for each operation, the
  library module it instantiates and whether its own formal job proves this
  instance (mode, depth, engine, assumptions and parameters), proves it only at
  other parameters (both sets shown), or whether the module is only
  co-simulated, yields a stochastic estimate (stream length and seed; no error
  bound yet), or is inline logic with no separate proof. Tests bind the table to the IR operations, the
  emitter's instantiations and the committed formal jobs, and run the named
  jobs where the solvers are installed.

- Publish the engine wheels to PyPI through trusted publishing, as the Python
  package already does, instead of a stored API token. Every uploaded engine
  file then carries a provenance attestation; the 4.0.0 engine files had none.

- Drop the unused `pyo3-build-config` build dependency from the engine crate.
  The crate has no build script, so the dependency was never used; the
  published crate and the lockfiles no longer list it.

- Attach a signed SLSA build provenance attestation (Sigstore, through GitHub
  artifact attestations) to every GitHub Release file and to the release
  image pushed to GHCR. `SECURITY.md` shows how to verify them with
  `gh attestation verify`.

- Smoke-test every engine wheel before the PyPI upload on a runner of its own
  platform and with its own Python: Linux x86-64 and arm64, Windows and macOS,
  Python 3.11 to 3.14. Before, only the Linux x86-64 wheel for Python 3.12 was
  installed and imported, and the other 15 wheels were uploaded untried.

- Report library status and failures through module loggers instead of
  printing them: the Mojo kernel runner logs a failed build or benchmark as
  an error, the AER router supervisor logs its build, start and stop, and
  the Julia fusion solver logs its run and a failed run before raising it.
  `AERRoutingDaemon` accepts `router_dir` and `JuliaFusionSolver` accepts
  `julia_bin` to select another router directory or Julia executable. Their
  tests now run real processes instead of replacing the subprocess calls.

- Refuse an equation expression longer than 16,384 characters before it is
  parsed, with a message that states the length and does not repeat the
  text. The depth limit bounded nesting but not width, so a long balanced
  expression could make the parser, and a product of its literals, as large
  as its text. The longest expression of the model catalogue, and the longest
  symbolic derivative of its dynamics, have 738 characters. A validator can
  set its own bound with `max_length`, and an equation neuron applies the
  bound to every expression before any of its text is parsed.

- Record the engine-loader re-exports that 4.0.0 removed without a line in
  its notes. `bridges.quantum_annealing` no longer offers `get_ising_energy`,
  `get_batch_ising_energy`, `get_simulated_annealing` and
  `has_full_quantum_annealing_backend`; import them from
  `sc_neurocore_engine.quantum`. `optics.photonic_emitter` no longer offers
  `get_crosstalk_analyzer`, `get_crosstalk_bank_analyzer`,
  `get_crosstalk_pair_analyzer` and `has_full_photonic_crosstalk_backend`;
  import them from `sc_neurocore_engine.photonics`. `studio.models` no longer
  offers `get_batch_simulate`; import it from `sc_neurocore_engine.studio`. A
  repository test keeps this list true on both sides.

- Push the container image that the vulnerability scan passed, instead of
  building it a second time for the push. Each release tag must name the
  scanned image in the runner's Docker daemon before it is pushed from there.

- Omit the machine hostname from the seven remaining benchmark producers that
  recorded it: the shared non-resetting LIF harness and the MAT, nonlinear
  LIF, SC resetting MAT, SFA, spike response and SpiNNaker LIF benchmarks.
  Committed results keep their recorded values. A repository test now refuses
  any benchmark module that reads the hostname.

- Query crates.io through one tool that identifies the project, in both the
  publishing crate job and the dry-run crate job, so a dispatch without
  publishing proves the registry query a tag run depends on. The tool reads the
  engine crate name and version from its manifest and stops the publication on
  any registry answer other than found or not found.

- Correct the published SHD hardware claims. The committed Vivado reports
  describe an out-of-context synthesis of `sc_shd_top` without weights or
  delays (990 LUT, 670 FF), because the weight ROM and delay tables load only
  in simulation and the AXI wrapper has no load path; they are not the cost of
  the network with its weights. Remove the claims of an end-to-end silicon run
  and of fitting in under 2% of the device. The 1 317 LUT and 848 FF figures
  came close to the out-of-context synthesis of the AXI wrapper IP (1 319 LUT,
  848 FF), also without weights, and are no longer quoted. The bitstream and
  routed reports of 13 April 2026, built from the same design without weights,
  were removed from the repository on 26 July 2026 as reproducible outputs.
  The hardware number provenance test
  now covers the roadmap, the benchmark page and the generated manifests, and
  requires the qualifier beside every quoted figure.

- Omit machine hostnames from newly recorded benchmark context while retaining
  processor, architecture, memory, runtime versions, affinity and load for
  reproducibility. Verify the shared context and its equation, Verilog and
  repeated model benchmark readers through their public entry points.

- Initialise every AXI-Lite dense-layer configuration register on reset and
  hold completion until the next accepted start, keeping final rates readable
  after delayed or repeated polls. Preserve the rate bank's one-clock completion
  pulse and continuously clocked neuron state. Correct the hardware examples to
  use unsigned encoder thresholds for inputs and weights; current endpoints,
  leak and gain retain their signed Q8.8 format. Add complete AXI lifecycle
  regressions and bounded completion/reset properties.


- Emit the catalogue AdEx formal RTL in Q16.16 so its capacitance is
  representable, preserving the reset and event assertions. Compare the shipped
  RTL with the Python model at four input currents.
- Regenerate the EscapeRate and clipped rational recovery-map clamp expressions
  from the maintained compiler and check generated catalogue DUTs for drift.
- Refresh the five-backend AdEx benchmark against the repaired formal RTL.
  Retain the dated Yosys statistics as historical measurements.

- Attest admitted host profiles in public formal evidence while retaining machine identity privately through hash-bound originals. Reject private hostnames and missing or mismatched profiles without changing measurements, resource contracts or proof gates.

- Give schema co-simulation tests and evidence references descriptive names. Advance the schema-gap report to version 2 with readable validation priorities and reject stale report vocabulary.

- Use descriptive public formal-execution labels and retain private execution custody through an explicit hash-bound conversion. Preserve native invocation identities, measurements and proof gates.

- Add typed discovery for six spike-codec families and two stream interfaces, with variant, wire-format and native-component descriptions plus explicit availability diagnostics. Preserve the five-name raster factory and existing packet compatibility.

- Reject ISI varint rasters with 16,777,216 or more time steps and delta groups outside 1–256 before encoding, preventing ambiguous headers and reference-index wrap without changing valid packet bytes. Data encoded with 4.0.0 using such varint rasters (including auto-selected varint), or delta groups larger than 256, may not decode to the original. Re-encode affected data from its original source.

- Repair AXI-Lite register updates, byte strobes and responses under backpressure;
  retain stalled read responses and verify consecutive write transactions.
- Align formal reset sampling and internal state observers with their RTL, and
  check that every assertion enable is reachable within its declared proof depth.
- Validate external native proof receipts against source, tools, execution
  accounting and independent reviews; require a separate non-vacuity audit.
- Build the pinned Yices SDK for generic x86-64, verify its instruction set and
  SAT/UNSAT behaviour, and distribute the same artifact to hosted and external runs.
- Count 91 formal job configurations and 496 declared statements, including
  the inline top-wrapper properties.
- Keep background graph job controls readable in both enabled and disabled states.

- Read DCLS synthesis statistics from native JSON, preserve FPGA fit verdicts for hierarchy debug metadata, and qualify resource counts against each supported Yosys version.
- Qualify the complete committed Q16.16 Wang–Buzsaki RTL through four-input LUT mapping with the paired native ABC executable, retaining the synthesis timeout and source event-count check.
- Install the locked compliance scanners in every native test shard and refresh the five-language evolutionary benchmark from its current sources.

- Run network-canvas simulations as durable supervised process jobs, with graph-bound
  submission, history, cancellation and result downloads that survive reconnects and
  API restarts. Preserve the existing synchronous simulation limits and numerical results.
- Save graph job recovery with workspaces and named sessions; retain the current
  canvas and undo history when opening legacy sessions without a graph. Clear
  account-owned recovery and results when the signed-in account changes.
- Refuse unavailable graph job artifacts without exposing internal exception text;
  retain the original graph and result identity through delayed replies and failed exports.

- Refresh the compatible Python, frontend, Julia and Mojo dependency locks,
  including CPU-only Torch 2.15.0, Mypy 2.4.1 and Node 26.13.0. Retain Rust
  1.98.1 and Mojo 1.0.0 while their recorded migration constraints remain.
- Use ndarray 0.17 with the NumPy 0.29 Rust bridge so native array types agree.
- Pin the CPU container's Python, Rust and base image. Accept an optional
  `BUILD_JOBS` build argument that limits both Cargo and CMake; leaving it unset
  preserves normal or inherited build parallelism.
- Update the pinned security scanners and their native command contracts,
  including Syft 1.55.0 and Grype 0.121.0 for the locked Conda audit.
- Build and cache source-pinned Yosys 0.70 and SBY 0.70 once for the CI test
  shards and synthesis workflow. Read both legacy and current native resource
  tables so the newer column order cannot silently produce zero counts.
- Expose the shared HDL identifier validator through the public library API;
  Studio project loading and native co-simulation use that same documented contract.
- Read-only model discovery API shared by the Studio catalogue, simulation,
  candidate authoring and network validation.
- Identify the release client in crates.io registry queries so the version check can proceed before publishing.
- Refuse Studio CSV export and spike-triggered averaging when full-resolution samples were explicitly omitted; show the server reason, retain JSON and display SVG, and allow recovery by shortening the simulation.
- Make the REUSE lint blocking in the Python compliance lane of the security packet and the release sweep.
- Add a manual option to the synthesis workflow that runs and times every tracked SymbiYosys job in six parallel shards; a job that exceeds its memory cap is recorded as such instead of stopping the runner.

## [4.0.0] - 2026-10-29

- The curated summary of this release is `docs/development/RELEASE_NOTES_v4.0.0.md`, which is also the GitHub Release text.
- Stop shipping the quantum cognition test suite inside the package. Its 88 cases moved from `src/sc_neurocore/quantum_cognition/tests` to `tests/test_quantum_cognition`, where CI now collects them; before, pytest never ran them and every wheel carried them.
- Keep keyboard focus inside the first-run Studio tour and let Escape dismiss it. Preserve dismissal across visits, and keep tour keys from activating workbench shortcuts.
- Pin the Studio frontend's development-only `minimatch` dependency to 11.0.1 to address GHSA-g3hv-75wg-m249; production code is unchanged.
- Add public `function_unit_functions()` discovery and `emit_function_unit_verilog()` generation for isolated combinational fixed-point functions. Unary and binary units expose independent signed ports and reuse the existing arithmetic. The new API defaults to `v2`; the existing equation compiler retains its `lut-v1` default.
- Add `from_snntorch()` and `to_snntorch()` for supported sequential inference networks on CPU float64, preserving admitted spikes, membrane states and independently owned parameters. These APIs exchange fresh-state inference definitions rather than training checkpoints; unsupported dynamics, topologies, precision and SC bitstream, encoding or Q-format contracts are refused.
- Add separate `interop-snntorch` and `interop-sinabs` optional dependency profiles. The Sinabs profile uses its compatible legacy NIR file route and does not imply identical native Sinabs and SC discrete dynamics.
- Preserve NIR graph and node metadata, primitive definitions and graph ordering through real file round trips. A versioned execution-metadata extension carries the timestep and reset convention.
- Let `from_nir(dt=None, reset_mode=None)` inherit recorded execution settings. An omitted CLI `--dt` also uses the saved clock; foreign graphs without the extension retain the default timestep of one and reset-to-potential convention.
- Add the explicit `snntorch-subtract` convention for immediate snnTorch subtraction, including the previous membrane's reset signal after an overshoot. Ordinary SC subtract reset remains a distinct convention.
- Change hardware graph lowering to refuse subtract-reset dynamics, including `snntorch-subtract`, replacing the previous implicit reset-to-`v_reset` lowering.
- Own convolution geometry as integer scalars or tuples at import and export, keeping execution and file round trips independent of caller-owned arrays. Refuse noninteger or invalid geometry before execution.
- Refuse exporting nonzero reset potentials or CubaLI primitives to NIR versions that cannot represent them.
- Write Studio graph envelope version 3 with timestep, duration, graph seed, population drives, layout and projection identifiers. Continue reading versions 1 and 2 and the legacy envelope.
- Record 58.63% median public NIR parsing/lowering overhead against the parent in a 30-sample shared-workstation comparison, with byte-identical generated output; these are local regression timings, not isolated throughput measurements. Retaining definitions copies weights beyond runtime parameter storage, so copy work grows with weight size; metadata custody and per-graph order/execution validation add work with their input sizes and were not timed separately.
- Preserve a positive signed LUT index bound for narrow fixed-point words. In particular, `sqrt(0)` and `log(1)` at signed two-bit integer precision now return zero; exhaustive small-word RTL regressions check every admitted input code against independent table oracles. C and Rust kernels retain their existing full-width clamp.
- Publish the SHA-256 of exactly the files attached to the GitHub Release; the release workflow refuses a missing or empty attachment.
- Refuse GPU waveform stream dimensions beyond the CPU packet and shader limits, and reject buffers larger than the device allocation or storage-binding limits before creating them. Add physical-GPU packet parity, lossless decoding and error-recovery checks at channel, ADC, workgroup and adaptive-count boundaries.
- State the 4.0.x security support policy, with support for 3.x ended, and update the public roadmap and group map for the 4.0.0 release.
- Install the hash-locked CPU build of torch before the development extras in every CI job that installs them, and check that build afterwards.
- Accept CVE-2026-103995 (Low) for conda-forge Python 3.15.2 in the Mojo lock while no Python 3.15 release carries the fix. The exception is valid for at most two weeks per review. Each audit checks that the installed `http/client.py` is the exact 3.15.2 release file, and any newer conda-forge release blocks the audit.
- Raise the release cargo-fuzz build limit from 30 minutes to one hour. A timed-out or interrupted fuzz command now terminates its whole process group.

- Start a workspace-local candidate from the selected catalogue model's complete canonical schema. Preview it before explicitly replacing an existing draft; retain declaration order, numeric tokens, source and profile metadata, and leave undeclared units and proposal authorship for the author to complete. Validate, diff, run, review, export and reopen the proposal through the existing candidate workflow. Canonical models and their evidence remain unchanged.

- Breaking: candidate simulation and reference-review HTTP routes now return 202 durable process-job receipts. Follow the actor-owned status route for results; recover history and artifacts after reconnecting or restarting the API, and request cancellation through the supervisor. Preserve ordered scientific inputs, numerical profiles and canonical candidate/review digests. Selected results retain their original identity when the draft changes. Direct numerical Python calls retain their existing return values. Independent scientific acceptance and promotion remain separate.

- Require the published stable Studio platform SDK 0.11.3 for federation, with its verified release hashes and matching capability manifest. Retain the other locked dependency versions and package hashes.

- Validate candidate equations against every declared unit and numeric scale before simulation or review, including conditional branches, resets and stochastic thresholds. Report located refusals without converting model magnitudes; preserve the real AdEx trace. Give diffusion noise its actual inverse-square-root time unit in candidate and strict-unit equation admission, preserving seeded sampling and integration. Candidate runs remain synchronous, catalogue-start assistance is deferred, and scientific acceptance and promotion remain separate.

- Breaking: the `lava` optional extra is removed. Its only requirement, `lava-nc`, has no release for Python 3.11 or later (the latest, 0.10.0, requires `<3.11`), so on the supported Pythons `pip install "sc-neurocore[lava]"` installed nothing. The Lava adapter remains a handoff artefact that names `lava-nc` in its generated manifest; the optional dependency matrix states why it cannot be installed alongside SC-NeuroCore.

- Breaking: Python 3.11 is now the minimum supported version (`requires-python = ">=3.11"`); Python 3.10 is no longer tested, built or published for, and the `tomli` fallback for Python 3.10 is dropped. Install on Python 3.11 to 3.14.

- The generated safety-manual template names the fields an assessor must fill as "to be supplied by the assessor" instead of printing `TODO`. The template remains a non-certifying draft.

- Keep one changelog. `CHANGELOG.md` and `docs/CHANGELOG.md` had been edited separately and diverged by about 1,350 entries and 26 release sections. They are merged into `CHANGELOG.md`: every entry and section of both files is kept once, in its release section, and where the same entry had been corrected only in the documentation copy, the corrected wording is kept. `docs/CHANGELOG.md`, which the documentation site and the release workflow read, is now a byte copy, and a test requires it.

- Restored `GLIFNeuron` to the Teeter et al. five-state GLIF5 equations across
  Python, production and safety Rust, Julia, Go, Mojo, paired schemas, receipts,
  benchmarks, Q32.32 RTL, Yosys, and bounded formal evidence. The former
  four-state RK4 recurrence remains available as the count-neutral
  `SCFourStateGLIFNeuron` compatibility identity.

- Require the published federation SDK 0.11.3.dev0 for typed local-host capabilities, and pin its verified release artifacts in the federation dependency profile.

- Lower native compound LIF, Dense, DCLS, scalar arithmetic and constant Q8.16 graph, attention and Kuramoto operations to typed HW/Comb/Seq MLIR. Verify operand formats and dimensions before hardware emission, preserve Boolean spike composition and compute parameterized DCLS tents without fixed-width clipping. Add independent integer trajectories through real RTL and CIRCT.

- Lock the Mojo environment to conda-forge Python 3.15.2, which fixes CVE-2026-114583 (`tarfile.tar_filter` link bypass) reported for 3.15.0, with the matching ICU-enabled SQLite 3.54.1. Remove the now unused CVE-2025-15367 exception and keep its acceptance rules tested on the recorded hosted scanner match. The dependency-audit policy test also checks in-repository composite actions.

- Accept partial numeric initial values and held state in Studio projects,
  simulation and native equation compilation. Preserve held variables in the
  experiment digest, recorded traces, source replay and exported requests;
  keep strict-unit equation initial values explicit.

- Preserve partial numeric initial states and held-state variables through Python integration, native IR/source replay, registered and folded RTL, fixed C/Rust kernels, adaptive wrappers and generated testbenches. Keep strict-unit initial quantities explicit and retain the legacy function-lowering default.

- Report source-located encoded argument intervals and separate numerical measurements for each approximation call, including event and exponential-Euler expressions. Preserve mathematical-domain exclusions, empty-domain results and exhaustive versus sampled scope. Add a source-bound first-call/cached analysis benchmark; generated arithmetic and the legacy lowering default remain unchanged.
- Keep the desktop HTTP cache separate for each interface content version so
  returning from rollback cannot display an older interface with a newer
  backend. Preserve browser sessions, the stable local origin and existing caches.

- Add opt-in v2 general constant and dynamic positive-base powers across Verilog, fixed C/Rust and source-bound native IR, with portable C++ math-library execution. Keep literal exponents independent of the base format and publish conditional domains, sampled measurements and conservative arithmetic bounds. Preserve the legacy default and repeated-multiply arithmetic.
- Add opt-in v2 `rsqrt(x)` across Python equations, Verilog, fixed C/Rust, portable C++ and native IR/source replay. Preserve inverse-square-root physical units; round positive encoded inputs with a half-LSB bound using bounded integer arithmetic and require proven positivity for state-dependent derivatives.
- Reject unsupported model compiler Q-formats as invalid requests before
  starting compile or co-simulation jobs. Preserve the offered Q8.8 and
  Q16.16 formats, their default and accepted zero-padded spelling.

- Browse registered hardware profiles, NIR constraints, FPGA toolchain targets
  and the board-driver interface in Studio. Keep registration distinct from
  tool availability and physical evidence, and select synthesis targets from
  the same backend catalogue without starting a tool or device operation.

- Connect hosted Studio to an explicitly configured local runtime for simulation,
  analysis and native equation jobs. Preserve original scientific JSON through
  real process workers; expose actual cancellation, digest-bound results and
  parent-mediated downloads without changing the opaque iframe sandbox.

- Declare `federation:config` through the typed SDK in local-host Studio bundles, preserving the published default and requiring SDK support before packaging.

- Require explicit equation initial quantities for strict-unit native compilation, and keep native compiler tests in their real-backend browser and integration lanes.

- Compile authored Studio equations through bounded native IR workers, preserving initial state, units, integrators, arithmetic profiles and exact random keys. Separate graph checks from source replay and native RTL emission; retain compiler settings in projects and sessions and export source-bound job evidence.

- Add opt-in v2 `hypot(x, y)` across equations, source-bound native IR, C/Rust fixed kernels, C++ expressions and RTL. Preserve compatible physical units and round the norm once with a half-LSB bound; refuse an unproven singular derivative domain.

- Refresh the equation-program benchmark with constant/dynamic powers, reciprocal root, rounding, Heaviside and Euclidean norm cases: checked expressions take between 1.0 and 2.7 times their direct Python references on the recorded shared host. These are local regression timings, not production speed claims; see `benchmarks/results/bench_equation_program.json`.
- Split the instrumented full-suite CI run into two coverage shards. On
  2026-10-27 the single coverage leg ran 5 h 27 min of batches and reached
  GitHub's six-hour job limit before its 100 % gate. Each shard now runs every
  second batch of the same global batch list. A union job keeps the required
  `test (3.12)` check name and combines both shards with the native Windows
  refusal data. It applies the unchanged 100 % gate and fails unless every shard
  published its data. `tools/run_full_cov.sh` gains `--shard=K/N` and
  `--combine-from=DIR --expect-shards=N`.

- Stop declaring a `clk` input in the Studio IR builder. The native IR reserves
  that name for the ports the SystemVerilog emitter generates, so IR verification
  and SystemVerilog emission from `/api/ir/build` output were refused.

- Move the evolutionary substrate crate to `sha2` 0.11, which replaces the
  vulnerable `block-buffer` 0.10.4 with 0.12.1 (GHSA-qwgh-2vcv-g2f7). Genome
  identifiers are unchanged because the SHA-256 output is the same.

- Show a read-only hosted view when the federated Studio panel is mounted in an
  iframe. It asks only its immediate parent for the Hub's `federation:config`
  metadata, offers a retry after refusal, invalid data or a five-second timeout,
  and makes no backend, WebSocket or browser-storage requests. A top-level host
  still mounts the standalone workspace, now loaded as a separate chunk.

- Report model-library and experiment-library load failures in separate alerts
  with their own Retry buttons. Previously loaded entries stay visible, and an
  older response can no longer overwrite a newer successful load.

- Add `--local-host` to `tools.build_studio_remote`. It selects the producer's
  relative remote entry and pins its bytes with `remote_entry_digest`, and the
  selected UI descriptor contributes to the manifest content digest. The default
  bundle keeps the published URL. Neither mode establishes host admission.

- Keep the desktop window's closing state when Chromium reports a renderer loss
  after the close began. The window stops listening for renderer loss as soon as
  it replaces the interface, instead of depending on whether the page is deleted
  before the loss notification arrives.

- Add opt-in compiler v2 `round(x)` with ties to even and `heaviside(x, at_zero)` with a dynamic zero value across Python, generated C/Rust/C++, RTL and native IR/source replay. Fixed-point results saturate to the signed word; strict-unit rounding uses base units. The default `lut-v1` lowering is unchanged.

- Describe the Studio simulation input receipt using its public contract name.

- Reject missing, unknown or dimensionally incompatible display units when replaying
  persisted equation graphs, including source metadata with valid native checksums.

- Pin the complete Linux desktop and backend dependency closure with download
  hashes. Install the root package without resolving different runtime versions
  and document application-specific Linux sandbox policy requirements.

- Add public equation-to-native-IR compilation and source replay with typed scalar
  state, runtime range checks and product registers. Preserve authored integration,
  event, random-stream and timestep behavior through direct SystemVerilog and CIRCT.
  Reject unsupported scalar ports and unproved variable-division domains explicitly.
  Bind complete equation source to persisted graphs and refresh the measured engine
  interface inventory and compiler benchmark evidence.

- Make generated Studio federation type archives independent of build time and
  host timezone while preserving declarations and archive metadata. Add real
  command acceptance with complete coverage and retain strict TypeScript checks.

- Export complete sealed Linux desktop bundles as reproducible gzip archives, preserving
  payload permissions, links and build identities. Refuse changed inputs and existing
  output files, and exercise installation from the extracted runtime.

- Add verified per-user Linux desktop installation, versioned updates, compatible
  rollback and removal that retains research profiles. Refuse changed or unreadable
  payloads and foreign launch files. Add complete-runtime installation acceptance
  and user instructions for managing supplied desktop preview bundles.

- Add the optional `snn-studio` native Linux window with persistent browser storage,
  owned backend shutdown, native export dialogs and restricted navigation. Add
  hashlocked Qt dependencies, desktop integration checks and source-launch guidance.

- Add owned Linux desktop backend admission, retained user profiles and origins,
  and verified cancellation, worker shutdown and process-group cleanup.

- Restore the engine-test inventory for the five published native IR suites and bind the historical NIR graph benchmark gate to its measured source snapshot, including all eight per-file hashes. Upgrade the workstation notebook converter to 7.17.2 for GHSA-pjvx-prjx-h659 and verify the exact development documentation stylesheet policy. Refresh the pinned Rust cache and PyPI publishing Actions, and update crossbeam-channel to 0.5.17.

- Keep unreadable Studio worker processes in the survivor set instead of reporting
  them as exited when process metadata cannot be read.

- Lower native constant division by one to unsigned bit-preserving wiring in
  direct SystemVerilog and CIRCT, avoiding a wide-division simulator discrepancy
  on Windows while retaining the original numeric expectations. Zero-extend
  short signed words and preserve single-tick decoder timing.

- Add typed native-IR scalar registers with atomic next-state stores, Boolean
  enables and asynchronous reset. Preserve state bindings in text and expose
  matching Rust/Python builders, direct SystemVerilog and CIRCT lowering;
  reject invalid drivers, types and use of terminators as values.
- Preserve raw integer reset words during graph persistence, including Q8.8
  Rate values, with explicit integer constructor tags where text is ambiguous.
  Retain the historical meaning of untagged Rate literals.

- Read each facet receipt once per Studio catalogue, model-detail or seal operation,
  retaining per-model verification and fresh inputs for later detail requests.
- Refresh four co-simulation receipts through executed Python-to-generated-RTL
  validators against the current compiler and synchronise installed readiness.

- Discover Studio interface resources without importing the scientific API,
  preserving packaged-asset precedence and existing frontend imports.

- Refresh hash-locked certifi, Pillow and virtualenv dependencies for current
  advisories, and update Semgrep's OpenTelemetry API, SDK, exporter and
  instrumentation together through the existing overrides. Pillow 13 remains
  in the Python 3.12 optics and workstation profiles.

- Align Go module minima and pinned CI toolchains on 1.27.3 for
  [GO-2026-6809](https://pkg.go.dev/vuln/GO-2026-6809).

- Group PyO3 and NumPy Cargo updates at the workspace root so their ABI
  companions and Cargo.lock advance together. Update the pinned setup-julia
  action to 3.0.3 while retaining the supported Julia runtime contracts.

- Add typed native-IR scalar arithmetic with explicit resize/bitcast, Boolean comparison/selection, five rounding modes and inclusive clamps. Preserve full signed/unsigned word division in direct SystemVerilog and CIRCT, and expose the same operations through the Python graph builder. Refuse duplicate, invalid and compiler-owned port names, including clock and Dense conversion-bus collisions; verify deep graphs without recursive cycle traversal. Validate unused declarations and refuse invalid widths, excessive vector nesting and overflowing parser IDs before emission.

- Parse generated API documentation structurally and bind inline scripts to
  exact CSP hashes and route-specific image resources. Update Studio's locked
  Vite, Vitest and DOMPurify versions.

- Add a task-oriented SNN Studio user manual: installation, first experiment,
  analysis, networks/training, workspace recovery, reproducibility, FPGA target
  profiles, troubleshooting and terminology. Correct equation-editor directives,
  compiler/pipeline guidance and current-control replay-export boundaries.

- Preserve temporal native-IR bitstream windows and actual packed widths in population counts. Add explicit Q8.8 `decode` and Python `ScGraph.emit_mlir()`; keep integer `div_const` semantics, reject incompatible shapes/windows and zero divisors, and compile the public synapse example to real RTL. Check both direct RTL and native CIRCT against numerical references, including full encoder periods and wide-word division.

- Add per-user desktop profile paths and bounded, versioned configuration. Preserve legacy projects, refuse unusable profile roots and retain the selected origin and project directory through atomic private writes.

- Spike-event layer for Bluetooth-class links (`sc_neurocore.spike_codec.spike_events`: `SpikeEventEncoder`, `SpikeEventDecoder`, `SpikeEventWindow`; Rust `SpikeEventEncoder`/`SpikeEventDecoder`). Threshold-crossing events in 1 ms bins and spike-band power from raw int16 samples: Q14 second-order Butterworth high-pass (250 Hz) and low-pass (1000 Hz), streaming median noise estimate, -4.5 sigma downward crossings with refractory period and warm-up, adaptive binary coding with 16-bit event probabilities; Rust and the pure-Python reference write identical packets. On the 743 Neuralink challenge recordings the detector finds injected spikes as often as an offline float reference (73.7 % vs 73.8 % at 5 noise SD, 99.9 % at 8), and 1024 channels need 51 kbit/s for events and 157 kbit/s with band power in 10 ms packets. Fixes from the independent review of the streaming lossless coder: `sc_neurocore.spike_codec` imports in the base distribution (the predictive codec imports the optional world model only for its world-model predictor), decoders refuse sample requests a packet cannot hold before allocating (a native decode of 2**63 samples aborted the process), and `detect_adc_bits` refuses fractional or out-of-range samples instead of casting them.

- Streaming lossless coder for live multi-channel electrode streams (`sc_neurocore.spike_codec.waveform_stream`: `LosslessStreamEncoder`, `LosslessStreamDecoder`, `detect_adc_bits`; Rust `WaveformLosslessEncoder`/`WaveformLosslessDecoder`, and `GpuStreamEncoder` with the engine's `gpu` feature). ADC levels with dense numbering (missing codes leave it), a sign-sign LMS predictor of order 16 and an adaptive binary range coder; state persists across windows (each packet adds about 9 bytes per channel group, so 1 ms packets cost little only when many channels share a group). Rust, the pure-Python reference and the GPU encoder write identical packets. `WaveformCodec(mode="lossless")` now uses it (mode byte 4, ADC width detected or `adc_bits`); the earlier Rice coder stays as `mode="lossless_rice"` (mode byte 3). Lossless modes in a legacy `WFCX` stream are refused. On the public Neuralink compression-challenge data (743 files) `lossless` measures 3.457x with every file byte-identical after decoding (`lossless_rice` 3.33x); FLAC, WavPack, Monkey's Audio, brainwire, hxrdxk and Sac measured on the same files are in the BCI codec guide.

- `WaveformCodec` gains `decompress` for every mode and an exact `lossless` mode (per-channel ADC-level mapping, fixed polynomial prediction, block-adaptive Rice codes). Stream format `WFC2` adds per-snippet template scales, the background's first row, closed-loop background quantisation and a CRC-32 trailer; corrupted streams and headers declaring more than `max_samples` raise `ValueError`. Legacy `WFCX` streams decode their spike raster only. Noise thresholds, detection, template matching, background extraction and quantisation and the lossless coder run in Rust in `sc_neurocore_engine` (`backend="auto"|"rust"|"python"`), writing byte-identical streams to the NumPy reference. On the public Neuralink compression-challenge data (743 files, 62.6 minutes) the lossless mode measures 3.33x with every file byte-identical after decoding, encoding in 10.1 s with the Rust kernels; the BCI guide's stale 24x synthetic figure is replaced by measured values.

- Match optional Studio simulator getter types to the native bridge callable contract, including installations without the optional engine.

- Use the existing typed Torch module base for exact-reset cells so strict checking also works without the optional Torch dependency; runtime inheritance and numerical updates are unchanged.

- Use `SC_NEUROCORE_STUDIO_PROJECT_ROOT` consistently for Studio project storage and backup planning. Preserve the legacy directory by default and refuse unusable configuration before startup writes.

### Changed

- Native IR `dense_forward` now uses per-neuron, row-major weights and checks declared probability and signed Q8 formats before emitting RTL. Python's default `y_max` is 256, matching the Rust default and the previous emitted hardware; explicit endpoints must fit the signed state word. Regenerate graphs that relied on the previously ignored value 65535.

### Added

- Add stable beta-one `softplus` to Python equations, symbolic derivatives, v2 Verilog, fixed C/Rust and portable C++ expressions, with full-word numerical domains and an explicit arithmetic error bound. Refuse extra smooth-unary derivative arguments instead of discarding them. Refresh complete compiler/expression benchmarks with measured hardware context.

- Add opt-in v2 `atan2(y, x)` to Python equations, Verilog, fixed C/Rust and portable C++ expressions, with two-operand accuracy reports and explicit derivative branch-cut refusals. Preserve numeric name binding through compiler-inserted noise and integrator calls. Refresh compiler/expression benchmarks with measured CPU and RAM context; emit Rust units without redundant parentheses.

- Add indexed operands to function-unit integer programs, with matching Python, Verilog, C and Rust bindings and explicit refusal of missing or extra operands.

### Fixed

- Native IR identifier allocation now refuses exhaustion before changing the graph.
  `ScGraph::try_fresh_id()` exposes a recoverable error; the existing allocator
  keeps its signature and fails deterministically instead of wrapping in release
  builds when extending a sparse graph loaded from text.

- Refuse non-regular desktop configuration before reading it, including named pipes without a writer. Keep existing regular-file symlinks readable and preserve refused paths.

- Preserve constant word widths, binary points and vector shapes in native IR emission; reject non-finite, overflowing and malformed constants instead of silently coercing them.
- Restore declared encoder seeds on every Dense reset, including a single reset clock or a pulse between clocks. Reject nonpositive dot-product fan-in with an explicit configuration diagnostic and align the full-pipeline Python reference with signed truncation and registered encoder latency.

- Fix full-range signed current interpolation in the HDL dot-product mapper, including descending endpoint ranges; retain every product bit until signed division and compare real RTL against an exact integer reference.

- Preserve every LIF, dense and delay-coded layer parameter in native IR text. Parse delay-coded layers, retain ordered tap offsets, and reject malformed explicit attributes instead of silently substituting defaults.

- Open the local Studio browser only after its server is listening; refuse an occupied port without opening a page.

- Add stable beta-one `softplus` across Python equations, symbolic derivatives, v2 Verilog, fixed C/Rust and portable C++ expressions. Document the full signed-word domain and separate arithmetic error bound; exercise actual native outputs and refresh both full benchmark protocols with hardware context. Fix smooth unary differentiation silently discarding extra positional arguments.

- Add opt-in v2 `atan2(y, x)` through the equation runtime, Verilog, fixed C/Rust and C++ expression paths. Preserve both operands and the fixed-point zero convention, reject invalid domains and unproven derivative branch cuts, and report two-dimensional accuracy. Refresh source-bound compiler/expression benchmarks with CPU and RAM context. Remove redundant Rust expression parentheses so generated units compile with warnings denied.
- Retain and check every timed model trajectory across all five runtimes; reject non-finite states and inconsistent dimensions before writing benchmark evidence. Record CPU, RAM and runtime context with each report.

- The integer-power bound of equation programs also covers a computed
  exponent. A re-reading showed that `2 ** (60 * 60 * 60)` passed, because
  the bound was applied only where the base was itself a power.
- Declare the Studio remote panel in its federation manifest and content digest.
  Add a local remote-bundle command that verifies frozen frontend bytes and
  writes a reproducible archive with a digest-pinned deploy descriptor.

- Two bounds of the expression gate moved from the shape of the text to the
  values, after an outside review showed that a slice, a choice or a call
  around a literal carried it past the rule. `*` and `+` refuse a tuple or
  list that the expression built, and an integer power of a power longer than
  4,096 bits is refused before it is computed. The rule that a power of a
  power needs literal exponents is withdrawn; `(a**b)**2` is valid again.
- The parity rerun workflow installs a hash-pinned profile
  (`requirements/brunel-parity.txt`) and runs the package from the checkout.
- Preserve diffusion-noise scaling for every positive equation timestep,
  including values below `1e-12` and subnormal floats. Keep zero-step Euler
  states frozen and reject timesteps outside the selected hardware format.

- `docs/guides/supported_workflows.md` gives four workflows of the base
  package with the output each prints, and says what each way of installing
  provides and which subpackages only a source checkout has. The tests run
  every block from the source tree and from a built wheel installed outside
  the repository. The README's quick start now stands before the roadmap and
  points to that page.
- `NetworkRunner.add_projection` admits its CSR arrays before allocating:
  population indices, `n_source + 1` row offsets starting at 0 and never
  decreasing, equal column and value lengths, a last offset within them, and
  every used column inside the target population. Malformed connectivity now
  raises `ValueError` and leaves the runner unchanged; before, it silently
  dropped driven connections or aborted the process on short storage.

- The Brunel comparison can be rerun and judged. Each repetition of
  `benchmarks/scaling_benchmark.py` now has its own seed, the Brian2
  reference is seeded, and the spread of rates across seeds is recorded.
  `tools/brunel_parity_check.py` judges a recording by the published 5 %,
  and the weekly `brunel-parity.yml` workflow reruns both lanes on current
  code with eight seeds and one second simulated. A rerun of 2026-10-12 is
  committed: rate ratios 1.015 and 0.996 at 1,000 and 5,000 neurons.
- `EquationNeuron(..., threshold_rng="philox")` is an optional generator for
  Poisson and escape-rate thresholds: a step fires when `rand()` of the reserved
  call site `2**31` is below the probability, keyed by the neuron's seed and
  stream (its own stream per neuron, `2**-32` resolution). The compiler lowers
  it as the level threshold `rand(S) < p`, so the C and Rust kernels compile it
  as well as the RTL; for a probability with an exact Q-format code the model,
  RTL, C and Rust fire on the same steps. The 16-bit LFSR stays the default and
  its RTL is unchanged.

- The expression gate bounds three more things. An array function of the
  equation namespace cannot be handed an output array: a positional argument
  after its inputs must be the literal `None`. A tuple or list
  literal cannot be repeated or concatenated. A power of a power needs literal
  exponents whose product is at most 64; `(a**2)**3` stays valid.
- The cost of equation programs has a committed benchmark,
  `benchmarks/bench_equation_program.py`. Its receipt
  `benchmarks/results/bench_equation_program.json` (loaded development host)
  puts a program at between 0.9 and 3.2 times the same arithmetic written as
  a Python function.
- The diffusion-noise symbol `xi` is the counter-based normal draw of a reserved
  call site (`XI_SITE = 2**32 - 1`, sub-step `k` reads `XI_SITE - k`) divided by
  `sqrt(dt)`, keyed by the neuron's seed and stream like `randn()`: replayable,
  shared by every occurrence in a step, and compiled bit for bit to RTL, C and
  Rust (`compiler.diffusion_noise`). It was drawn from NumPy's global generator
  and, despite the documented `1/sqrt(dt)`, equalled `N(0, 1)`, so an Euler
  increment had variance `dt**2` instead of `dt`; seeded runs with `xi` change.
  A caller's `noise_rng` still draws `xi` (Python only, refused by the compiler);
  explicit call sites must lie below `2**31`. Model profiles count `xi` as
  `philox-draws` (replayable), Studio keys it by the run seed, and the precision
  comparison runs it and states the draws' key and stream.

- Published claims are bound to their sources after an outside review. The
  simulator-comparison gate enforces the 5 % the README states (it allowed
  10 %), the README says what the recording is and that no test reruns it,
  both workflow counts and the workflow table follow the workflow directory,
  and `VALIDATION.md` states what the coverage gate measures and omits from a
  block generated from the coverage configuration (`tools/coverage_scope.py`,
  checked on every commit). The README row on five-runtime completion no
  longer carries the heading "High-fidelity".
- Four co-simulation receipts are recorded again. The equation-program change
  altered two of their subject files, which withdrew the verified RTL label of
  four models.
- The audit module and one test import the TOML reader in the form Python 3.10
  has, and the distribution build's refusal of a missing comparison script
  has its test.
- Docstrings and strict typing: the StochasticIF and Cazelles model modules
  and 12 native-dispatch, projection and StochasticIF test modules meet the
  NumPy docstring convention and `mypy --strict`.

- Equation expressions are no longer executed by the interpreter. The safety
  gate returns the checked tree and a new translator turns it into a program
  of plain function calls; `EquationNeuron`, the strict-units check and the
  Studio nullcline service run that program. The three `eval` sites and their
  suppressions are gone (code scanning alerts 483 and 487).
- The expression grammar is closed. Only a function under a plain name can be
  called, and only the attributes `real` and `imag` can be read. Method calls
  and other attributes were admitted before whenever their name was not on a
  block list; numeric scalars and unit quantities carry methods that write
  files. An expression that used such a construct is now refused when the
  neuron is built.
- `evaluate_scalar` and `evaluate_condition` take a program and the bindings;
  the evaluation-globals argument and `EVAL_GLOBALS` are removed.
- Numerical results are unchanged. A program costs more than bytecode. One
  measurement on a loaded development host, not yet reproduced by a committed
  benchmark, gave 0.9 to 2.0 times per expression and 1 % to 22 % per neuron
  step.
- Checking whether a network can run natively no longer moves the caller's
  random generators: the state probe constructs and steps default neurons
  inside a generator transaction, and a model that draws from NumPy's global
  generator or `random` (`StochasticIFNeuron`) is not dispatched natively, since
  a native run would not advance it. Before, `backend="auto"` and a refused
  `backend="rust"` advanced `np.random` by 44 draws.

- Vendor-record exceptions are bound more tightly after an outside review.
  The vendor identifiers of a channel are a constant of the audit code keyed
  by the host and channel read from the lock, so a table can no longer choose
  whose records are read, and a package from a channel without an entry is
  never excepted. Each reviewed record carries the products it named at
  review; a changed product list fails the check. A busy source is asked
  three times and then blocks with `source unavailable`.
- Docstrings: `training/snn_modules.py`, the Brunel-Wang model module and
  `schema_contracts` meet the NumPy convention, and 66 compiler, DSL, readiness
  and training test modules carry real docstrings and strict typing.

- The Julia SC Compte network lane depends on the `FFTW_jll` transform binary
  alone and calls it through the C interface. The 1,000-step state digests of
  the lane are unchanged. Its lock falls from 42 to 10 packages and no longer
  names the package manager, the downloader or their network libraries, which
  carried all 77 Julia advisory findings. The `SHA` standard library the
  module already used is now declared. The three receipts bound to the changed
  sources are regenerated.
- `compile_network_to_fpga()` takes `random_precision` ("standard" or "exact")
  and lowers `randn()` in every neuron module and folded processing element
  with it; with "exact" every state word of the direct and the folded network
  equals the exact 60-bit reference.

- A package the Pixi scanner cannot audit is accepted only through a
  `vendor_records_reviewed` exception. The audit requires the vendor channel
  to list the locked file with the lock's digest, and the vulnerability
  records bound to the vendor's product identifiers to be exactly the reviewed
  ones, none naming a package of the lock. The four Modular packages of the
  Mojo lock are recorded this way.
- `Network.to_torch()` steps in float64, the Python models' precision, for any
  input dtype (a float32 bridge overflowed or rounded where the model did not),
  keeps the models' runtime refusals of a non-finite voltage and Lapicque
  candidate, and carries a StochasticLIF neuron's live refractory counter.

- Advisory exceptions are bound to the lock: channel and platform come from
  the locked artefact's address, a newer build of the locked version ends an
  exception, and a table for another ecosystem or an unread lock fails.
- Compiler fidelity: a comparison used as a comparison operand
  (`(v > 0.25) > 0.5`) is 1.0 or 0.0 on RTL, C and Rust as in the model (the
  RTL and C never fired, the Rust kernel did not compile); a stochastic
  threshold at Q1.f compares with 1.0 one bit wider (Q1.15 fired every step);
  a dt outside the format is refused by every emitter wherever it multiplies
  (dt = 1.0 at Q1.15 was integrated as -1.0), a map excepted; the adaptive
  precision compile forwards `function_lowering` and `random_precision` to both
  datapaths and its manifest (the CLI wrote the standard randn unit for
  `--random-precision exact`) and refuses `--pipeline`; the numeric contract's
  divisor check rounds bankers' half to even as the RTL does and folds huge
  finite values exactly, and `precision_report` and the `precision` CLI report a
  huge dt or parameter instead of raising `OverflowError`.

- A test binds the version in `.zenodo.json` and `CITATION.cff` to the package
  and engine versions. Releases 3.13.4 to 3.16.0 were archived under the stale
  label 3.13.3 because nothing checked it.
- The folded network interconnect reads graded spike values and analogue
  voltages from its packed buses as signed words: a negative value or voltage
  entered the product and the source-threshold comparison as a large positive
  number (-0.5 times +1 gave +32767). Populations of one neuron type must now
  share one template definition (equations, threshold, reset, method, initial
  values, detection, spike value, seed); a second definition under the same
  type name ran the first one's module and is now refused before any RTL.

- The Pixi lock audit accepts a finding only through a recorded exception whose
  reason it re-verifies on every run: no newer version in the channel, or the
  affected program absent from the locked artefact. Unrecorded findings, failed
  checks, stale records and unaudited packages still fail.
- `randn()` can be lowered at full precision: `random_precision="exact"`
  (on `compile_to_verilog`, `compile_to_datapath`, the bit-true C and Rust
  kernels, `hardware_numeric_contract` and `--random-precision exact` in the
  CLI) computes the Box–Muller transform with 60-bit internals and rounds
  once, so every code is within 0.5 LSB + `2^(fraction-45)` of the exact
  value (0.50002 LSB at Q3.29, where the standard unit can be 32.5 LSB away).
  The standard unit stays the default and its RTL is unchanged.

- Wheel consumer jobs build the checkout into a wheel and install that file,
  so every `pip install` of the job consumes a hash-locked profile or a local
  artefact.
- The FPGA network compiler learns: `ConnectionSpec(plasticity=STDPRule(...))`
  applies trace-based pair STDP (the rule of `Projection.update_plasticity`) in
  the network's fixed-point format, in the direct and the folded
  interconnect. `nir_bridge.stdp.stdp_tick` is the integer golden; every
  weight of both interconnects equals it each tick. Delayed, thresholded,
  external, analogue or graded plastic sources are refused, and a plastic
  network is never placed on the address-event interconnect.

- Report full-suite coverage for the repository source tree only. Wheel copies
  that distribution tests run from temporary directories no longer abort the
  final report; the 100 % threshold and the measured sources are unchanged.
- The FPGA network compiler carries graded spikes: a population whose
  template has `spike_value` adds `(value · weight) >>> fraction` to its
  targets (a value of 1.0 is the binary case; a source threshold compares the
  value). The direct interconnect wires each instance's `spike_value_out`, the
  folded one commits the values to a `val_bus` with `spike_bus`, both with
  delays; a network with a graded source is not auto-selected for the
  address-event interconnect, which carries addresses only.

- Update the pinned Pixi advisory scanner to 0.2.0 and validate its renamed
  report fields. Continue refusing every advisory match, ignored rule,
  unaudited package and incomplete or inconsistent response.
- The table-accuracy measurement of wide formats places each table cell where
  the RTL indexes it; at fraction 0, where a 1/8 step is finer than one code,
  it takes every code inside the table. Sampled cos at 32/0 and 62/0 reported
  0.042 although the table's integer codes reach 0.46 (the exhaustive 16/0
  value); the sampled result now equals it and is still stated as sampled.

- Refresh the complete compatible Mojo support environment to Python 3.15.0
  and binutils 2.46.1 while retaining Mojo 1.0.0. Include the matching shared
  `libpython` required by Mojo's Python interoperability, and retain SQLite's
  ICU-enabled build for Unicode casing and collation.
- Network monitors and repeated runs report the run that happened. A
  `RateMonitor` bin divides by the time its whole steps cover (a 10 ms bin at
  dt 3 ms is 9 ms: 333 Hz, not 300 Hz); `StateMonitor` and
  `Population.get_states` read each model's `get_state()`, so an
  `EquationNeuron` is no longer recorded as zeros; a second `Network.run`
  continues the step index and delivers the last step's spikes (another dt is
  refused); `SpikeMonitor.cross_correlation` keeps every requested lag;
  `StateMonitor(record=[])` records no neuron instead of crashing; and
  neurons built from one explicit initial state no longer share it.
- Spike gating admits a model's purity before its probe copies or resets
  anything, restores both process generators around the whole probe, and
  treats a self-referential state as not comparable instead of raising
  `RecursionError`.
- `scale_free` starts from a star on `m + 1` nodes and attaches by actual
  degree (seed nodes counted a degree they had no edge for); `ring_topology`
  refuses `k > (n - 1) // 2`, which doubled the antipode or added self
  connections.

- Refuse incomplete or malformed fixed-point LIF co-simulation traces instead
  of reporting a false bit-exact match. Require a nonempty reference, exactly
  one two-field decimal RTL row per step, positive step counts and nonnegative
  seeds; report unreadable traces without a traceback. Release testbench reset
  on a falling edge so the first stimulus is consumed exactly once.
- `Network` dispatches to the native runner only what the runner
  reproduces. It returns voltages only, so a model with state besides `v`
  (AdEx's `w` stayed 0 after 1000 native steps; Rulkov's `x`, `y`) is
  refused; `LapicqueNeuron` names the 1907 source profile in the engine and
  is refused (`SCLapicqueLIFNeuron` is the same on both sides); embedded
  stimuli, per-synapse delays, `weight_threshold` pruning and plasticity,
  which the runner dropped, are refused. `auto` falls back to Python and a
  forced `rust` raises naming each reason. In Python, `weight_threshold` now
  also prunes projections with per-synapse delays.

- Update the Mojo environment's locked Expat to 2.8.5, SQLite to 3.54.0,
  libuuid to 2.42.4 and Tornado to 6.5.10. Retain SQLite's ICU support,
  Python 3.14.4 and Mojo 1.0.0.
- The compiler encodes parameters, literals and initial states exactly
  (`round(Fraction(value)·2^fraction)`): a finite value as large as 1e308
  overflowed the float product and raised `OverflowError` in the RTL, the
  kernel and the hardware numeric contract; it is now out of range like any
  other value outside the format. `export_verilog` refuses it, and a derived
  parameter that is not finite, by name with `SCHardwareError`, and checks
  every neuron's Lapicque profile.

- Update the Mojo environment's locked OpenSSL dependency from 3.6.2 to 3.6.5,
  retaining the Mojo compiler and every other locked package.
- `Network.to_torch()` steps each model with its own operations in its own
  order (`(R·I)·dt`, the SC flow through `v_inf`), so a float64 run equals the
  Python voltages bit for bit and a resistance of 1e308 no longer turns a
  zero-input state into NaN through a precomputed `R·dt`. Every neuron's
  Lapicque profile is checked, not only the first one's.

- Report Julia as selected bundled sources requiring JuliaCall, including the
  adaptive-threshold kernel already shipped in the wheel. Verify installed
  kernel trajectories and final states against Python while retaining explicit
  refusal for Julia models whose sources are absent. Go and Mojo still need
  built shared libraries.
- With `substeps > 1`, `<state>_prev` in a threshold or graded value is the
  macro step's entry state in the RTL and the C and Rust kernels, as in
  `EquationNeuron.step`; the RTL compared with the penultimate sub-step
  (`x < x_prev` with `x -> 1 - x` spiked once where the model never does).
  The folded PE refuses `substeps > 1`, which it did not lower.

- A population derives its neurons' seeds by distinct generator states: the
  LFSR16 models (`PoissonNeuron`, `EscapeRateNeuron`) declare that seed 0 runs
  as `0xACE1` (`SEED_ALIASES`), so the two are never both handed out and the
  capacity is 65535 neurons. `Population(PoissonNeuron, 65536, seed=11)` gave
  two neurons one stream; seeds before the first alias are unchanged.

- The Studio equation playground sees `rand()`, `randn()` and `poisson()` in
  reset assignments (including `;`-separated ones): a seed is accepted and
  replayed, an unseeded run discloses its counter draws, and a fresh trial is
  not cached. A reset-only draw was refused a seed and labelled deterministic.

- `export_verilog` checks every lowered parameter and the initial voltage at
  its format before writing: a value outside the range (threshold 300 was
  written as 44 in Q8.8, a gain of 256 as 0) or a non-zero value that rounds
  to zero is refused with `SCHardwareError` naming the quantity. The new
  keyword arguments `data_width` and `fraction` choose a wider format (for
  example Q16.16), in which those models step as the Python neurons do.

- `Network.to_torch()` keeps projection weights in float64, the precision of
  the CSR data, and casts them to the input's dtype in the forward pass: a
  float64 run uses the network's own weights and a sync without training
  writes every weight back unchanged (a weight of 0.99999999 had become 1.0).
  A step whose voltage overflows to infinity now resets to `v_reset` instead
  of turning the state into NaN and losing every later spike.

- Spike gating probes only models whose purity is declared (the catalogue
  models, or a user class with `quiescence_pure = True`); any other model is
  never skipped. The probe also restores the standard `random` generator and
  refuses a model that moved it, and reads private `__slots__` under their
  mangled names. A user neuron drawing from `random` was skipped after one
  sampled zero spike and a private slot crashed gated stepping.

- `network.export_verilog` writes RTL that elaborates: each population is
  lowered by the equation compiler to a neuron module with the model's own
  update, `>=` threshold and hard reset (SC LIF Lapicque exact flow,
  StochasticLIF forward Euler), one instance per neuron with a Q8.8 current
  word. It instantiated an `sc_lif_array` no file defined and collapsed every
  LIF-family model and parameter set into one file. Other models, the 1907
  Lapicque profile, noise, refractoriness, per-neuron parameters, an empty
  network or population are refused before anything is written.

- `Network.to_torch()` steps as the supported models step: the SC LIF Lapicque
  profile by its exact flow and StochasticLIF by its forward-Euler update, both
  with the models' `>=` threshold and hard reset (surrogate gradient in the
  backward pass), each neuron from its own voltage. The 1907 Lapicque source
  profile, parallel CSR connections, a nonzero `weight_threshold` and
  heterogeneous per-neuron parameters are refused instead of being substituted,
  overwritten or ignored. Surrogate spikes keep the input's floating dtype, so a
  float64 bridge runs past its first projected step.

- `small_world` is a Watts-Strogatz simple graph: a rewired edge goes to a node
  that is neither the source nor already its neighbour, the graph keeps `n·k`
  entries in both directions, and an odd or too large degree, a probability
  outside `[0, 1]` or `n < 1` is refused (one invalid call used to loop). It
  could emit a parallel pair, which delivered twice the weight per spike.
  `grid_topology` is documented as the Chebyshev (square) neighbourhood it is.

- The network compiler accepts equation-defined populations: a `NeuronSpec`
  with a `template` (equations, threshold, reset, default parameters, and
  optionally method, detection, `spike_value`, `rng_seed` and a base
  `random_stream`) compiles in the direct and the folded interconnect. A
  population that draws gives each neuron the stream `base + global index`:
  per-instance `P_RNG_STREAM` overrides in the direct top, and in the folded
  top one network step counter with the active neuron's stream on the shared
  PE. Stochastic detection and rising-edge templates stay on the direct path.

- A state whose name takes a compiled form's own field is refused by name
  instead of emitting a duplicate declaration: in the C and Rust kernels a
  state named `spike` (field `spike_out`) or, for an edge-crossing neuron,
  `thr_prev`; in the registered RTL a state named `spike` (port `spike_out`).

- Add `poisson_grid(lam)`, a Poisson draw whose mean is a variable expression:
  the mean is floored to a multiple of 1/16 and clamped to `[0, 16]`, and word
  0 of the Philox block is inverted against the exact thresholds of that grid
  mean (257 rows of 47 thresholds, as for a constant `poisson()`). The model,
  the RTL and the C and Rust kernels give the same counts; `poisson()` keeps
  its exact constant mean and its refusal now names `poisson_grid()`.

- The bit-true C and Rust kernels mirror every integrator the RTL lowers:
  `rk4`, `exp_euler` and `gauss_seidel` join `euler` and `map`, and the macro
  steps of sub-stepped crossing models are mirrored (draws once per macro
  step, graded value on the boundary). Every executable bundled schema using
  these methods gives identical RTL and C traces at Q8.8 and Q16.16 (150 steps
  each). The RTL's integrator scalings (`dt`, the RK4 stage factors and
  `dt/6`) now round under the format's rounding mode like every other product:
  with `rounding="nearest"` the Euler kernel rounded the `dt` product while the
  RTL truncated it, so the two disagreed from the first step. With the default
  truncation the RTL of all 80 bundled schemas at six formats is byte-identical.

- Counter-based draws and graded spikes beyond the compiler:
  - `sc-neurocore compile` takes `--data-width`, `--fraction` (both were
    documented but missing), `--method`, `--function-lowering`, `--rng-seed`,
    `--random-stream` and `--spike-value`; `from_equations` takes `rng_seed`,
    `random_stream` and `spike_value`.
  - Schemas: `threshold.spike_value` makes a schema neuron graded and
    `UniversalNeuron(random_stream=...)` separates instances; the model profile
    reports `philox-draws` (with its seed) and admits a seed override for a
    schema that draws; the schema validator uses the model's real function
    vocabulary, so `rand`, `sigmoid`, `exprel`, `clip` and conditionals no
    longer raise unknown-name warnings.
  - The Studio equation playground accepts a seed for equations that call
    `rand()`, `randn()` or `poisson()`, keys their draws with it and reports
    the Philox generator (`counter-draws`, or `counter-draws+diffusion-noise`
    with `xi`).
- Close the remaining review P3 items: the v2 accuracy report is linked from
  `compile_to_verilog`; `BitstreamSpikeRecorder` (Python, Rust, Julia) refuses a
  NaN or infinite `dt_ms`; eight C-emitter test modules are type-complete under
  strict mypy and documented.

- Fix the equation compiler findings of the 2026-10-19 reviews:
  - `randn()` near `u = 1` (word 0 close to `2^32`) was up to 1.84 LSB from the
    model at Q16.16; the radius there is now summed as a series, and the
    accuracy is stated as half an LSB plus a measured transform error of
    `2^-24` (`random_units.NORMAL_TRANSFORM_ERROR`), which dominates at high
    fractions. The earlier unrestricted 0.5 LSB statement was wrong.
  - The C and Rust kernels evaluate reset and graded-value expressions only on
    a spike, so an expression undefined off the spike no longer traps them.
  - `xi` in any expression, the graded value included, is refused by the
    registered module, the folded PE and both kernels instead of emitting an
    unbound name; `uses_diffusion_noise` sees the graded value.
  - The folded PE of a crossing neuron without reset takes its threshold
    history on `thr_prev_in` and returns it on `thr_cond_out`.
  - Compiling a neuron that has already stepped no longer changes the reset
    state of its edge history: the initial threshold uses the draws of step 0.
  - Draws are detected from the parsed calls, so a state named `rng_step`
    without draws compiles in the kernels; with draws the clash is refused. A
    state named `spike_value` is refused with a graded spike. The
    `{var: derivative}` kernel refuses draws.
  - `expit` runs in the Python model; the hardware contract describes the
    function lowering in use, the actual table index widths and the draws'
    seed and stream; the table measurement no longer overflows int64; a failed
    stochastic step restores `last_spike_value`.

- Random draws compile in pipelined and multi-substep registered modules:
  `rng_step` advances on the pipeline's valid cycle and on the macro
  boundary, so each logical step draws what the unpipelined module draws
  and every sub-step of a macro step draws the same numbers, as the Python
  model does. A graded spike (`spike_value`) is set on the macro boundary of
  a multi-substep module from the state after its last sub-step.

- Add graded spikes: `EquationNeuron(spike_value=...)` evaluates an expression
  on a spike (on the state before reset) and keeps it as `last_spike_value`.
  The registered RTL emits `spike_value_out`, the folded PE drives it, and the
  bit-true C and Rust kernels store it; all agree with each other and with the
  model on a dyadic grid.

- The folded datapath PE (`compile_to_datapath`) accepts `rand()`, `randn()`
  and `poisson()`: it takes the neuron's step counter and stream on the inputs
  `rng_step` and `P_RNG_STREAM`, and with the same values matches the
  per-instance module state for state.

- Add `poisson(lam)` to equation-defined neurons and the equation compiler. The
  count inverts the Poisson distribution on one 32-bit Philox word against the
  thresholds `ceil(2^32·CDF(k))`, summed at 50 significant digits, so the
  Python model, the RTL and the bit-true kernel return the same count. A
  compiled mean must be a constant in `[0, 100]`.
- `exp_euler` differentiates `tan`, `arctan`, `arcsin`, `arccos`, `exp2`,
  `log2`, `log10`, `expm1`, `log1p` and `erf`; random draws stay as
  coefficients.

- Add `rand()` and `randn()` to equation-defined neurons and the equation
  compiler. Each call draws a Philox4x32-10 block at counter `(site, step,
  stream)` keyed by `rng_seed`; `EquationNeuron` numbers the call sites when it is
  built, so the Python model and the compiler read the same text, and takes a new
  `random_stream` argument. Registered RTL gains `P_RNG_SEED`, `P_RNG_STREAM` and
  an `rng_step` register; the bit-true C and Rust kernels carry the same counter.
  `rand()` equals `floor(rand·2^fraction)` of the model exactly and `randn()` (an
  integer Box–Muller) is within 0.5 LSB; RTL equals the kernel. Pipelined,
  multi-substep and folded modules and the HLS lowering refuse draws.
- Fix the bit-true C and Rust kernels: statements needed by a reset expression
  (table reads, function units) were not emitted, so a reset such as
  `exp(v) - 1.5` produced a kernel that did not compile.

- Extend the equation vocabulary with the functions Brian2 and LEMS provide.
  New v2 function units: `tan`, `arctan`, `arcsin`, `arccos`, `exp2`, `log2`,
  `log10`, `expm1`, `log1p` and `erf`. New exact functions `floor`, `ceil`,
  `trunc` and `sign` work in both lowerings. All are bound in the Python model
  to their NumPy functions (`erf` to `scipy.special.erf`) and guarded against
  output arrays like the existing ufuncs. Every 16-bit-format code is correctly
  rounded except `tan` (0.52 LSB). Near a pole, the `tan` error follows its
  conditioning.

- Add function lowering v2 to the equation compiler: `compile_to_verilog(...,
  function_lowering="v2")` (also `compile_to_datapath`, the bit-true C/Rust
  kernel and the hardware numeric contract). exp, log, sqrt, sin, cos, sigmoid,
  tanh, cosh, sinh, exprel and `x ** (1/3)` are computed by range reduction and
  quadratic tables at a 30-bit internal fraction. Each unit is one 64-bit
  integer program, emitted to Verilog, C and Rust, and equal to a separately
  written integer reference. Every 16-bit-format code is correctly rounded
  (cosh and sinh are within 0.63 LSB); `docs/validation/function_units_v2_accuracy.md`
  lists every format. `lut-v1` stays the default and its RTL is unchanged except
  for the fixes below. `sinh` is new and exists only in v2.
- Fix the lut-v1 lowering. `clip` follows numpy (`min(max(x, lo), hi)`), so
  reversed bounds give the upper bound. Table entries that do not fit the word
  saturate instead of wrapping: `cos(0)` at Q1.7 gave -1. A left-shifted
  table index is widened before it is clamped: `exp(64)` at Q8.0 gave 1. The
  accuracy measurement mirrors these widths and handles 64-bit words. The HLS
  C++ lowering no longer turns `sinh(v)` into `v`.

- State the approximation error of the equation compiler's transcendental tables.
  `sc_neurocore.compiler.lut_accuracy` reads each table exactly as the generated
  RTL indexes it (checked by Icarus simulation). It compares the result with the
  Python model's function, at every input code up to 18 bits and sampled above.
  The hardware numeric contract now reports the error for every table a neuron
  uses at its format, and `docs/validation/lut_accuracy_v1.md` lists it per
  function and format. The current lowering is coarse: inside its table, `exp`
  errs by up to 13 at Q8.8 and 3765 at Q16.16, `sin` by 0.125, and `sqrt` by
  0.71.

- Stop silent mis-lowering in the equation compiler. `max` and `min` now compare
  every argument; they used only the first two, and the GLIF5 schema's tolerance
  term has three. A call the Python model would not evaluate as written is now
  refused instead of passed through, in the Verilog, HLS C++ and bit-true C/Rust
  lowerers alike. Such calls include `max(x)`, `clip(x)`, `clip(x, lo)`, keyword
  arguments, and a second argument to a one-argument function. `pi` is encoded
  as a literal instead of an undeclared net. The transcendental-table index now
  holds its offset at formats whose integer range is narrower than 16, such as
  Q4.12, Q1.15, Q4.4 and Q1.7. At those formats the RTL and the bit-true kernel
  both read wrong entries; for example, `exp(0)` gave 0 at Q4.12. RTL at Q8.8,
  Q9.9, Q16.16 and Q8.24 is byte-identical for every bundled schema except GLIF.

- Retain every completed stochastic network benchmark repetition in JSON and
  Markdown, including its seed and step count. Record timezone-aware UTC and
  distinguish measured fused totals from estimated native component times.

- Declare `typing-extensions` in the base runtime and Conda profiles so the
  reference MAT model imports without developer extras. Exercise a real MAT
  step in the measured installation smoke check.

- Keep stateless-model custody checks independent of numerical timebase admission;
  verify real driven instances and preserve refusal of untimed simulation runs.
- Reject optimizer searches with no layers or missing scoring weights before
  candidate construction, returning `ValueError` instead of aborting Python.
  Preserve seeded results and trailing-weight normalization; reject malformed
  native Rust candidate dimensions with `None`.

- Keep Studio model metadata browsable when a simulation timestep profile is
  unavailable. Show constructor fields as read-only and continue rejecting
  simulation requests whose clock cannot be verified.

- Preserve integer seed text across Studio saves, reopened workspaces and
  exports. Bind simulations to admitted constructor parameters and model
  timebases.

- Gate Python package publication and releases on browser replay against an
  isolated installed Studio wheel. Use the same installed interpreter for
  downloaded replay packs and retain browser evidence.

- Preserve exact integer constructor parameters through Studio model requests
  and report oversized floating-point inputs as field errors. Reload model
  metadata before running saved inputs and keep floating parameter controls
  continuous.

- Clarify v2 replay-pack state samples and the retained v1 digest-only
  comparison contract.

- Report browser session save and delete failures beside the session list.
  Keep page changes usable after storage refuses a write, explain reload
  behaviour and distinguish browser sessions from saved server projects.

- Reject oversized GPU LIF and Izhikevich batch outputs before buffer creation
  or dispatch. Return recoverable exceptions for storage, shader-index and
  workgroup limits while preserving accepted trajectories and empty shapes.
  Apply the CPU fixed-point configuration checks to GPU LIF batches, including
  empty runs.

- Update the Semgrep environment to MCP 1.30.0 to address CVE-2026-59951.
  Mark embedded receipt-header examples so REUSE checks the source headers
  without interpreting generated SPDX text as additional file metadata.

- Connect SHD AXI input commits to a shared logical-step enable throughout
  the network. Snapshot complete input vectors, reset each inference, drain
  padding automatically and retain completion for software polling. Exercise
  the AXI interface against Python Q8.8 golden outputs with stalled transfers.

- Recognize guarded optional reference imports in test-inventory audits while
  rejecting unexplained or overly broad module skips.

- Replace the Basilisk-backed Pixi dependency audit with checksum-pinned Syft
  and Grype scanning of the locked Mojo environment. Basilisk now requires
  account access, a granted role and a CI API token; the replacement uses public
  vulnerability data without that login. Document its Conda CPE/NVD matching
  limits and retain exact, evidence-checked exceptions for fixed vendor code.

- Return catchable dense-layer storage refusals for constructors, weight copies,
  packed weight rebuilds and every forward representation. Preserve weight
  replacements and caller-owned batch output on failed allocation.

- Check GPU dense construction against shader index widths and device buffer
  and dispatch limits, and return typed refusals from single and batch GPU
  forwards before submission. Rejected shapes keep packed weights intact.
  GPU-feature wheels re-export `GpuDenseLayer`, `GpuLifBatch`, `GpuKuramoto`
  and `GpuIzhikevichBatch` from `sc_neurocore_engine`.

- Restore the optimizer bridge's per-layer parameter and result descriptions
  lost during wrapper extraction.

- Align candidate API and symbolic comparison tests with equation admission:
  reject unavailable attributes and module-qualified calls before simulation,
  while retaining authored diagnostics for failures of admitted expressions.

- Report the SC Compte benchmark's actual PATH-resolved tools and module-selected
  Go version, with loaded Mojo library identity separate from compiler observations.

- Record facet evidence and refresh the profile ledger, readiness ledger and
  Studio seal with one command, including failed validator outcomes. Preserve
  immutable receipts and detect loss of the established co-simulation labels.

- Describe vectorized SC outputs as channel sums with optional bias. Preserve
  unclipped MAC values above one and signed outputs; they are not firing fractions.

- Correct LDS documentation: finite Sobol and Halton bitstreams can have nonzero
  covariance, even when probabilities use distinct sequence dimensions. Retain
  the existing sampler, scrambling and thresholding behavior. Remove the same
  unsupported independence claim from the RTL barrel-shifter comments.

- Make global Rayon startup retryable after worker creation fails. Preserve
  the shared pool and thread configuration, and return catchable worker-resource
  refusals from batch encoding, fixed-point batch simulation, Kuramoto stepping
  and network execution. Preserve validation precedence and neuron state on
  worker refusal, and leave empty network and zero-step Kuramoto work lazy.
  Validate attention shapes and admit workers before encoding, keeping empty
  queries lazy across linear, softmax, stochastic and multi-head paths.
  Admit graph rate-mode workers after feature validation while retaining serial
  stochastic graph execution. Validate all cortical CSR blocks before output
  mutation and worker startup, including malformed final rows.
  Validate dense-layer dimensions before fallible worker startup in the
  constructor. Check Pareto columns before worker admission, retain lazy DNA
  comparisons for fewer than two strands, and leave empty quantum operations
  lazy. Refuse active annealing couplings with an out-of-range neighbor before
  starting workers; retain zero-read and zero-sweep behavior.
  Admit differentiable dense constructors before weight allocation. Keep empty
  photonic operations and serial MZI/bank analysis lazy, and validate routing
  adjacency storage before starting workers. Preserve zipped column prefixes.
  Admit workers before evolutionary population operations and leave empty work
  lazy. Refuse nonzero tournament selection from an empty population; retain
  serial seeded sampling and ragged gene-vector prefix behavior.

- Exclude Python interpreter caches from native wheel and source distributions,
  and reject cached bytecode before wheel artifacts are uploaded by CI.

- Grow generic bitstream and HDC input storage from actual sequence items,
  avoiding aborts or false allocation refusals from inaccurate length hints.

- Allocate batch-encoded NumPy matrices before sampling and fill their rows
  directly with the retained parallel Xoshiro/SIMD trace. Make scalar batch
  word reservation and Python outputs fallible, preserving ChaCha8 draws,
  empty shapes and exact seeded words.

- Allocate typed packed-bitstream NumPy output fallibly before SIMD packing,
  preserving nonzero-byte semantics and bit order. Return an owning output
  and remove intermediate packed-word and partial-tail allocations.
  Check generic packing input copies, Python output lists and word allocation,
  retaining byte buffers, nested sequences and conversion refusals.

- Accept empty unpacked bitstreams and refuse truncated packed storage or
  unrepresentable lengths with Python exceptions. Allocate unpacked bytes
  and NumPy outputs fallibly, preserving input words and valid bit order.

- Reject empty bitstream-averager windows and unallocatable native buffers
  with catchable exceptions. Refuse nonbinary samples before changing the
  native sliding-window state, matching Python's binary-sample contract.
  Keep Python's running sum in integer arithmetic for large windows and
  NumPy integer samples.

- Update the Studio's locked `source-map-js` dependency to 1.2.2, fixing
  GHSA-68fv-2mgg-jv7q without changing the remaining dependency graph.

- Refuse fixed-point Brunel constructor and simulation allocation with
  `MemoryError`. Allocate the NumPy output before changing network state,
  preserving inputs and seeded continuation after failure.

- Audit Julia manifest formats 2.0 and 2.1, retaining every exact dependency
  version and refusing unsupported formats or local replacements.

- Clear the currently selected pending address in the clocked AER encoder,
  preventing duplicate events from simultaneous spike pulses. Add actual RTL
  mask, address, coalescing, timestamp-wrap and reset regressions, with current
  generic synthesis and optimization-equivalence observations.

- Measure isolated SHD workers in subprocess coverage and make the automatic
  NumPy fallback explicit. Installed-wheel CI now round-trips every public
  engine class/function global through all supported pickle protocols.

- Show the reason again when a Studio training run cannot start on its data:
  a missing torchvision for MNIST and a validation split that serves no samples
  are owned training refusals, so the job status and the sealed status carry
  their text instead of the fixed fallback. A process worker reports each as a
  finite code and the supervisor renders the source-owned wording.

- Add the pinned `hdc-reference` test extra and a required CPU CI comparison
  against TorchHD BSC binding, odd bundling and cyclic permutation semantics.

- Pin the workstation Jupyter Notebook dependency to 7.6.3, fixing
  GHSA-6966-vjj6-99xv while retaining JupyterLab 4.6.4.

- Candidate validation reports authored diagnostics for malformed units,
  excessively deep expressions, unsupported function arguments and invalid
  Unicode or JSON values. Unencodable documents have no candidate digest.
  Expected equation-value failures report a failed simulation step instead of
  leaking interpreter text; corrupt parent resources remain server errors.

### Added
- Pin the workstation notebook dependency JupyterLab to 4.6.4, fixing
  GHSA-3jqq-pw4j-pqcj, GHSA-6966-vjj6-99xv and GHSA-3325-v43h-43rv.

- A training request can declare a preregistered acceptance criterion
  (`val_accuracy` at least or `val_loss` at most a bound, with a rationale). It
  is stored and digested with the job's configuration at submission, and the
  completed run reports a `preregistration_verdict` judged on the unrounded
  validation metric. The Training Monitor declares the criterion, keeps it in
  the workspace and shows the verdict.
- Studio training accepts local N-MNIST, SHD and DVS-CIFAR10 recordings through
  an event contract: a manifest of the files on disk, a whole-group split and a
  declared encoder, digested and stored with the job. The API and the worker
  verify file bytes, labels and groups before and at the end of a run; samples
  reach the network as `(timesteps, batch, channels)` spike tensors on one CPU
  thread; an input budget (`SC_NEUROCORE_STUDIO_EVENT_INPUT_MAX_BYTES`, 64 MiB by
  default) is admitted before a job exists; large declarations are stored beside
  the job record; exact resume requires the contract unchanged. Isolated workers
  receive dataset roots and native readers only from the operator's launcher
  configuration.
- Native event-recording readers: N-MNIST decoders, indexed SHD readers and
  DVS-CIFAR10 readers in Rust, Go, Julia and Mojo beside the NumPy/h5py
  readers. A reader is selected explicitly or from the operator's declared
  configuration; for N-MNIST and SHD, `auto` orders the configured readers by
  the benchmarks measured on this CPU, else by a static order, with NumPy last.
  A configured native reader that fails refuses the read; nothing silently
  falls back.
- `convert` lowers the model's actual forward invocations into a dense
  integrate-and-fire network: shared modules keep every call, unused
  registrations are ignored, consecutive affine maps are composed, each ReLU
  call (module, function or tensor method) gets its own calibrated scale, and
  each QCFS call keeps its learned threshold and half-threshold preload. A final
  affine layer is a signed linear readout. Unsupported operators, residual
  arithmetic, several outputs and input-dependent control flow are refused.
- `ConvertedSNN.replay` runs explicit frames under the
  `dense-if-f64-sequential-v1` profile with complete state and event traces,
  continuation from previous states and a checked working-byte budget; `run`,
  `rates` and `classify` share it. Replay runs in NumPy, Rust, Go, Mojo or Julia
  with identical float64 results; `auto` follows a measured local comparison
  (`SC_NEUROCORE_IF_BENCHMARK`) only when it matches this CPU, these sources and
  the configured libraries, and otherwise a static order before NumPy.
- `measure_conversion_loss` classifies labelled samples with a PyTorch source
  and its converted network and reports both accuracies, their agreement, the
  decoded-rate error, the replay runtime that executed and digests of the
  source, the converted network and the data.
- Studio training has a `qcfs_conversion` route: it trains a QCFS ANN, converts
  it to a dense IF network with the same step budget, reports the converted
  network's validation accuracy as `val_accuracy` beside the source's, and seals
  the comparison as `training/conversion_report.json`. The criterion metric
  `conversion_accuracy_drop` judges it. The Training Monitor's Model selector,
  the HTTP route and `sc-neurocore train` all reach it.
- `calibrate_for_target` fits a converted network into a hardware profile's
  fixed-point format at a per-layer power-of-two scale, replays the rounded
  network on calibration samples and reports rounding errors, measured
  membrane headroom, overflow and the accuracy it costs. A conversion run can
  name a `target_profile` (listed by `GET /api/training/target-profiles`) and
  seals `training/target_report.json`.
- `sc_neurocore.hardware.experiment` defines the hardware experiment protocol
  declared before a device run (operator opt-in and identity, device, image
  digest, network and data digests, latency including transport with declared
  warmup, calibrated power instrument, preregistered criteria) and the receipt
  that seals raw observations against it. The receipt computes accuracy,
  nearest-rank latency percentiles, energy and verdicts itself, refuses energy
  not measured by the declared calibrated instrument, and `verify_receipt`
  recomputes a stored receipt end to end.
- `sc-neurocore train` runs a Studio training request from the command line
  through the same contract and job manager, prints status, metrics and
  verdict, and exits 3 when the criterion was missed.
- `qcfs_forward` and `qcfs_backward` evaluate QCFS quantisation and its
  straight-through derivatives without PyTorch in NumPy, Rust, Go, Mojo and
  Julia. All five return the float64 bits of `QCFSActivation` and its autograd,
  including NaN payloads and the sign of zero. Each native runtime exports one
  QCFS array ABI; `auto` follows the local five-runtime comparison
  (`benchmarks/bench_qcfs_runtimes.py`), and the wheel ships the QCFS sources.
- `sc-neurocore-if-benchmark` runs the five-runtime dense IF comparison from an
  installed wheel. The wheel ships the comparison scripts and the dense IF Rust,
  Go, Mojo and Julia sources, so an installation builds its native libraries,
  measures all five runtimes and admits its own report. Reports bind the source
  bytes of the imported package and of the scripts that ran; a report captured
  in a source checkout is admitted by an installation with byte-identical
  sources.
- Typing scanner reports use supported Mypy JSON diagnostics and retain actual
  commands, versions and execution failures. Explicit file scopes are labelled;
  stale reports and mismatched tools cannot qualify a successful scan.

- Python dependency auditing covers every maintained hashlocked profile and
  every marker branch. Reports retain original pins and hashes, explicitly
  identify the official Torch CPU upstream advisory query, and fail on missing
  dependencies or incomplete coverage. Release evidence rejects the former
  release-tooling-only report as unqualified coverage.

- Laboratory HTTP refusals preserve explicitly authored messages and hide generated numeric-conversion and replay-validation exception text.

### Changed

- `QCFSActivation` accepts `T` in `[1, 2**32 - 1]`, the step domain shared by
  every QCFS runtime.
- Align Ruff hooks, development declarations, and hash-locked CI environments
  on 0.16.9. The formatter retains its Python, stub, and notebook scope.

### Fixed
- MAT IB and FS factories accept explicit baseline and history-increment
  overrides through complete constructor validation, preserving named defaults.
- Source `MATNeuron` reset validates retained configuration before committing
  zero-rest dynamics. Checked Rust and Go resets report refusal; legacy void
  resets preserve refused states. Julia validates empty inputs and preserves
  caller state on any batch failure, with complete constant-current admission.
- MAT Go and Mojo C ABIs reject invalid complete profiles even for empty
  batches and refuse null pointers before output writes. Mojo enforces the
  source threshold envelopes and retains SC resetting-MAT derivative arithmetic
  order, preventing a false event for large finite cancelling terms.
- `SCResettingMATNeuron` validates its complete resting candidate before
  changing state. Checked Rust and Go resets preserve invalid configurations;
  Julia rejects invalid empty inputs and preserves caller state when a later
  batch sample fails. Native source MAT and SC resetting-MAT batches use owning
  arrays and propagate allocation failures as Python `MemoryError`, releasing
  partial outputs.
- `SCNonResettingAdaptiveLIFNeuron` validates its complete resting candidate
  before reset commits either dynamic value. Its native batch propagates
  allocation failures as Python `MemoryError` and releases partial outputs.
  Julia's constant-current simulator rejects invalid current or timestep even
  for zero samples; valid empty calls retain their established result.
- The published Studio OpenAPI reference matches the running application again.
  Three landed changes never reached it: `GET /api/graph/models/{name}` was
  absent entirely, and `/api/export/svg` and `/api/graph/validate` carried
  outdated response schemas. The reference is published through MkDocs, so the
  public API documentation was describing an application that no longer exists.
  The pinned backend route count moves 125 → 126 for the added route.

- Use the SHA-256 verified official Python 3.12 reference inventory to keep
  strict Sphinx documentation builds available during upstream outages.
### Changed
- The Studio network interchange envelope says what it is. Export wrote
  `{"format": "nir", "version": "0.1"}` for a payload whose node `type` is a
  catalogue model name and whose edges carry a weight and a delay — no NIR
  primitive is mapped anywhere in it, and the importer never checked the field
  at all. It now writes `sc-neurocore.studio.network-graph` at version `1`.
  The loader still accepts the legacy `nir` envelope and a payload that
  declares no format, so nothing an operator saved stops opening, and it now
  refuses an envelope it cannot read — a real NIR file used to fail on an
  unrecognised node type instead. The route names are unchanged.

- Rust engine, standalone core crates and fuzz lockfiles use the maintained
  `chacha20 0.10.2`, replacing the yanked `0.10.0` with its SSE2 backend fix.
  The engine fuzz lock also includes the existing engine SHA-256 dependency.
### Fixed
- The Studio pipeline no longer reports a hardware result for a graph it cannot
  lower. `POST /api/pipeline/run` validated and simulated the caller's network,
  then compiled one hardcoded leaky integrate-and-fire equation — ignoring the
  graph entirely — synthesised it, and returned success labelled
  `graph → simulate → compile → synthesise`. An operator building a
  Hodgkin-Huxley network received a successful synthesis report for a generic
  neuron that shared nothing with it. The compile step now refuses, names why,
  and lists any declared model whose descriptor records no silicon lowering.
  The validation and simulation it genuinely performed are still reported.

- The install-profile audit recognises packaged dataset and conversion runtime
  resources, rejects broader native research patterns and missing runtime files,
  and records the current packaging contract in its derived report.
- Event training initialises its selected N-MNIST decoder before Torch,
  including warm starts and exact resumes. This avoids JuliaCall's unsafe
  Torch-first import order; parent-process custody checks no longer import Torch.
- The Studio's bifurcation sweep ran every point under a sine drive whatever
  the reader had chosen: its request schema had no protocol field, so the
  protocol the Studio sent was dropped. For the default catalogue model the
  sine's negative half-cycle left the model's safety bounds at t = 62.4 ms
  for every swept value and step size, so the bifurcation view failed every
  time. Bifurcation and 2-D sweeps now run under the requested `protocol`
  (and `frequency_hz`), default constant, and their results state the drive.
- A failed simulation is said in full ("ATypeKNeuron (python) simulation
  failed at step 173 (86.5 ms): …") instead of `model_simulation_failed`,
  a failed analysis job is explained instead of showing an exception class
  name, and the error banner can be dismissed.
- Characterisation no longer waits minutes at "Starting characterisation…
  0%" when its progress socket does not open: after five seconds it runs over
  HTTP. The development server and preview now forward `/ws` to the API.
- The Studio can be used from the keyboard and read by a screen reader
  throughout. Choosing a model was a click on a `div`, so no model could be
  chosen by keyboard; model rows, saved sessions, experiment presets, the
  behaviour filters, the onboarding step dots, the ISI shortcut and the
  code one-liner are now real buttons with names, the chosen model marked
  `aria-current`. The model search, current protocol, model B, FPGA target
  and E-I sliders have names; the 133 links called only "DOI" say whose
  source they are. Every "docs" and "Documentation" link returned 404 (it
  pointed at the Studio's own server); they now open the published
  documentation, and a test requires each capability's page to exist. The
  left-panel sections are headings.
- The Operator workbench's buttons do what they say: "Save project" asks
  for a name and saves, "Browse models" puts the keyboard in the model
  search, "Open projects" shows the project list. All of them used to switch
  the view to the trace.
- Saving a project or session and importing a trace use a labelled Studio
  dialog (focus held inside, Escape closes it, a multi-line field for a
  pasted trace) instead of the browser's `prompt()` box.
- When the capability registry cannot be read, a banner says so and offers
  Retry, instead of a "capability check failed" note with the reason in a
  tooltip beside a Studio in which every control was silently disabled.
- The page is dark from its first paint and says it is loading, instead of a
  white page until the bundle ran. The left panel no longer scrolls sideways
  (a model's comma-joined state-variable list and fixed-width slider values
  pushed it out), the admin identity form wraps in a half-width column, and
  admin status words and metrics are no longer cut off.
- The Studio's E-I network produced no spike for any setting its controls
  could reach. Both implementations (Rust engine and NumPy fallback)
  multiplied each delta-synapse jump by the step size, so a spike moved the
  membrane by weight × dt (0.01 mV for the default E→E weight), and gave each
  neuron one external input, so the whole 0.1–100 Hz slider range stayed
  silent; activity began near 1,000 Hz, which is what the engine's own test
  used. The network is now the Brunel (2000) style model the Studio page
  describes: weights are millivolt jumps, every neuron receives 800 external
  Poisson inputs of 0.1 mV, the drive reaches threshold at about 9.4 Hz, and
  the default 12 Hz gives about 31 Hz with balanced recurrence. The mean rate
  now counts silent time (it averaged only the bins that held a spike), the
  Poisson sampler no longer loops forever once `exp(-mean)` underflows, the
  route refuses a negative or absurd external rate, and the E→I and I→E
  weight sliders, whose labels were swapped against the model's post-pre
  names, are relabelled and named for screen readers.
- A Studio plot view with no result of its own no longer shows the voltage
  trace under its own name. The 2-D sweep, bifurcation, sensitivity, f-I
  curve, STA, frequency response, characterization, multi-model, A/B compare,
  Q8.8 precision and E-I network views fell through to the trace, and a
  screen reader was given the trace's description. Each now states that it is
  empty and what produces its result (the sweep parameters it still needs, or
  the action button that fills it), and every plot view has a heading. A view
  is called the same thing in the switcher, its heading, its empty state and
  its unavailable notice.
- The Studio header is navigable. It carried one wrapping row of about forty
  controls: abbreviated view buttons ("Bif", "Sens", "2D", "Char") with
  nothing saying which view was showing, four views repeated as buttons with
  the same names ("Canvas", "Train", "Admin", "Freq"), sweep selects whose
  only name was a capability tooltip, and an analysis control that showed raw
  refusal identifiers such as `analysis_selection_heatmap_param_x_blank`. The
  header now has three rows: identity and source; actions in named groups
  (Simulate, Analyse, sweep parameters, Code and hardware, Import and export)
  with verb names and three visual weights; and a view switcher that is a real
  `tablist` with full names in five groups (Neuron, Network, Code and
  hardware, Research, Operator), the selected view announced and underlined,
  arrow keys, Home and End moving between views, and the view area as its
  `tabpanel` and the page's `main` landmark. RTL and FPGA synthesis are listed
  in both source modes, because both can be opened from either. A refused
  analysis request is explained in a sentence, with the identifier kept as
  `data-error-code`. On a phone-width screen the page scrolls as a whole
  instead of the header covering most of it, the page title is no longer
  removed from the accessibility tree, and the view switcher no longer
  overhangs the page at 200 % zoom.
- The mocked browser run (`npm run test:e2e`, run in CI) no longer includes the
  three live training specs added with the train-to-hardware chain; they need
  a real backend and failed there on every run. A unit test now requires every
  live spec to be excluded from the mocked run and matched by a live config.
- The Studio's text is readable. Measured in a real browser on 2026-09-29,
  most of its text was set at 8 to 10 pixels; the Studio now has a type scale
  whose floor is 11 pixels for dense metadata and 12 for body text, including
  the axis labels drawn into the plots. Text fields, text areas, file pickers,
  checkboxes and radios follow the dark palette instead of rendering as white
  browser defaults (the fitting laboratory, the training monitor and the
  candidate editor were affected), and every focusable control, links and
  disclosure summaries included, shows the keyboard focus ring.
- Opening the Studio's Delays view no longer freezes the whole server. The
  view asks for the kernel information and the parity evaluation at once; in a
  server process that had not yet loaded PyTorch both requests probed the Julia
  backend from separate request threads, JuliaCall deadlocked, and no route
  answered again, `/api/health` included. The Studio server now declares Julia
  for the DCLS kernel but never runs it in its own process, as it already did
  whenever PyTorch was loaded; the kernel's Julia parity stays covered by the
  offline parity suite. A test sends the view's requests concurrently to a
  fresh server process and requires every answer without JuliaCall loaded.
- Public formal-verification counts now match the git-tracked `hdl/formal/`
  inventory everywhere (90 proof jobs, 447 statements: 296 assert, 115 assume,
  36 cover). The README diagram said 61 jobs; the formal-verification
  tutorial's table said 293/112/441; five further pages said 18 jobs and 130
  statements, and the v3.13 report presented those as the current inventory.
  A test now scans every public page for such counts instead of three named
  pages, so a new page with a stale count fails.
- The model fidelity page named `this commit` as the evidence for 31
  polyglot-complete models. Each row now names its committed benchmark,
  receipt, trace or formal file, and a test requires every evidence cell to
  name a tracked file or an existing commit. The COBA LIF pages described its
  formal job as a generated depth-4 reset-safety check; since the curated
  harness landed it is a depth-8 job asserting reset, first-event and
  refractory behaviour, and the pages now say so.
- Conversion restores the caller's accelerator random generators as well as
  Python, NumPy and PyTorch CPU ones while user forward code runs, and
  `calibrate_activation_thresholds` now restores them at all. Custody sections
  are serialised, so two conversions in one process no longer restore each
  other's states.
- A saturated infinite or NaN element no longer turns a batch's QCFS threshold
  gradient into NaN; only interior elements carry a gradient, and infinite
  inputs contribute their own saturated derivative.
- Julia replay and Julia QCFS no longer reparse the locked Julia project on
  every call: an admitted configuration is reused while every setting, option
  and file identity it read is unchanged (about 2.4 ms to 35 µs per call).
- The Julia lane is reported present only for the model kernel sources. The
  dense IF conversion and event-dataset Julia sources the wheel ships for their
  own APIs no longer make an installation with JuliaCall claim the Julia lane.
### Fixed
- Three Rust safety validators check something. `validate_astrocyte`,
  `validate_astrocyte_adapter` and `validate_quantum_inspired_lif` returned
  `true` unconditionally; each now mirrors exactly what the maintained model
  enforces at construction — non-negative calcium and IP3, a gating variable
  within `[0, 1]`, strictly positive ER, SERCA, receptor and timestep
  parameters, cytosolic calcium below the total cell calcium, and a whole-valued
  seed in `[1, 2**64)`. 15 cases were added; reverting the three bodies to the
  old stub fails 12 of them, which is what makes them evidence.
- A module comment in the Compte safety kernel documented the constant beneath
  it rather than the module, so `cargo clippy -D warnings` failed on the crate
  before any of the above could be verified.

- Studio embedded process jobs now take per-process data, CPU, descriptor,
  output-file and core-dump limits before importing task code. A refused
  allocation ends as a failed job; nested Yosys limits respect the worker's
  inherited hard ceiling.
### Added
- `tools/benchmark_coverage_census.py` counts how many catalogue models have a
  benchmark measured, so the figure stops being asserted by hand. It separates
  three states — a committed record naming the model, a benchmark script naming
  it with no record, and neither — because a script that exists is apparatus,
  not a measurement. Measured now: **28 of 185 measured (15%), 73
  apparatus_only, 84 absent**. It answers a different question from
  `tools/benchmark_evidence_gate.py`, which judges the quality of the artefacts
  that exist rather than how many models have any.

- ANN-to-SNN conversion no longer changes the network it converts. It took
  layers in registration rather than forward order and counted a shared module
  once; accepted `Conv2d` weights the dense runtime could not execute; added
  each bias as `bias / T` per step, so a constant bias reached the output
  diluted by the timestep budget; made the final layer fire, losing negative
  outputs; calibrated only registered ReLU modules; and padded missing QCFS
  thresholds with 1.0. `replace_relu_with_qcfs` also left aliases of one shared
  activation with separate thresholds.
- `sc-neurocore deploy` now exports the network a `.pt` checkpoint holds. It
  converted the network and then discarded it, ordered layers by key name (so
  `10.weight` came before `2.weight`), zeroed every trained bias, calibrated on
  random normal inputs and reported a Q8.8 quantisation it never performed. It
  now rebuilds layers in registration order with their biases, calibrates on
  `--calibration` samples (or unit scales), refuses parameters a dense ReLU
  chain lacks, rebuilds Studio `qcfs_conversion` checkpoints exactly, writes
  `converted_network.npz` with a digest manifest and, with calibration samples,
  `target_report.json`. The generated RTL is labelled as the generic template
  it is. The tutorials' checkpoint examples now pass the required digest.
- Reading an event-training job record no longer re-validates its whole event
  declaration every time. With the SHD publisher manifest one read took about
  275 ms, so `sc-neurocore train` and any status poll held a full CPU core for
  the length of a run. A validated snapshot is now reused while its stored
  bytes and the admission limit are unchanged (about 9 ms), and waiting for a
  job backs off to half a second between durable reads.
- The SHD event-training example binned 100 steps of 1 ms, keeping only the
  first tenth of each one-second utterance; it now uses 10 ms steps.
- The CLI architecture test now lists the `train` command.
- The training weight fingerprint no longer treats `hidden: []`, the direct
  input-to-output network, as the default single hidden layer of 128.
- The Studio training architecture guard now covers every `_training_*`
  module; a new module could previously escape its size and dependency limits.
- `ConvertedSNN` docstrings described an auto replay order without Mojo and
  without the measured comparison; they now state the order dispatch uses.
### Fixed
- Benchmark records name the compiler that actually built the lane. `GOTOOLCHAIN`
  resolves a Go toolchain per module, so `go version` asked at the repository
  root reported the installed Go — 1.24.0 on the measuring host — while the lane
  is built inside `accel/go` by the 1.26.7 its `go.mod` requires and hosted CI
  pins. And `mojo` on `PATH` was a prerelease, while the build and CI both use
  the version `accel/mojo/pixi.toml` pins. Nine records published compilers that
  did not build their lanes; all now record `go1.26.7` and `Mojo 1.0.0`, matching
  the CI pins. The probes ask Go inside the accelerator module and Mojo through
  the pinned manifest, the way CI exposes it.
- A run of Brunel-Wang or Compte-WM now records the role its own profile states.
  Two defects met in one variable. The committed descriptor calls the refractory
  register `ref_remaining` while the canonical schema calls it
  `refractory_time`, and the role join matched by name alone, so the variable
  came back `unassigned` on every run. Separately, neither schema authored a
  `[profile]` section, so all eleven and thirty-one of their state variables
  resolved as `biological` — including the ones each schema's own
  `state_scope_note` calls deterministic lowering registers. A role split where
  everything is biological carries no information.
  Both schemas now author the split their note already stated: 2 biological and
  9 auxiliary for Brunel-Wang, 6 and 25 for Compte-WM, resolving with no
  contract problems. The join carries an explicit per-identity record of the
  descriptor's word for the refractory register; the schema's word is never
  repeated, because the profile names it itself and the two cannot drift apart.
  No public name changed, so committed conformance evidence and operator-visible
  trace fields are untouched.
- Seven benchmark gates that were red on stale evidence are green again. Their
  records bound source digests that no longer matched the files they name —
  `ci.yml`, `pyproject.toml`, two model and accelerator sources, two schemas and
  the equation builder — so a reviewer could not tell a stale digest from a real
  parity failure without re-running. Each was re-run at its own recorded
  configuration, on the CPU its record names, and two of them additionally moved
  from a slightly older release extension to the current one. No published claim
  changed: every parity, event-count and safety assertion held before and after.
- Five published benchmark records no longer rate the Rust lane against a debug
  build. They bound a 525 MB unoptimised engine extension where every other
  record bound the ~35 MB release build, so the Rust lane was measured several
  times slower than it runs: 2.3x to 5.8x on the four records that carry
  per-lane timings, and in all four the lane moved from *last* of the four
  native lanes to first or second. The measurement configuration of each record
  is unchanged — same step count, same drive, same pinned CPU — so the binary is
  the only thing that differs. The whole record is refreshed, and the machine's
  Go, Julia and Mojo toolchains are older now than when the records were taken,
  so the other lanes' absolute times moved too and are not comparable with the
  August figures; the Rust correction is the claim being made.
- A benchmark rerun can no longer lose the name of the compiler that produced
  it. The shared runner behind four of those records asked for versions only
  through `.venv/bin`, where three of the four shims do not exist, so every
  rerun silently recorded `unavailable` for compilers that were installed and in
  use. A pinned shim is now preferred when present and the tool is otherwise
  resolved on `PATH`, which is how the rest of the benchmark surface finds it.
  `tests/test_benchmark_evidence_provenance.py` refuses both defects across the
  committed corpus: no record may bind an unoptimised extension, and no record
  that names a toolchain may leave one unrecorded.

- Studio training: a run that asks for MNIST on a host without torchvision now
  fails with that reason. It used to train on the synthetic demonstration data
  while recording itself as an MNIST run.
### Changed
- The eight benchmark files this touched are now inside the docstring ratchet
  (`docs/docstring_policy.toml`), so their public surface is enforced rather
  than merely documented. The benchmark tree was entirely outside that scope.
- A model that holds no state can now say so. `declared_state` answered
  `undeclared` both when a descriptor was absent and when it declared nothing,
  so `McCullochPittsNeuron` and `SiegertTransferFunction` — which evolve nothing
  by construction — were reported with `complete=false` for ever and could not
  be told apart from a model whose state nobody had declared. A descriptor now
  asserts `stateless` per identity, and such a model answers `descriptor` with
  no variables and complete custody. The published `studio.state-layout.v1`
  payload is unchanged: `source` keeps its three values and no field was added
  or removed.
  The assertion is never inferred — no static rule and no single drive can tell
  "holds nothing" from "was not exercised" — and it stays falsifiable, because
  an asserted-stateless model that mutates anything is still reported
  incomplete, naming what moved. Both identities were measured over 100
  exercised steps with varying output before the assertion was recorded.
  `InhomogeneousPoissonNeuron` mutates no attribute either and is deliberately
  not asserted: it draws from the process-wide NumPy generator, so its run
  cannot be reproduced from anything recorded.
- The Amari neural field no longer records its site count as state. `n`, the
  number of uniformly spaced sites on the periodic ring, is fixed at
  construction and read by every step without ever being written, yet the
  descriptor declared it in the state table with `init = 64.0`. A run therefore
  traced a constant per step, and the runtime state census charged the Rust
  batch lane with dropping it — a conformance gap that existed only because the
  descriptor asked the lane to carry a number that never moves. The model's own
  documentation page has always listed `n` under Parameters, and the Rust,
  Julia, Go and Mojo lanes all take it as a size argument beside the parameter
  block. It is now a curated parameter with unit, meaning and range, and the
  census reports 415 dropped variables instead of 416. Nothing about the model's
  behaviour changed.
- A variable reclassified from state to parameter can now complete the move.
  The descriptor merge preserves curated state entries the regeneration cannot
  produce, because runtime state assigned outside the dataclass field list
  exists nowhere else; the rule had no exception, so a name moved to the
  parameter table kept its old curated state entry and the descriptor declared
  it in both tables at once.
- Spike gating returns the same numbers as an ungated run. A neuron was skipped
  when its input was zero and its voltage sat within one percent of rest, which
  froze the leak, the adaptation current and the refractory countdown of
  exactly the neurons that still had relaxation to do; the documented claim
  that skipped neurons still decayed by leak was not true of the code. A neuron
  is now skipped only when its input is exactly zero and its whole state
  matches a state that model's zero-input map was measured to leave unchanged,
  so a gated run and an ungated run agree exactly. Across the catalogue, 179
  models are comparable under a zero drive and none of them now differs;
  previously 108 of 185 had no fixed point at rest and were skipped anyway.
  Fewer neurons are skipped than before, which is the cost of the agreement.
- `backend='auto'` no longer runs a different network than the one it was
  given. The native network runner receives a model name and a neuron count, so
  it builds default neurons; a population carrying constructor parameters,
  independently derived seeds, or state from an earlier run was replaced by
  defaults while every Python object went on reporting the caller's values. A
  two-population AdEx network with `v_threshold=-45` and `v_rest=-60` ended at
  a summed membrane voltage of -1273.982376 in Python and -1299.940210 under
  `auto`, the latter being the default network's own result. `auto` now runs
  such a network in Python and a forced `backend='rust'` raises, naming the
  population and the attributes; the MPI rank-local runner keeps the Python
  path for a rank it cannot reproduce. Neurons changed in place before a run,
  and a second `run()` on the same network, were discarded the same way and are
  fixed with it.
- Importing `sc_neurocore.network` no longer provisions a Julia toolchain. The
  cortical column discovered its native accelerators, and the gamma-oscillation
  circuit loaded its Julia kernel, while their modules were being read, so
  importing any module in the package resolved and could install a Julia
  environment. Both now load on first construction of the object that uses
  them.
- A run's custody audit sees a random generator advance. Attribute
  fingerprinting used `repr`, which for a NumPy generator is the address-based
  default and does not change between draws, so `GLMNeuron`,
  `GammaRenewalNeuron` and `SCStochasticRateAdaptationNeuron` reported complete
  custody while the register that determines their whole trajectory moved every
  step. A generator now fingerprints by its generator state, and a value that
  cannot be fingerprinted is recorded as opaque rather than reported unchanged.
- Every neuron of a population receives its own random stream. The population
  constructed every neuron with identical keyword arguments, so a seeded model
  gave all of them the same stream; each neuron now takes a distinct seed
  derived reproducibly from the population's base seed.
- `sc_neurocore.benchmarks` imports without the optional training extras. The
  package eagerly imported a Torch-backed module, so the metrics, the MLPerf-SC
  schema, the runner and report, the online-O1 benchmark and the task registry
  were all unreachable in an installation without `training`, `research` or
  `full`. The Torch-backed names now resolve lazily and raise at the point of
  use.
- The Studio simulation plot draws its time axis from the run rather than from
  the request, so an exported figure carries the timebase the model actually
  ran at.

### Documentation
### Added
- `Projection.delay_steps` reports the delay that executes, beside `delay`
  which keeps the value that was asked for. The class documents that its delay
  is in timesteps rather than milliseconds, that rounding is half to even, that
  a positive value below half a step is floored at one step, and that the
  scalar and per-synapse forms round differently.
- A model may declare a state variable it holds as a vector. The descriptor's
  initial value is now optional, because a compartment vector, a ring-attractor
  activity profile or a competing-unit array has no single number to declare as
  its start, and the layout reads a numeric list or tuple as the vector it is.
  `ContinuousAttractorNeuron`, `LeakyCompeteFireNeuron` and `RallCableNeuron`
  declare and record their state and report complete custody;
  `AmariNeuralField.u` and `SCResettingParallelSpikingNeuron.buffer` no longer
  claim a scalar start they do not have.

- Studio: "Reviewing and Sharing Experiments" (`docs/studio/collaboration.md`)
  describes the procedure two researchers follow to review each other's work.
  It covers what to hand over (transfer document, replay pack, notebooks) and
  how to check that a run reproduces and that both sides are on the same
  state digest. It also covers commenting on a revision, keeping both sides of
  a refused save, and what is not provided.
### Changed
- `BrunelWangNeuron` and `CompteWMNeuron` expose `ref_remaining` under the name
  their descriptors declare and `get_state` already returned, in place of the
  private `_ref_remaining`. The refractory guard's message names the same
  attribute.
- Deleting a population on the Network Canvas now deletes it. The canvas
  computed the new node array from a change batch and then wrote back positions
  only, so a removal was computed and discarded: the population and every
  projection touching it stayed in the graph and reappeared on the next render.
  `removePopulation` already existed in the store, removed the population with
  its incident projections, and was called from nowhere in the frontend. A
  change batch is now read as moves and removals separately, so a delete reaches
  the graph and a drag still changes the layout only — a position travels with
  the graph but never changes what is simulated. Deleting a population deletes
  the edges into and out of it, because a network cannot hold an edge whose
  endpoint is gone; `docs/studio/network-canvas.md` states that, and that there
  is no undo yet.
- `QuadraticIFNeuron` accepted foreign event arrays its six siblings refuse. Its
  complete-packet validator cast the array to `uint8` before checking it, so
  `256` wrapped to `0` and `0.5` truncated to `0` — a spike returned by a Rust,
  Julia, Go or Mojo lane became a silent non-spike — while `-1` and `2` were
  caught. The event domain is now checked on the values as they arrived, as
  every other model already did, and a contract test drives the same four
  out-of-domain arrays through every model that has such a boundary so one
  validator cannot drift away from the others again.
- A foreign runtime may no longer name a state variable the model does not
  have. The Studio's Rust batch lane transports one scalar trace, the soma
  voltage, and the payload placed it under `states["v"]` unconditionally.
  Measured over the catalogue: the lane runs 158 of the 185 models; of the 152
  with a declared layout it carries 111 declared variables and drops 356; and
  41 of them declare no `v` at all — `PinskyRinzelNeuron` declares
  `v_s, v_d, h, n, s, c, q, ca` — so the payload named a trace after a
  variable that model does not have, while the layout printed beside it
  correctly reported `recorded: []`. The lane now records its trace only under
  a declared name, and a model it can name nothing for records no state and
  says why in its custody notes. `sc-neurocore.runtime-state-packet.v1` states
  what a lane transports and how much of a model's declared state that
  accounts for.
- The same boundary now validates structure before anything is built from it.
  A 100-step request against a ten-sample drive returned ten voltages into a
  payload built for 100, so the reported rate covered a duration the trace
  never spanned; spike indices were used as sample positions without a check
  that they fall inside the run or arrive in order. Both are refused by name.
  Finiteness is deliberately not part of this check: a diverged membrane
  voltage is a failed simulation, and is still reported as one at the step
  where it diverged.
- Pressing Stop on a training run that had just finished returned a server
  error. `cancel` reads the job's status and then writes the transition, and
  the read is outside the ledger transaction, so under load a run reached a
  terminal state in that gap; the ledger refused `cancelled -> cancelling` and
  the refusal reached the operator as HTTP 500. Cancelling a job that has
  already stopped is now a no-op even when it stops inside that window, and a
  refusal for any other reason still propagates. `POST /api/training/stop` also
  reports the state the run actually reached instead of claiming `stopping` for
  a run nothing can still affect.
- An exported evidence pack can now be rechecked by whoever receives it.
  Measured on the real surface before this change: a model run of `AdExNeuron`
  recorded `result_sha256` `771d8e51…`, and the identical payload — after the
  JSON round trip every exported artefact makes through the operator's browser —
  re-sealed to `813d3005…`. Nothing about the run had changed; `1.0` had come
  back as `1`, `-70.0` as `-70` and `1e-07` as `1e-7`. No verifier could exist
  while a recorded seal failed to reproduce for every honest run.
  `studio.evidence-seal.v1` encodes values rather than one runtime's rendering
  of them, and the server and the browser implement it identically against one
  shared vector set. A value that would not survive the trip intact — a
  non-finite number, an integer no double holds exactly, an unpaired surrogate —
  is refused rather than silently altered.
- Every Studio evidence payload now carries a receipt
  (`studio.evidence-receipt.v1`) in one shape across the simulation, analysis,
  training, compile, co-simulation and synthesis lanes: the seal of its subject,
  the identity it was produced under, the inputs it rests on named by value
  rather than by position, and whether it attests production or only the export.
  A bundle is verified before it is written (`studio.evidence-chain.v1`) and
  refuses to publish a pack that contradicts itself — a payload that no longer
  matches its seal, two subjects that disagree about the model or the numerical
  profile, evidence produced before the input it claims. A pack exported without
  its inputs is reported as incomplete rather than refused, and `verified` and
  `complete` are recorded separately.
- Continuing a training run is now a different operation from starting one
  from its weights, and the difference is measurable. Restoring weights alone
  is a **warm start**: a new run with a fresh optimiser and generator at epoch
  zero, which discards Adam's moment estimates and restarts the shuffle order,
  so it does not reach where the interrupted run was heading. **Exact resume**
  carries the optimiser state, the Python, NumPy and Torch generator states as
  they stood at the epoch boundary, and the epochs already completed. Training
  two epochs, and training one then resuming for the second, now produce
  identical metrics. `POST /api/studio/training/weight-restore/attach` takes a
  `mode` of `warm_start` (the default, unchanged) or `exact_resume`, and a
  resume into a different architecture or configuration is refused rather than
  approximated; asking for more epochs is not a difference. A checkpoint
  written by a build that recorded no position supports a warm start and says
  so. The saved position travels inside the weight checkpoint as tensors and
  plain integers, so the artefact still loads under `weights_only=True` — a
  checkpoint arrives from a user, and unpickling one would trade that
  guarantee for convenience. Checkpoints also record a dataset fingerprint
  covering the dataset length, batch size, sample layout and the first and
  last samples; the boundary of what it detects is documented rather than
  implied.
- The Studio trains what was asked for, or refuses before it starts. Measured
  through the public HTTP surface: a request for hidden widths `[128, 64]` on
  `cifar10` with surrogate `not_a_real_surrogate` completed successfully, and
  its exported checkpoint recorded that request beside the architecture
  `64->128->128->10` — 128 twice, 64 nowhere, on synthetic data, with the
  default surrogate — both blocks sealed under one `config_sha256`, so a
  digest-verified artefact attested a configuration that never ran. Every
  request is now resolved against a typed contract
  (`studio.training-config.v1`) before a dataset is loaded or a model is built:
  an unsupported dataset or surrogate, a width that is not a positive integer,
  or a field nobody reads is refused with HTTP 422 naming the field and the
  supported set. `SpikingNet` accepts a width per hidden layer, so `[128, 64]`
  builds that network; a single width with a layer count still works. The
  capability routes and the contract share one list, so the Studio cannot offer
  a name its runner would reject.
- A training run is replayable from the seed its checkpoint records. The seed
  now travels in the request and is applied to the Python, NumPy and Torch
  generators before the loaders and the model exist; two runs with the same
  seed and configuration produce identical metrics, and a different seed does
  not. It used to depend on whatever the calling process had seeded.
- A process worker handed a configuration nobody can run seals failed evidence
  saying why. No HTTP client sees that refusal, so the sandbox is the only
  place the reason can survive.

### The model catalogue is built once on a cold start
### Added
- The Network Canvas has undo and redo for graph edits: `Ctrl+Z` and
  `Ctrl+Shift+Z`, or the toolbar buttons. Adding, deleting and changing a
  population or a projection are edits; **moving a node is not**, because a drag
  writes a position on every frame and recording those would bury the edits
  worth undoing under layout noise. Deleting a population removes every
  projection touching it, which is the edit a user most wants back and
  previously cost a return to a saved workspace revision — discarding
  everything done since. The history holds the last 50 edits, lives for the
  session rather than being saved with the workspace, and drops the redo branch
  when an edit follows an undo.
- `docs/_generated/runtime_state_conformance.json` records, per catalogue model
  and per lane, which declared state variables a foreign runtime could carry,
  which it would drop, and whether it exports anything the model has no name
  for. Across all 185 models the Rust batch lane could carry 123 declared
  variables and would drop 415; it fully accounts for 18 models and can name
  nothing in 52. It is derived from the declared layouts and the lane packets,
  so it is reproducible from committed sources, carries no timestamp or commit
  hash, and a test fails when the file and the live contract disagree. It
  deliberately does not report whether a lane is built or installed on a given
  machine: that is a property of a toolchain rather than of a contract, and the
  binding coverage tests hold that question.
- `tools/studio_evidence_verify.py` rechecks an exported evidence pack without
  the Studio: it recomputes every file digest against the manifest, re-seals
  every subject against its receipt, re-resolves the dependency graph, and
  compares its own finding with the verdict the pack records — the exporter
  wrote both the evidence and that verdict. It exits non-zero when a file is
  missing or altered, when a subject no longer matches its seal, or when the two
  verdicts disagree. `docs/studio/evidence-verification.md` documents the
  receipt fields, the verdicts, and what a receipt does not claim.
- The reference-trace corpus now says, per identity, what it is a reference
  *for*. Backends agreeing with each other can reproduce one shared scientific
  error, so `provenance.kind` — previously free text — is a closed vocabulary
  adjudicated into a derivation (how the expected values were produced) and an
  attribution (what the formulation is sourced from). The two combine into the
  class a claim may use: 34 traces rest on an externally published source, 1 on
  published values, 1 on a pinned third-party implementation, and 9 on a
  formulation this repository retains with no whole-model publication to be
  independent of. Four of those nine declare an independent or analytic
  derivation, which is accurate, while citing a project recurrence — the
  citation settles the class. A trace declaring an unadjudicated kind is now
  refused at load time.
- Negative controls for the same corpus. Five deliberate errors — a flipped
  drive sign, a unit scale off by three orders of magnitude, the drive removed,
  events recorded on the wrong side of the state update, and a threshold
  displaced by up to 100 % — are applied to every deterministic trace, and each
  outcome is recorded as detected, refused (the mutated run left the numeric
  domain), inapplicable (the mutation provably changes nothing about that
  protocol) or undetected. No trace in the corpus survives every control. Six
  applied controls do not bite and each is stated: a logical all-or-none unit is
  scale-invariant above threshold and its truth table cannot distinguish
  pre-update from post-update state, and four conductance spiking traces
  overshoot far enough inside one timestep that they validate the trajectory
  rather than the threshold surface.
- `tools/reference_trace_evidence.py` prints both surfaces and exits non-zero
  when a trace cannot be adjudicated or survives every control.

- Opening the Studio asks for the model list, its facets and a query at the
  same moment, and each request built the whole catalogue on a cold server;
  with requests still running from pages already closed, the builds competed
  for the interpreter and the list could take most of a minute. Concurrent
  first requests now share one build.
### Fixed
- A workspace saved before revisions existed is no longer invisible. The
  previous store kept one flat `<name>.json` per workspace in the project
  root, and a store that reads `<name>/head.json` found an existing
  installation's saved work simply gone. Such a file is now adopted as
  revision 1 on first access — read, list, save and history alike — and the
  flat file is left on disk untouched, so the adoption is reversible. Adoption
  is idempotent, and a file in the project root that is not a readable
  workspace is adopted by nobody and never allowed to take down the listing.
- Studio job records are returned in submission order. Creation timestamps
  have one-second resolution and the tie was broken by `job_id`, which is a
  digest: "retain the latest archive" kept whichever random hex sorted higher,
  so audit-quarantine retention pruned the wrong archive whenever the ids
  sorted against submission order. The tie is now broken by insertion order.
- Saving a Studio workspace can no longer destroy what was already saved.
  Three failures, each reproduced through the public API before it was
  replaced: two editors loaded the same workspace and the second save silently
  replaced the first; `delete_project` removed the file with no way back; and a
  save truncated the live file and wrote into it, so under a real
  `RLIMIT_FSIZE` refusal a 156-byte workspace became 65,536 bytes of
  unterminated JSON and loading it raised. A workspace is now an append-only
  list of immutable revisions (`studio.workspace.v1`). A save states the
  revision it was made from and a stale one is refused with HTTP 409 and the
  revision that is current; a delete moves the workspace, with its whole
  history, to a trash it can be restored from; and a revision is written to a
  sibling temporary file, `fsync`ed and `os.replace`d into place, so a failed
  write leaves the previous revision untouched. A workspace conflict is
  translated once, in the Studio API error boundary, because it is a
  `ValueError` and a route that forgot it would report "invalid input" for a
  save that was valid and merely arrived second.
- The Studio editor carries the revision it loaded and sends it with every
  save. It does not adopt the revision a conflict reports: saving again with
  the state it still holds is the lost update the conflict exists to prevent.
  A refused save says so and says nothing was overwritten, and a failed request
  now surfaces a structured error detail instead of `[object Object]`.

### NIR interoperability acceptance corpus
### Added
- Workspace history, forking and transfer: `GET /api/project/{name}/revisions`
  lists every revision with its parent and state digest,
  `GET /api/project/load/{name}?revision=N` reads one as it was written,
  `POST /api/project/{name}/fork` copies a revision into a new workspace
  without touching the source, and `GET /api/project/{name}/export` with
  `POST /api/project/import` moves a workspace between installations. Forking
  or importing onto a name in use is refused, not merged.
- Recoverable deletion: `GET /api/project/deleted` lists what is waiting with a
  restore token, `POST /api/project/restore` brings a workspace back under its
  original name with every revision it had, and restoring onto a live name is
  refused. The Studio projects panel shows the trash with a restore action, so
  recovery does not require an API call.
- `docs/studio/workspaces.md`: the on-disk layout, the save-against-a-revision
  contract, history, transfer, deletion and what happens to a workspace written
  by a newer schema.
- A Studio started without a configured job root no longer shares one fixed
  directory with every other Studio on the host. Since the job ledger became
  durable, that shared path meant a second process — or a later run — opened
  the first one's ledger and reported its jobs as its own, and on a multi-user
  machine the directory belonged to whichever user created it first.
  Durability is now a property of a configured root
  (`SC_NEUROCORE_STUDIO_JOB_ROOT`); without one, Studio creates a private
  directory for the process and says so through `configured: false`.
- Repaired three stale test expectations that were pinning behaviour the
  code no longer has: the code-export route now emits a script that
  re-resolves the exported request and guards its digest rather than building
  a model inline, a cached simulation response differs from the first one by
  its cache report, and the catalogue-scan stubs now carry the cancellation
  callback the real scan takes — a stub without it passed while the route it
  stood in for raised `TypeError`.
- Stopping a Studio job now stops the work, or says that it did not. A process
  worker's descendants survived its termination: only the direct child was
  signalled, so a job that spawned a subprocess left it running while the
  record read `timed_out`. Workers now lead their own process group, the group
  is signalled, escalated from SIGTERM to SIGKILL and then checked, and a group
  that survives is reported with the number of processes that may still be
  running. A thread task that never checks `context.cancelled` cannot be killed
  at all; instead of reporting only a timeout, the record states that the
  worker did not stop and that uncooperative work belongs in a process job, and
  the job appears in `unreaped_workers`.
- The catalogue scan consults its cancellation context between models. A
  cancelled sweep stops rather than running all 185 models and being discarded,
  and a partial sweep is never cached as a scan of the catalogue.
- The synchronous analysis budget projects a request's cost from the model it
  names — resolved timestep, integrator substeps and declared state count —
  instead of `ceil(duration / dt)` at a reference timestep. A four-state
  substepped conductance model was previously budgeted as a scalar map.
- Studio job records survive the process that made them. They lived in one
  in-memory dictionary, so a restarted API answered `404` for a job it had
  completed a second earlier, a second API process over the same job root saw
  none of the first one's work while its artifacts sat on disk, the same
  `request_id` submitted twice ran twice, and a job that was running when its
  process died had no terminal state and no way to acquire one. Job state now
  lives in a durable transactional ledger under the job root
  (`sc_neurocore.studio.platform.jobs_ledger`, `studio.job-ledger.v1`: SQLite
  in WAL mode, forward schema migrations, an append-only transition log
  enforced by triggers, and a state machine that refuses to rewrite a terminal
  record). Each job carries its actor and workspace, an idempotency key (a
  second submission with the same key returns the first job instead of running
  the work again), the effective experiment digest, the admission decision, a
  lease with expiry and heartbeat, and the terminal result and artifact
  manifest committed in one transaction with the status. Reads are scopeable by
  actor and workspace; a job outside the scope raises `KeyError` rather than
  reporting that it exists.
- Startup recovery resolves the jobs a departed supervisor left behind: one
  whose supervisor is provably gone, or whose lease expired without a
  heartbeat, becomes `interrupted`; one whose supervisor this host cannot probe
  becomes `unknown`, which is not terminal and awaits verification. Nothing is
  promoted to `completed` and no side effect is repeated. An artifact written
  before a crash stays on disk and out of the manifest the job never committed.
- Route policy counts and the operator status payload, which the
  `/api/export/replay-pack` route had left stale.
- Studio code export reproduces the experiment it came from. The generated
  script is built from the resolved `studio.experiment-spec.v1` and runs it
  through the public runner, instead of constructing the model with its default
  constructor, calling `step(current=…)` and reading `neuron.v`. Those
  assumptions were false for much of the catalogue: a model whose public step
  takes `drive` produced a script that raised `TypeError`, a requested time step
  the model did not receive ran a different length of model time while reporting
  the requested duration, and a non-constant drive protocol was silently
  exported as a constant one. The exported script now carries the experiment
  digest and stops rather than reporting a result when the installed package
  resolves a different experiment.

- The NIR graphs published with the NIR paper — written by Norse, Rockpool,
  Sinabs and snnTorch, and a SpiNNaker2 debugging example — are vendored
  unchanged from the NIR repository (BSD-3-Clause) with a manifest of their
  source commit, SHA-256, writer, node types and the Studio's answer. Every
  file is refused today, each with the node or parameter responsible, through
  the importer and through `/api/graph/import-nir`; the canvas guide lists
  them.
### Added
- `sc_neurocore.studio.platform.jobs_reaper`: process-group reaping with
  escalation and verification, reporting the outcome rather than raising.
- `sc_neurocore.studio.platform.jobs_admission`: a bounded number of running
  jobs with a bounded queue behind them; overflow is refused with
  `job_queue_full` and never reaches the ledger. The job status payload reports
  `admission` and `unreaped_workers`.
- `docs/studio/job-ledger.md`: the job ledger's migration, recovery, retention
  and backup contract, including how to read a job's transition history and how
  to resolve an `unknown` job by hand.
- `sc_neurocore.studio.replay_pack`: a sealed `studio.replay-pack.v1` document
  with the re-resolvable request (a drawn stochastic seed is pinned and the
  trial sealed as a replay), the public specification, an
  `experiment_identity_sha256` over the scientific blocks only, the complete
  expectation (every spike event, a digest per state trace with its endpoints
  and range, the initial and final state, the drive digest, the run statistics)
  and the sealing environment. `python -m sc_neurocore.studio.replay_pack
  <pack.json>` admits or refuses the pack before anything executes — unsupported
  schema, a document whose digest does not describe its own specification, an
  unexecutable request field, an experiment that resolves differently here, a
  drifted model revision, or unadmitted runtime drift — then runs it and
  compares in full, exiting 0, 1 or 2. Identity digests describe JSON values, so
  a pack saved by a browser still verifies.
- `POST /api/export/replay-pack` and the Studio's **Replay pack** export.
  `/api/codegen` now takes the same fail-closed request as the simulation
  routes, requires the `mode` discriminator, honours `protocol`,
  `frequency_hz`, `seed` and `trial`, and returns the experiment digest, the
  pinned request and a replay script alongside the script and one-liner.
- `docs/studio/replay-and-export.md`, and a browser-to-server-to-replay
  contract (`npm run test:e2e:export`) that drives the built Studio bundle,
  downloads a pack and replays it in a separate interpreter outside the
  repository.

### Network graphs refuse populations their model would not build
### Changed
- `studio.jobs.status.v1` → `studio.jobs.status.v2`: the job status snapshot
  gained `interrupted_count`, `unknown_count` and `recovery`. A v1 consumer
  would have counted an interrupted job as neither active nor failed, and seen
  no reason why. `StudioJobStatus` gained `interrupted` and `unknown`, and
  `StudioJobRecord` gained the custody fields the ledger stores.
- The jobs package keeps one responsibility per module: the ledger's schema,
  writes, reads, recovery and supervisor identity are separate owners, and the
  job manager's durable read surface moved to `jobs_manager_custody`.
- `vite preview` now proxies `/api` to the Studio backend, as the dev server
  does; the built bundle previously had no way to reach it.
- Firing-pattern classification moved from `studio.codegen` to
  `studio.firing_pattern`; code export and run description are separate
  responsibilities with separate owners.
- The Studio network canvas runs the graph it shows. A graph is resolved
  into a versioned specification (`sc_neurocore.studio.network_graph_spec`,
  `studio.network-graph-spec.v1`): every population names a catalogue model
  whose constructor contract accepts its parameters and the graph timestep,
  declares its external input (`none`, `constant`, `poisson`) and a positive
  integral count; every projection carries a signed weight that agrees with
  its source population's type, an explicit connection rule (`random` with a
  probability or `all_to_all`), a delay that is a whole number of timesteps
  and a seed that is given or derived from the graph seed; validation reports
  every error of a graph at once and unknown fields, sign conflicts, rule
  conflicts, fractional delays, inadmissible models (integer drive, shared
  seed field) and oversized runs are rejected, never clamped, rounded or
  shortened. The specification is lowered to public `Population`,
  `Projection`, `SpikeMonitor` and stimulus objects and run through the
  reference Python loop (`sc_neurocore.studio.network_execution`); the
  result (`studio.network-graph-result.v1`) reports the executed
  specification with its digest, the loop's semantics (previous-step
  propagation, delay buffering, one-step synaptic current), the rejected Rust
  runner with its reason, a topology artefact with the CSR digest and arrays
  of every projection, every spike event of every population, rates under a
  metric contract and a final finiteness check. The former mapping of any
  two-population graph onto the E-I template (model, parameters and delays
  ignored, `abs(weight)`, one probability from the last edge, template
  defaults for absent edges) is gone; the E-I template remains available as
  `/api/network/ei`. The canvas default model is `SCLapicqueLIFNeuron`
  (`LIFNeuron` was never a catalogue model); new excitatory populations
  carry a constant drive, new projections a weight signed by their source
  type; NIR-named import requires catalogue model names and validates the
  assembled graph. `/api/graph/models` lists only admissible models.
- Studio analyses now state what they measure and where they are valid.
  Every analysis payload carries a metric contract
  (`sc_neurocore.studio.analysis_contract`, `studio.metric-contract.v1`:
  kind, definition, units, applicability, limitations and a
  complete/partial/empty domain verdict) and the analysis manifest records
  its kind and domain. The precision comparison (`/api/precision`,
  `/api/ir/cosim`) runs genuine fixed-point arithmetic: the generated
  bit-true C kernel of the equation system, proven bit-identical to the RTL,
  is compiled with the host C compiler and driven with one encoded input
  word per step (`sc_neurocore.studio.bit_true_execution`); the kernel
  generator states its value encoding, multiply width collapse and rounding,
  accumulate overflow policy, LUT geometry and threshold/reset sequencing
  (`kernel_arithmetic_contract`). The result keeps the float64 reference,
  the bit-true run (decoded words, every state and spike) and a separately
  labelled float64 run with parameters and initial state rounded to the word
  resolution apart, compares every variable and the spike train, aligns the
  error trace with the display projection, and reports saturation; the word
  format, overflow and rounding modes, protocol and sine frequency are
  request fields. A parameter, expression constant, initial value, time step
  or drive sample the word cannot hold is rejected with its field instead of
  being clamped, and a missing C compiler is answered with HTTP 503 instead
  of an estimate. Nullclines (`sc_neurocore.studio.nullclines`) evaluate each
  drift-field component with floating-point errors raised and return
  per-component validity masks, contour cells only where all four corners
  are valid, the held variables and the input current; an invalid sample is
  never a zero derivative. Sensitivity reports an undefined elasticity
  (zero base rate or zero parameter) as `null` with a reason instead of
  `0.0`; the bifurcation route is labelled a numerical extrema sweep with
  the analysed variable and protocol; f-I, heatmap and frequency response
  state their rate definition. The Studio UI sends the explicit protocol and
  word format, draws the bit-true and parameter-quantisation errors on the
  display axis, labels a partial nullcline domain and shows undefined
  sensitivities with their reason.
- Every Studio simulation request now resolves into one effective experiment
  (`sc_neurocore.studio.experiment_spec`, `studio.experiment-spec.v1`) before
  it runs: model revision (package, module, descriptor and canonical schema
  digests) or equation digest, numerical profile and effective time step,
  exact step count, typed initial state, protocol with an explicit sine
  frequency and the drive digest, randomness contract (kind, seed and its
  source, replay or fresh trial), backend selection with rejected
  alternatives and the runtime digest. The result cache is keyed by that
  digest, so runs that differ in any of them cannot share an entry, and a
  fresh stochastic trial is never cached; explicit request defaults resolve
  to the same experiment as omitted ones. A seed on a deterministic model or
  on noise-free equations is rejected; playground diffusion noise draws from
  a per-run generator instead of the process-global stream; an oversized
  synchronous run is refused with a pointer to the new `simulate` job kind
  instead of being shortened; a protocol typo on the playground route is
  rejected instead of becoming constant. Run manifests carry the experiment
  digest and trial; the Studio UI sends explicit frequency, seed and trial
  and persists them in saved sessions.
- Studio simulation results now carry complete-state and raw-result custody.
  Catalogue runs record the state the committed descriptor declares, joined
  with the canonical profile roles (`sc_neurocore.studio.state_layout`), with
  observed scalar/vector shapes, exact initial and final snapshots, a
  post-step observation clock and a start-to-end mutation audit; name
  heuristics are no longer a state source (they recorded AdEx parameters as
  state and dropped the adaptive threshold). Every response returns the
  full-resolution `raw` block next to a bounded display projection whose
  `display.sample_index` maps each point to its raw step and preserves every
  bucket extremum, the first and the last sample
  (`sc_neurocore.studio.trace_projection`). Run manifests move to
  `studio.simulation-run.v2` (raw digest, layout source, custody verdict,
  observation clock); evidence bundles accept v1 and v2. The catalogue and
  multi-simulate routes run on the Python custody backend; the Rust batch
  result is labelled voltage-only without an initial snapshot. Analyses,
  the spike-triggered average and CSV export read the raw traces.
- Added the versioned model profile contract `sc-neurocore.model-profile.v1`
  (`sc_neurocore.neurons.model_profile`): every schema-DSL document resolves
  into a scientific model (authored equations, biological state and source
  parameters with units), a numerical realisation (method and exactness class,
  step and time unit, sub-steps and whether they subdivide time or fold the
  stages of one scheme, macro step, evaluation order, auxiliary registers,
  implementation and timebase parameters, admissible methods, randomness
  contract) and a lowering profile (what the RTL emitter accepts and under
  which limits). An optional `[profile]` section authors what cannot be
  derived; twelve representative schemas (`lapicque`, `sc_lapicque_lif`,
  `lif`, `adex`, `hodgkin_huxley`, `wang_buzsaki`, `exp_if`, `rulkov_map`,
  `chialvo_map`, `escape_rate`, `poisson`, `coba_lif`) carry one in both
  formats. No equation, parameter, step, threshold or reset value changed.
  Exactness claims are checked: `exact-linear-relaxation` for exponential
  Euler is verified symbolically, `exact-flow` is admitted only for an
  authored recurrence, and an explicit-Euler or RK4 schema cannot claim either.
- `UniversalNeuron` refuses a schema whose profile contradicts it or that is a
  descriptive record outside the executable vocabulary, and admits overrides
  through the profile: a method must stay in the family (a published map is
  never integrated as an ODE; an ODE may select another ODE integrator and the
  realisation is then reported as derived), a step override moves the
  timebase parameters (`dt`, `dt_ms`) with it and is refused for a recurrence
  without a continuous timebase, a parameter override that contradicts the
  effective step is refused, and a seed is refused for a deterministic model.
  `neuron.profile` and `neuron.realised_profile()` expose both views.
- The static schema validator reports profile contradictions as errors,
  classifies the thirteen descriptive records (methods or detections outside
  the executable vocabulary) as warnings instead of passing them silently, and
  checks the profile section for TOML/JSON parity.
- Studio model details carry `profile_contract`; the compile configuration
  offers only integrators inside the profile's family and refuses one that
  leaves it; compile evidence records the realised profile (declared and
  effective method, exactness, step, sub-steps, macro step, randomness,
  derived flag) next to the schema digest. The descriptor skeleton generator
  resolves the curated schema per class instead of per module.
- Added the per-profile inventory and validator registry
  (`sc_neurocore.neurons.profile_registry`, `tools/model_profile_ledger.py`,
  generated `docs/_generated/model_profile_ledger.json`): one row per bound
  (class, schema profile) with the three layers, the descriptor's own
  integration label next to the profile method, readiness verified for exactly
  that profile, every declared validator per facet with its resolution and
  scope, and an admission verdict. A non-canonical profile is never admitted
  by the canonical profile's class-scoped tests; a facet declared by prose or
  a report alone blocks admission; declared-but-unvalidated backends are
  listed, not decided. Documented in `docs/api/model_profiles.md`.
- Four descriptors (`AiharaMapNeuron`, `NagumoSatoMapNeuron`,
  `SCAdaptiveThresholdMapNeuron`, `SCChaoticMapNeuron`) stated
  `method = "euler"`, `dt = 0.1` because the generator resolved their schema by
  module name and found none; regenerated per class they state the bound map
  profile (`map`, `dt = 1.0`). The `ermentrout_kopell_theta_euler_doi`
  reference protocol declared an inert `dt = 1.0` for a map whose 0.1 Euler
  step is fixed inside its expression; it now declares 0.1 with unchanged
  expected features. The eight representative facet receipts were re-recorded
  against the changed schema and compiler-front-end subjects.
- Added evidence-bound readiness next to the declared dual-axis tiers
  (`sc_neurocore.neurons.readiness`, `facet_receipts`, `evidence_references`).
  Descriptor evidence fields are parsed into typed references and resolved on
  disk; a facet is credited only through an immutable facet receipt
  (`tools/facet_receipt.py`) that records the executed command, its tool and
  runtime, its passed/failed/skipped counts and the digests of every subject
  (descriptor contract, model module, schema profiles, compiler, committed RTL,
  reports, validator tests), so one changed subject invalidates exactly the
  facets that consume it. `tools/readiness_evidence_ledger.py` writes and
  checks `docs/_generated/readiness_evidence_ledger.json`; Studio catalogue
  entries carry verified tiers next to the declared ones; see
  `docs/api/readiness_evidence.md`.
- Corrected four descriptor evidence pointers that named test functions that
  never existed under that name (AdaptiveThresholdIF, Brunel-Wang and
  resonate-and-fire co-simulation; McCulloch-Pitts validation) and withdrew the
  co-simulation claim of `SCInclusivePerfectIntegratorNeuron`, whose retained
  schema has no RTL trace test. The corpus now fails on any evidence reference
  that does not resolve. Eight receipts recorded from executed tests bind
  Lapicque (S5, H1), the three corrected co-simulations and the McCulloch-Pitts
  validation.
- Added the canonical model identity registry
  (`sc_neurocore.neurons.model_identity`): every registered class and import
  alias resolves to exactly one identity record with its kind (published source,
  project-original design, `SC` compatibility identity or API alias), count
  membership, taxonomy family, bound schema-DSL profiles, source locator, public
  fidelity status, revalidation status and open evidence gates. Ambiguous or
  missing joins raise instead of being dropped: an `SC` identity with a
  literature DOI, a schema stem owned by no class or by two, or a public row
  naming an unregistered class. `tools/model_identity_ledger.py` writes and
  checks the generated `docs/_generated/model_identity_ledger.json`.
- Corrected the public catalogue figures to 57 of 160. The page sentence said
  sixty while its own strict table carried 57 source rows, and the 155
  denominator was a running hand tally with no live definition; the registry
  derives the source catalogue as the 185 registered classes minus the 25
  count-neutral `SC` compatibility identities (import aliases are never
  registered classes), so 103 source-model units remain. The count-neutral SC
  three-state phantom is listed as a compatibility identity awaiting its own
  source-hashed five-runtime benchmark instead of being strict-promoted on the
  distinct Bertram artefact. Of the 57 promoted source identities, 20 are bound
  to an independent source receipt and 37 are explicitly recorded as not
  revalidated.
- Classified the three retained `SC` compatibility identities
  (`SCInclusivePerfectIntegratorNeuron`, `SCLapicqueLIFNeuron`,
  `SCSymmetricQuadraticIFNeuron`) in the family taxonomy, so the taxonomy
  completeness test passes again, and bound the seven schema-DSL stems whose
  module names differ from their stem (`aihara_map`, `nagumo_sato_map` and the
  five `sc_*` paired profiles) in the schema alias table. The schema-gap report binds stems
  through the same table, so paired `sc_*` profiles are no longer reported as
  orphan schemas and its net-missing count is modules without any schema.
- Studio model runs (`simulate_model`, `POST /api/models/simulate`,
  `POST /api/export/svg`) are fail-closed. A new run contract validates every
  request against the model class: unknown, non-numeric, non-finite, private,
  derived or non-numeric-typed parameter overrides, fractional values for
  integer fields, foreign `dt` on fixed-step or timestep-less models,
  unsupported protocols, fractional samples for integer-drive models, and
  `step` inputs the protocol cannot supply are rejected with the field and
  reason (HTTP 422 `invalid_model_input`). Constructor failures, `step`
  exceptions, non-finite or non-scalar state are reported with backend, step
  and simulated time (HTTP 422 `model_simulation_failed`) instead of default
  reconstruction, retried steps, or zero substitution; failed requests are not
  cached. Successful runs carry a `studio.model-run-inputs.v1` receipt with
  the effective parameters, timestep source, drive contract and every excluded
  state variable. `IntegerQIFNeuron` now reports its fixed 1.0 ms step in the
  catalogue and on both backends. The request schema forbids unknown keys and
  non-finite numbers, and a body carrying `NaN` or `Infinity` returns 422
  instead of 500. The OpenAPI reference declares the 422 contract.
- Removed the pre-existing integer round-trip from the Go alpha kernel's
  `unsafe.Pointer` region assembly. The direct pointer slice preserves the C
  ABI while restoring clean `go vet ./...` and module-test validation.
- Re-closed `DPINeuron` against Indiveri, Stefanini, and Chicca's 2010
  current-mode equations (2)-(3), separating the source circuit terms from
  maintained normalised defaults and discrete event ordering. Python,
  production and safety Rust, PyO3, Julia, Go, and Mojo now expose
  failure-atomic complete membrane/adaptation/refractory/event packets for the
  configurable 18-field contract. A PDF-bound independent complete-state
  receipt, source-hashed five-runtime benchmark, paired schemas, NetworkRunner
  execution, 112,953-cell Q16.16 Yosys synthesis, and depth-8 reached-event
  reset/refractory safety establish H2 without timing, PPA, device, board,
  physical-silicon, transistor-mismatch, or universal-equivalence claims.
- Re-closed `ThetaNeuron` against Ermentrout and Kopell's equation (2.5), with
  `current` explicitly representing source parameter `a` or a frozen equation
  (3.3) drive rather than the paper's full slow-oscillator bursting system.
  Python, production and safety Rust, PyO3, Julia, Go, and Mojo now expose
  failure-atomic complete phase/event packets for arbitrary maintained state;
  over-wide held-drive steps fail instead of collapsing multiple passages into
  one event bit. A PDF-bound independent receipt, source-hashed five-runtime
  benchmark, paired exact-software/Euler-RTL contracts, 6,203-cell Q16.16 Yosys
  receipt, and depth-110 receipt-drive phase/event safety establish the honest
  H2 boundary without
  timing, PPA, board, physical-silicon, or exact-flow hardware-equivalence
  claims.
- Corrected `QuadraticIFNeuron` to Latham et al.'s normalized source numerical
  boundaries (`v=-1`, apex `31/3`, reset `-3`, `dt=.05`) while retaining the
  historical `-1/+1/.01` recurrence as count-neutral
  `SCSymmetricQuadraticIFNeuron`. Five failure-atomic complete runtimes, paired
  schemas, a DOI/PDF-bound receipt, NetworkRunner routing, source-hashed
  benchmark, dedicated Q16.16 source RTL, 9,379-cell Yosys synthesis, and
  depth-20 reset/event safety now establish the honest H2 envelope without
  timing, PPA, device, board, physical-silicon, or universal-equivalence claims.
- Re-closed `ExpIFNeuron` against the Fourcaud-Trocmé et al. (2003) fitted
  exponential integrate-and-fire protocol. The catalogue-counted source
  factory now keeps the paper's -30 mV numerical handoff, deterministic
  zero-noise Heun RK2 specialization, derived analytical exponential-only
  tail duration, and 1.7 ms refractory interval distinct from the retained historical SC
  candidate-first RK4/+30 mV/Q32.32 compatibility profile. Python,
  production and safety Rust, Julia, Go, Mojo, PyO3, NetworkRunner, paired
  schemas, an independent complete-packet receipt, source-hashed benchmark,
  fixed-point co-simulation, 484,938-cell Yosys synthesis, and bounded
  reset/event safety now establish the honest H2 boundary without timing, PPA, device, board,
  physical-silicon, or universal-equivalence claims.
- Re-closed `AdExNeuron` against the Brette–Gerstner equations and separated
  the paper's regular-spiking fit from backwards-compatible maintained
  defaults. Python, production and safety Rust, Julia, Go, and Mojo now expose
  failure-atomic complete `(v,w,event)` batches with full parameter transport;
  paired schemas and an independent source-fit receipt bind the published
  equations. The source-hashed five-runtime benchmark retains complete packet
  custody, while four-current Q16.16 co-simulation, a 52,014-cell Yosys receipt,
  and depth-6 reset/event safety establish the honest H2 boundary without
  timing, PPA, device, physical-silicon, or formal-equivalence claims.
- Corrected `WilsonHRNeuron` to Wilson's continuous 1999 polynomial cortical
  equations with source `C=0.8`, `tau_R=1.9 ms`, no hard reset, and an explicit
  sampled upward-crossing observation convention. The former unit-capacitance,
  level-detected reset recurrence remains available as the count-neutral
  `SCResettingWilsonHRNeuron`. Both identities now carry failure-atomic
  Python/Rust/Julia/Go/Mojo execution, production PyO3 and NetworkRunner wiring,
  paired schemas, source-bound benchmarks, generated Q16.16 co-simulation,
  Yosys synthesis, bounded formal safety, and language-native documentation.
  The source catalogue count remains 55/155.
- Re-closed `TermanWangOscillator` without changing its DOI-backed two-state
  cubic/`tanh` equations or continuous no-reset identity. Python, production
  Rust/PyO3, Julia, Go, and Mojo batches now reject invalid stages, candidates,
  and results without partial state mutation; the five-runtime benchmark is
  source-hash-bound and fail-closed; descriptor evidence resolves to current
  tests; and committed Q16.16 co-simulation, real Yosys synthesis, and depth-4
  bounded safety establish the honest H2 boundary without timing, PPA, device,
  silicon, or formal-equivalence claims.
- Re-closed `PernarowskiNeuron` without changing its DOI-backed three-state
  equations or continuous no-reset identity. Python, production Rust/PyO3,
  Julia, Go, and Mojo batches now reject invalid stages and candidates without
  partial state mutation; the five-runtime benchmark is source-hash-bound and
  fail-closed; descriptor evidence resolves to current tests; and committed
  Q16.16 co-simulation, real Yosys synthesis, and depth-4 bounded safety
  establish the honest H2 boundary without timing, PPA, device, or silicon
  claims.
- Re-closed `FitzHughRinzelNeuron` against Rinzel's 1987 equations (3.4)–(3.6)
  and parameter profile while keeping fixed-step RK4 and sampled no-reset
  threshold crossings explicit as repository specialisations. Python,
  production Rust/PyO3, safety Rust, Julia, Go, and Mojo now reject non-finite
  batch candidates without partial state mutation; the five-runtime benchmark
  is source-hash-bound and fail-closed; descriptor evidence is executable; and
  committed Q16.16 co-simulation, Yosys synthesis, and depth-4 bounded safety
  establish the honest H2 boundary without timing, PPA, device, or physical
  silicon claims.
- Corrected `BendaHerzNeuron` to Benda and Herz's deterministic universal
  adaptation equations (8) and phase spike generator (45), using the paper's
  Figure 8 square-root onset and linear steady-state adaptation example. The
  former logistic RK4/exponential-hazard recurrence remains exactly available
  as count-neutral `SCStochasticRateAdaptationNeuron` without paper
  attribution. Both identities carry Python/Rust/Julia/Go/Mojo surfaces,
  paired schemas, independent receipts, RTL/Yosys/bounded formal evidence,
  focused benchmarks, NetworkRunner wiring, and native documentation. Only the
  source identity increments the catalogue, now 55/155.
- Corrected `McKeanNeuron` to the source-bound space-clamped Heaviside system
  (`dv/dt=-lambda*v+mu*H(v-a)-w+I`, `dw/dt=b*v`) and pinned the
  right-continuous `H(0)=1`, simultaneous RK4, sampled upward switching-line
  event, and no-reset conventions. The former three-branch recurrence remains
  exactly available as count-neutral `SCTriangularMcKeanNeuron`. Both identities
  now carry Python/Rust/Julia/Go/Mojo execution, paired schemas, independent
  receipts, signed-Q32.32 RTL co-simulation, Yosys and bounded formal evidence,
  source/binary-bound benchmarks, NetworkRunner wiring, and language-native
  documentation. Only the source identity increments the catalogue, now
  54/155; spatial diffusion and network behavior remain outside this scalar unit.
- Corrected `EnergyLIFNeuron` to the Fardet-Levina (2020) coupled eLIF source
  identity and the authors' 0.1 ms Brian RK4 profile. The former normalized
  exact-flow recurrence remains exactly available as the count-neutral
  `SCNormalizedEnergyLIFNeuron`. Both identities now carry complete
  Python/Rust/Julia/Go/Mojo dispatch, paired schemas, independent receipts,
  pinned Q32.32 RTL/Yosys/bounded-safety evidence, source/binary-bound
  benchmarks, and language-native documentation. Only the source identity
  increments the catalogue, now 53/155.
- Corrected `SigmaDeltaNeuron` to a source-bound sampled specialization of
  Yoon's asynchronous pulse sigma-delta encoder: integrating prefilter,
  exponentially decaying local reconstruction, upper-threshold quantizer, and
  unipolar feedback events. The former signed accumulator remains exactly
  available as `SCSigmaDeltaAccumulatorNeuron`. Both identities now carry
  Python/Rust/Julia/Go/Mojo execution, paired schemas, independent receipts,
  Q32.32 RTL/Yosys/CVC5 evidence, source/binary-bound benchmarks, and native
  documentation. Only the source identity increments the catalogue to 52/155.
- Corrected `NonResettingLIFNeuron` to the source Kobayashi-Tsubo-Shinomoto
  (2009) one-timescale MAT(1) identity: forward-Euler non-resetting membrane,
  exact 50 ms threshold history, and 2 ms absolute refractory gating. The
  frozen affine-rest exact-relaxation behavior remains as the explicitly
  project-defined `SCNonResettingAdaptiveLIFNeuron`. Both identities now carry
  Python/Rust/Julia/Go/Mojo execution, paired schemas, independent receipts,
  signed Q32.32 RTL/Yosys/sequence/formal evidence, source/binary-bound local
  benchmarks, and native-language documentation. Only MAT(1) increments the
  public source-model fidelity count, now 51/155; timing, PPA, device, and
  universal-equivalence claims remain open.
- Restored `MATNeuron` to the source Kobayashi-Tsubo-Shinomoto (2009) MAT*
  identity: non-resetting voltage, paper-profile parameters, forward-Euler
  membrane at 0.001 ms, exact two-timescale threshold memories, and a 2 ms
  refractory interval. The former candidate-first RK4/reset recurrence remains
  available as the explicitly project-defined `SCResettingMATNeuron`. Both
  identities now carry complete Python/Rust/Julia/Go/Mojo execution, paired
  schemas, independent trace receipts, signed Q32.32 RTL co-simulation, Yosys
  synthesis, sequence-bounded optimized-netlist equivalence, depth-12 CVC5 safety,
  source/binary-bound local benchmarks, and native-language documentation. Only
  the source MAT* identity increments the public fidelity count, now 50/155;
  timing, PPA, device, and universal equivalence claims remain open.
- Added the schema-bound `SC-COMPTE-WM-NETWORK` 16-bin circular E-to-E
  no-autapse RTL representative. Its unsigned Q16.16/Q32.32 contract has
  zero-LSB independent dense-oracle co-simulation across all targets,
  busy-time input rejection, 16-cycle bounded control proof, Yosys synthesis,
  and asserted post-`opt` equivalence. The committed receipt explicitly makes
  no full-network binary64, device timing/PPA, board, or physical-silicon
  claim.
- Added identical JSON/TOML contracts for the separately named
  `SC-COMPTE-WM-NETWORK`, covering topology, numerics, all nine state arrays,
  behavior protocol, receipts, and an explicit bounded-representative hardware
  boundary without conflating the preserved scalar Compte cell.
- Established the separately named 2,560-cell `SC-COMPTE-WM-NETWORK` behavior
  ensemble across Python, Rust, Julia, Go, and Mojo. Three reference seeds and
  five explicit runtime anchors now satisfy predeclared circular-statistics
  thresholds for spontaneous baseline, cue formation, persistent bump,
  bounded bidirectional drift, distractor resistance, global response, and
  reset, with exact long-run input/spike/count custody. Direct-current receipt
  hashing now uses documented `1e-9 pA` custody resolution across platform
  math libraries without changing executed binary64 currents. This is
  deterministic SC simulator evidence, not reproduction of the paper's larger
  distractor experiment or silicon/biological validation.
- Corrected `CompteWMNeuron` to the source-bounded Compte et al. (2000)
  pyramidal-cell membrane and incoming AMPA/NMDA/GABAA kinetics, including
  separate presynaptic event paths, coupled midpoint RK2 at 0.02 ms, and
  atomic refractory/reset semantics. Python, modular Rust, independent Rust
  safety, Julia, Go, and Mojo preserve the complete six-state/event receipt;
  an independent primary-equation trace, paired schemas, hand Q16.16 RTL
  co-simulation, Yosys and bounded formal collateral, a source/binary-bound
  200,000-step benchmark, and native-language documentation close together.
  The existing convenience `working_memory_circuit` is now explicitly an SC
  project-derived approximation. The paper's 2,560-cell ring, connectivity
  footprint, Poisson drive, persistent bump, distractor resistance, and
  network statistics are retained for a separately named SC network model.
- Corrected `BrunelWangNeuron` to the Brunel-Wang (2001) excitatory
  pyramidal-cell defaults and midpoint-RK2 membrane equation at 0.1 ms. Python,
  modular Rust, independent Rust safety, Julia, Go, and Mojo now preserve the
  complete configured four-gate voltage/refractory/event receipt; an independent
  primary-equation oracle, paired schemas, hand Q16.16 RTL co-simulation, Yosys
  and bounded formal collateral, a source/binary-bound 200,000-step benchmark,
  and language-native documentation define the evidence boundary. The distinct
  `BrunelNetwork` is unchanged, and network connectivity, Poisson input,
  synaptic-state kinetics, persistent activity, timing, PPA, device results,
  and binary64 formal equivalence remain unclaimed.
- Corrected `AmariNeuralField` to the source-level Heaviside activity in Amari
  (1977), restored a genuinely local-excitation/distal-inhibition default
  kernel, and promoted the finite periodic-grid specialization across the
  modular Rust engine, independent Rust safety, Julia, Go, and Mojo lanes.
  Complete vector-state receipts, an independent DOI oracle, paired schemas,
  Q16.16 four-site RTL co-simulation, Yosys/depth-12 safety induction, a
  source/binary-bound benchmark, and language-native API documentation now
  define the evidence boundary. Continuous-space convergence, persistent-bump
  defaults, formal equivalence, timing, PPA, and device results remain unclaimed.
- Graduated `SCChaoticMapNeuron` as its own project-specified high-fidelity
  model without restoring the removed Aihara attribution. The preserved
  Python/Rust/Julia/Go/Mojo recurrence is now bound to an independent project
  receipt, paired TOML/JSON schemas, measured behavior taxonomy, Q8.24
  schema-to-RTL co-simulation, Yosys/depth-12 safety evidence, a source- and
  binary-bound 200,000-step benchmark, and native-language API documentation.
- Split the misidentified `KilincBhattMapNeuron` into the source-faithful
  one-state `NagumoSatoMapNeuron` and the retained project-designed two-state
  `SCAdaptiveThresholdMapNeuron`; the historical name is now a deprecated
  compatibility alias only. Both canonical models carry complete
  Python/Rust/Julia/Go/Mojo receipts, independent equation/specification
  traces, paired schemas, fixed-point RTL co-simulation, Yosys/SymbiYosys
  evidence, source-bound benchmarks, and native-language API documentation.
- Adopted language-native API documentation as a project-wide forward ratchet:
  Python docstrings, Rustdoc, GoDoc, Julia docstrings, Mojo ABI/module comments,
  and RTL interface/fixed-point contracts accompany new or materially changed
  public APIs. Rust documentation builds now execute in the engine workflow;
  historical gaps remain explicit backfill work rather than implied compliance.
- Split two previously conflated model identities: `AiharaMapNeuron` now uses
  Aihara's primary-author one-state recurrence and Eq. 12 level waveform
  shaper, while the original SC two-state recurrence remains available as the
  explicitly project-designed `SCChaoticMapNeuron`. The SC model retains its
  two-state recurrence with measured Python/Rust/Julia/Go/Mojo trajectory and
  event parity. Separately, all Aihara lanes now share the source Figure 4/5
  parameter anchors, checked batch receipts, paired schemas, an independent
  primary-equation trace, dedicated documentation, Q8.24 bounded co-simulation,
  a depth-6 Z3 proof, and a source-periodic benchmark.
- Raised the Python coverage gate from 98% through the intermediate 99% ratchet
  to 100% in `pyproject.toml`, the primary CI matrix leg, and local preflight.
  The retained aggregate at tip `2e6118ba5` is 65,373/65,373 statements; the
  Makefile and hosted gate now reject any regression below that closure.
- Regenerated polyglot benchmark evidence for alpha, adaptive-threshold-IF,
  Ermentrout–Kopell population, and resonate-and-fire after the coverage-gate
  hash surface changed; refreshed `docs/API_REFERENCE.md`.
- Isolation throughput floors for Hindmarsh–Rose and NLIF now use the CI floor
  under active coverage tracing (as well as under `CI=`), so full-suite
  preflight with pytest-cov does not flake while uninstrumented local floors
  stay strict.
- Mojo ISA pin allowlist includes the split
  `test_build_accel_backends_commands_and_build` suite; DPI checkout bridge
  discovery appends the user site-packages root under `python -S`.

- Graph validation checked each population parameter against its own range
  but never built the model, so values each in range that no model accepts —
  a threshold below the resting potential, or a model whose timestep range
  excludes the graph's — passed validation and then failed while the
  network was being built, reported only as invalid input. Validation now
  builds one neuron with the validated keywords and reports the model's own
  refusal on the population's parameters.
- The population contract of the canvas's default model offered three fields
  it inherits but whose constructor does not take them (`capacitance`,
  `polarization_resistance`, `series_resistance`); they are no longer
  offered, and the contract says why.
- A NIR file with such a population — a self-driven LIF whose leak potential
  lies above its threshold — is now refused with that reason instead of
  failing while it runs.
### Added
- Added explicit fail-closed public dispatch for all five complete
  `SC-COMPTE-WM-NETWORK` runtimes. Python and Mojo execute in process;
  repository-native Rust, Julia, and Go JSON adapters expose the same run and
  window receipt contract without silent fallback. Rust now natively owns
  protocol stimuli, activity windows/statistics, and SHA-256 input/spike/state
  custody. A source/binary-bound three-repeat 1,000-step consolidated benchmark
  records exact Python/Rust/Julia/Go/Mojo input, spike, and count parity while
  keeping timings local and behavior/hardware claims false.
- Added the complete Mojo runtime for `SC-COMPTE-WM-NETWORK`, with a
  documented stable C ABI, all 2,560 states, native counter-Poisson input,
  dependency-free radix-2 FFT coupling, control/modulated modes, explicit-
  event co-simulation, protocol/window/run custody, and atomic validation.
  The repository-local Mojo 0.26.2 build is pinned to `x86-64-v3`; its
  source/binary-bound three-repeat 1,000-step receipt exactly matches the
  Python/Rust/Julia/Go input and spike evidence without claiming network
  behavior or hardware performance.
- Added the complete Go runtime for `SC-COMPTE-WM-NETWORK`, with GoDoc,
  all 2,560 states, counter Poisson input, an in-tree dependency-free radix-2
  FFT, optional structured E-to-I coupling, explicit-event co-simulation,
  stimuli/statistics/run receipts, and atomic validation. Native tests bind
  the counter, preserved scalar cell, recurrent fixture, determinism, and
  failure boundary; its source-bound three-repeat 1,000-step receipt exactly
  matches Python/Julia input and spike evidence without claiming behavior.
- Added the complete documented Julia runtime for `SC-COMPTE-WM-NETWORK`,
  including all 2,560 cells, repository-local pinned FFTW dependencies,
  counter Poisson drive, structured FFT recurrence, explicit-event
  co-simulation, stimuli, bounded activity receipts, and atomic validation.
  Native tests bind the counter, preserved scalar cell, and recurrent fixture
  to Python/Rust evidence; a source-bound three-repeat 1,000-step benchmark is
  local regression evidence only and does not claim bump persistence or
  distractor resistance.
- Added the documented modular Rust runtime for `SC-COMPTE-WM-NETWORK`, with
  the full fixed population state, preplanned FFT ring coupling,
  counter-addressed Poisson input, explicit-event co-simulation boundary,
  control/modulated modes, and atomic validation. Native tests bind its input
  fixture and isolated cell to Python/original-cell evidence and bind a
  non-trivial recurrent step to the Python dense oracle. A separately compiled
  dependency-free Rust safety surface validates complete state, input, and
  dense no-autapse connectivity. This is Python/Rust short-trace parity, not
  yet behavioral or all-runtime closure.
- Added the first executable `SC-COMPTE-WM-NETWORK` layer without conflating it
  with the preserved scalar `CompteWMNeuron`: deterministic counter-addressed
  1,800 Hz Poisson input receipts, all 2,048+512 membrane/channel states,
  midpoint RK2, no-autapse circular E-to-E and optional structured E-to-I
  coupling, protocol currents, atomic validation, window statistics, and
  canonical input/spike/state digests. Focused tests bind an isolated
  pyramidal impulse to the original cell model and independently check the FFT
  reduction against a dense target oracle. The source-bound 1,000-step Python
  benchmark is local regression evidence only; behavioral ensembles and
  native-network parity remain open and the neuron-model count is unchanged.
- Established `SC-COMPTE-WM-NETWORK` as the explicit SC project-model identity
  for the retained 2,560-cell working-memory ring. The public
  `SCCompteWMNetworkSpec` freezes the 2,048+512 population split, source-derived
  control/modulated conductances and synaptic constants, exactly normalized
  circular footprints, reproducible compact cue contract, circular distractor
  displacement, and population activity statistics. This specification does
  not yet claim persistent-bump behavior, distractor resistance, runtime
  parity, or another neuron-model completion.
- Exact MPS / Heisenberg TEBD contracts for `SpinPoolMPS` and topology +
  Kane-exchange contracts for `KaneSiliconMapper` (worst-offender coverage
  debt closure for quantum_cognition + ONNX protobuf path via installed
  `onnx`).
- Reconstructed `AdaptiveThresholdIFNeuron` as an explicitly composite
  reduced adaptive-threshold leaky integrate-and-fire: exact constant-input
  LIF membrane relaxation, the Mihalas–Niebur (2009) threshold equation at
  zero voltage coupling, and the Platkiewicz–Brette (2010) fixed post-spike
  threshold shift; the voltage-dependent threshold equilibrium, voltage
  coupling, and adaptation current are recorded as the reduction boundary,
  with defaults documented as catalogue/model-family choices.
- Added configured atomic batch execution through Python, the modular Rust
  engine, independent Rust safety, Julia, Go C-shared, and Mojo shared-library
  lanes; an independently derived composite-DOI reference trace; paired
  TOML/JSON exact-map schemas; Q32.32 generated-RTL co-simulation at the
  enrolled grid-exact operating point with the complete event vector
  identical; a depth-4 Z3 reset-safety job; and a source/binary-hashed
  five-runtime 200,000-step benchmark.
- Reconstructed `AlphaNeuron` as the dual excitatory/inhibitory alpha-synapse
  leaky integrate-and-fire on the Rall (1967) alpha kernel and the
  Gerstner-Kistler (2002) LIF framing. Every maintained runtime now executes
  the same exact piecewise-constant-input filter relaxation and alpha-current
  convolution, with somatic-only reset and atomic failure semantics.
- Added the modular Rust/PyO3, Rust-safety, Julia, Go and Mojo execution lanes;
  paired schemas; an independent dual-drive reference trace; Q32.32 generated
  RTL co-simulation with the complete event vector preserved at both enrolled
  inhibitory levels; a depth-4 reset-safety proof; and source/binary-bound
  five-runtime benchmark evidence.

### Network tutorial notebooks from the canvas
### Fixed
- Made the native-loading regression exercise both clean editable
  `maturin develop` layouts and installed-wheel layouts instead of requiring
  a stale site-packages copy of the engine package.
- Restricted the PyPI download metrics client to the exact HTTPS pypistats API
  origin and path, encoded package names as single URL path segments, and added
  fail-closed tests for untrusted schemes, hosts, credentials, ports, and paths.
- Corrected the `AdaptiveThresholdIFNeuron` "Platkiewicz & Bhatt 2010"
  typographical attribution and the descriptor `integration.method = "euler"`
  mismatch (production always used the exact constant-input relaxation);
  replaced the Rust engine's forward-Euler step with the maintained exact
  relaxation so every lane executes the same contract.
- Reconciled schema validation with the runtime's exact state-free level-
  threshold and Poisson contracts through one shared semantic predicate. Added
  the missing JSON peers for both rate-model schemas and refreshed the live
  schema-gap baseline to 38 unique schema stems.
- Made descriptor generation follow canonical schema/module aliases, preserve
  event-only integration semantics, and materialise public RNG replay state.
  The nine affected descriptors are regenerated byte-for-byte from that source
  contract, including the previously omitted resonate-and-fire equations.
- Generalised the Tier-3 reproducibility gate for strict integer inputs,
  constructor-declared reference parameters, tuple and array simulation returns,
  and explicit source-text and truth-table digests. IQIF, McCulloch-Pitts, sigmoid-
  rate, and threshold-linear reference configurations now execute through their
  real public contracts instead of a universal ``n_steps``/float/tuple shape.

- **Notebook** on the network canvas downloads a notebook that builds the
  drawn network by hand with the public `sc_neurocore.network` API: one
  visible `Population` call per population with its monitor and drive, one
  connectivity call and `Projection` per projection with its seed and delay
  in whole steps, and the run in the Studio's order.
- The Studio's own run of the graph is sealed into the notebook — graph
  digest, spike count, a digest over every spike event and the CSR digest of
  every projection — and the last cell prints whether its run matches. The
  notebook cites each catalogue model it uses and states that no hardware
  step is part of it.
- `POST /api/graph/notebook` serves it; a graph that does not validate or
  fails while running answers 422.
### Added
- Reconstructed `ResonateAndFireNeuron` from Izhikevich's 2001 complex
  resonator: `x` remains current-like, `y` remains voltage-like, the maintained
  step is the exact constant-input linear flow, events are sampled upward
  crossings of `y=threshold`, and the post-event state is the generalised
  source reset `z=i*threshold` rather than a radius test or origin reset.
- Added configured atomic batch execution through Python, the modular Rust
  engine, independent Rust safety, Julia, Go C-shared, and Mojo shared-library
  lanes; an independently derived DOI trace; paired TOML/JSON schemas;
  five-runtime complete-trajectory parity; source/binary-bound benchmark
  evidence; bounded Q32.32 H1 co-simulation; and a depth-4 reset-safety job.
  Sampled event timing, the non-monotone high-drive response, the lookup-grid
  envelope, and the absence of formal equivalence or higher-silicon evidence
  are explicit.
- Corrected the legacy-named `ErmentroutKopellPopulation` provenance and
  promoted the actual Montbrió, Pazó, and Roxin (2015) equation-(12) exact QIF
  mean field across Python, the modular Rust engine, independent Rust safety,
  Julia, Go C-shared, and Mojo shared-library execution. Every lane restores
  the maintained variables exactly through `R=tau*r` and `t'=t/tau`, then uses
  the same simultaneous two-state Euler update without the previous
  undocumented firing-rate clamp.
- Added public atomic batch dispatch, paired TOML/JSON schemas, an independent
  DOI-pinned equation trace, configured five-runtime trajectory parity,
  failure-closed native contracts, and source/binary-bound local benchmark
  evidence. Generated Q32.32 RTL is bounded by co-simulation and a depth-4
  reset-and-event-silence catalogue job; continuous rates are not spikes,
  explicit Euler is implementation scope, and no formal-equivalence,
  production-speed, or higher-silicon result is claimed.
- Promoted the Jansen and Rit (1995) equation-(6) cortical-column neural mass
  across Python, the modular Rust engine, independent Rust safety, Julia, Go
  C-shared, and Mojo shared-library execution. Every lane now preserves the
  published `C2*S(C1*y0)` and `C4*S(C3*y0)` connectivity placement and commits
  the six simultaneous explicit-Euler state candidates atomically.
- Added a public seven-trace/six-final-state batch, paired TOML/JSON schemas, an
  independent DOI and Brian2-source-pinned trace, five-runtime configured
  parity, failure-closed native contracts, and a source/binary-bound local
  benchmark. Generated Q32.32 RTL is bounded by co-simulation only; the 0.1 ms
  Euler step is recorded as pinned implementation scope, and no production-
  speed or higher-silicon result is claimed.
- Promoted the Wong and Wang (2006) two-choice Appendix reduction across
  Python, the modular Rust engine, independent Rust safety, Julia, Go C-shared,
  and Mojo shared-library execution. Every runtime carries the same
  pre-update transfer, simultaneous explicit-Euler NMDA gating, AMPA
  Ornstein-Uhlenbeck current states, and explicit Gaussian-sample boundary.
- Added a public atomic six-trace/four-final-state batch, paired TOML/JSON
  schemas, an independent DOI and author-source-pinned reference trace,
  five-runtime configured parity, failure-atomic native contracts, and a
  source/binary-bound local benchmark. Generated Q32.32 RTL is bounded by
  co-simulation only; no synthesis, timing, formal-equivalence, production-
  speed, or hardware result is claimed.
- Promoted the normalised `WilsonCowanUnit` coupled E/I population-rate
  reduction across Python, the modular Rust engine, independent Rust safety,
  Julia, Go C-shared, and Mojo shared-library lanes. Every batch carries the
  complete configurable shifted-sigmoid RK4 contract and commits E/I together.
- Added a public atomic dispatcher, paired TOML/JSON schemas, explicit
  Wilson-Cowan source boundary, five-runtime trajectory parity, native failure-
  atomicity tests, configuration-preserving reset, and a source/binary-bound
  benchmark. Continuous rates are not spikes; no production-speed, fixed-point
  RTL, or hardware result is claimed.
- Completed `ThresholdLinearRateNeuron` as a validated memoryless
  `gain * max(0, current - theta)` continuous-rate contract across Python, a
  modular Rust-engine batch, independent Rust safety, Julia, Go, and callable
  Mojo shared-library paths. Every native failure is atomic, reset preserves
  configuration, and positive rates are no longer misreported as spike counts.
- Added the algebraic TOML schema, curated Gerstner-et-al. source boundary,
  exact five-backend trace parity, real backend-loading tests, and a source-
  and binary-bound local benchmark. The benchmark is non-exclusive regression
  evidence and makes no production-speed, RTL, or hardware claim.
- Completed `SigmoidRateNeuron` across the Python reference, modular Rust-engine
  batch, independent Rust-safety module, Julia, Go, and callable Mojo shared
  library. Every lane carries the full configurable exact-relaxation contract;
  reset preserves configuration and native failures are atomic.
- Added executed five-backend trace parity and a source- and binary-bound
  benchmark. Python, Rust, Julia, and Go are byte-identical on the enrolled
  200,000-step trace; Mojo remains within `3.08e-14`. Timings are explicitly
  local, affinity-pinned, non-exclusive regression evidence, not production or
  hardware performance claims.
- Completed the source-faithful McCulloch-Pitts 1943 logical neuron across
  Python, the Rust engine and independent safety module, Julia, Go, and Mojo.
  The maintained contract is a positive active-excitatory-afferent count
  threshold with absolute veto by any active inhibitory afferent; it does not
  substitute the later real-weighted threshold abstraction.
- Added paired stateless TOML/JSON schemas, an independent primary-paper truth
  table, exact varying-input five-backend parity, signed-Q32.0 registered and
  folded RTL co-simulation, and a depth-4 Z3 safety job. The source- and
  binary-hashed benchmark is explicitly local, affinity-pinned,
  non-exclusive regression evidence rather than a portable speed claim.
- Replaced the 1,018-line predictive-model GodFile with a 63-line stable facade
  and six bounded responsibility modules for contracts, native dispatch,
  Kalman filtering, RTS smoothing, controlled EM, and planning forecasts.
  Historical public identities and pickle paths remain stable; architecture
  tests enforce exclusive ownership, an acyclic import graph, and module-size
  ceilings.
- Corrected controlled EM statistics, lag covariance orientation, covariance
  validation, final-candidate likelihood acceptance, and native-result shape
  enforcement. The five maintained forward backends now include a real
  observation-wider-than-state regression that pins the bounded Mojo workspace
  repair. The source-bound 25-sample benchmark records all raw data and
  loaded-host limitations without making a universal performance claim.
- Expanded the tracked formal inventory through the McCulloch-Pitts catalogue
  job: 50 proof jobs and 180 statements (150 assert, 7 assume, 23 cover), split
  across 18 non-catalogue and 32 catalogue jobs.
- Replaced the historical `IntegerQIFNeuron` square-law approximation with the
  exact piecewise-linear, signed-integer Q0.3 recurrence from the Wu et al.
  (2021) coauthor implementation pinned at `twetto/iq-neuron` commit
  `a8752eba49dba9ba43a64be74090b91a51044b2f`.
- Added atomic full-contract execution through Python, the Rust engine and
  independent safety module, Julia, Go, and Mojo, plus TOML/JSON schema parity,
  registered and folded signed-Q32.0 RTL co-simulation, and a depth-4 Z3 job.
- Added an independently evaluated 400-tick source trace and a source-hashed
  five-backend benchmark. Its timings are non-exclusive local regression
  evidence, not production, cross-host, hardware, or universal speed claims.
- Completed homogeneous Poisson binary-bin generation across the maintained
  Python model and dispatcher, the Rust engine and independent safety module,
  Julia, Go, and Mojo. Every runtime now uses the finite-bin probability
  `1 - exp(-rate_hz * dt_ms / 1000)` and the shared seeded LFSR16 contract,
  including deterministic reset and fail-closed batch state updates.
- Added paired Poisson schemas, registered and folded Q24.24 RTL co-simulation,
  a depth-4 formal safety job, independent full-period statistical evidence,
  and a source-hashed five-backend benchmark. The benchmark record is explicitly
  non-exclusive local-regression evidence and makes no production speed claim.
- Replaced the 1,065-line mixed neuromorphic-profile registration module with
  a 65-line historical facade and five bounded responsibility modules for
  event-driven hardware, programmable/aerospace targets, heterogeneous
  accelerators, memory compute, and physical-compute platforms. All 70 profile
  values retain the exact parent digest, registry insertion order, public
  `HardwareProfile` identity, and duplicate-name failure semantics.
- Added architecture contracts for exclusive profile ownership, registry-only
  private dependencies, the 18-step facade composition order, typed and
  documented registrars, side-effect-free private reloads, public lookup/class
  filtering, and module-specific size ceilings. The 206-test focused platform
  cohort covers all 108 statements and two branches in the six governed
  production modules exactly. This catalogue is static Python metadata, not a
  numerical kernel, so no polyglot mirror or throughput claim applies.
- Refreshed the generated API and capability inventory on accepted parent
  `33d74f904`. That generated delta contains the parent's EscapeRate API, one
  validation page, and four tests plus this lane's architecture test; it does
  not represent the nine unrelated uncommitted Rust safety files.
- Registered the accepted COBA-LIF and EscapeRate evidence pages in the public
  validation navigation, closing their inherited navigation-policy gap.
- Strict typing, lint, NumPy docstrings, security, generated documentation,
  strict MkDocs, and clean-overlay installed-wheel checks pass. The installed
  wheel contains every new private module and preserves the 194-profile public
  registry, exact 70-profile digest/order, and documented platform CLI.
- Replaced the 1,109-line photonic-emitter GodFile with a 64-line historical
  facade and seven bounded responsibility modules for types, conversion, FDTD,
  compilation, Meep, crosstalk, and bitstream emission. The exact public
  export set, signatures, object identities, pickle paths, and representative
  parent-output digests are preserved.
- Added fail-closed physical and optional-runtime validation plus real,
  source-bound Rust, Go, Julia, and Mojo mirrors for the crosstalk kernel. The
  102-test focused cohort covers all 702 statements and 214 branches in the
  facade and responsibility modules exactly. Committed 4,096-pair timings are
  labelled non-isolated local regression evidence, not a universal speed claim;
  FDTD, Meep, compilation, netlist, GDSII, and filesystem work remain Python-only.
- Replaced the 1,138-line UVM generator GodFile with a 55-line historical
  facade and six bounded responsibility modules for RTL parsing, configuration,
  artifact packaging, UVM component emission, simulator/formal harnesses, and
  orchestration. The exact 13-name package API, callable signatures, class
  identities, pickle paths, and six representative parent-generated payload
  digests are preserved.
- Added UVM architecture contracts for exclusive definition ownership, the
  acyclic import graph, module-specific size ceilings, and deterministic bundle
  fingerprints. The 82-test focused cohort covers all 422 statements and 112
  branches across the eight governed package modules exactly. The split does
  not promote the draft emitted UVM/formal surface or its unrelated
  Go/Rust/Julia/Mojo placeholders, and introduces no throughput claim.
- Replaced the 1,145-line CMOS profile registration GodFile with a 67-line
  historical facade and five bounded responsibility modules for FPGA targets,
  conventional accelerators, embedded processors, architecture paradigms, and
  ASIC/simulation references. The exact 72-profile value digest, historical
  registry insertion order, public `HardwareProfile` alias, and duplicate-name
  guard are preserved; private definition-module reloads are side-effect free.
- Added architecture contracts for profile ownership, one-way registry-only
  dependencies, facade composition order, and module-specific size ceilings.
  The 197-test focused platform cohort covers all 111 statements and two
  branches in the six governed production modules exactly. Strict typing,
  lint, NumPy docstrings, security, generated documentation, strict MkDocs,
  and installed-wheel checks pass. This is static hardware metadata, not a
  numerical kernel, so no polyglot mirror or throughput claim is introduced.
- Refreshed the generated API and capability inventory on the accepted
  COBA-LIF parent. That generated delta contains the parent's committed COBA
  API, validation page, and five tests plus this lane's architecture test; it
  does not absorb uncommitted peer work.
- Replaced the historical one-state DPI surrogate with the coupled
  current-domain silicon-neuron equations of Indiveri, Stefanini, and Chicca
  (2010), Eqs. (2)–(3): nonlinear positive feedback, membrane and
  after-hyperpolarisation DPIs, simultaneous explicit Euler, post-update event
  reset, and a spike-driven refractory pulse. Python, Rust safety and engine,
  Go, Julia, and Mojo now share the maintained event surface and reject invalid
  work before visible state mutation or buffer writes.
- Added source-bound five-backend DPI parity, benchmark, reference, descriptor,
  schema, Q16.16 co-simulation, and formal evidence. The compiled paths preserve
  `0/0/0/0/1/3/6/11/21` events at
  `I=-0.1/0/1/2/3/5/10/20/50` over 1,000 steps within a `5e-13` float-state
  envelope. Hand/TOML/JSON states agree within `1e-13`; generated Q16.16 RTL
  preserves 13 events at `I=5` over 5,000 steps with its fixed-point state and
  timing envelopes declared, and the generated depth-4 Z3 job passes.
- Completed Theta end-to-end fidelity across the tangent-half-angle exact-flow
  Python reference, Rust engine and safety module, full-contract Julia and Go,
  the Mojo shared library, paired Euler schemas, and generated Q16.16 RTL.
  The public dispatcher exposes every lane and probes
  Go/Julia/Mojo/compatible-Rust/Python while avoiding Julia initialisation when
  the Go shared library is available.
- Added source-bound Theta backend, benchmark, co-simulation, descriptor, and
  formal evidence. Every compiled acceleration lane preserves
  0/0/0/1/2/2/3/5/7/14/23 events over the enrolled 1,000-step current vector and
  remains within `2e-12` circular phase error. Generated Q16.16 RTL preserves
  the same counts, with its Euler/fixed-point phase and one-cycle timing bounds
  stated separately from the exact-flow software contract.
- Completed Quadratic IF end-to-end fidelity across the exact-flow Python
  reference, the Rust safety crate and engine, full-contract Julia and Go, the
  Mojo shared library, paired schemas, and generated Q16.16 RTL. The public
  dispatcher follows the Go/Julia/Mojo/compatible-Rust/Python production order,
  avoiding Julia runtime initialisation when the Go shared library is present;
  invalid native work is rejected before caller-visible state or buffer writes.
- Added source-bound Quadratic IF backend, benchmark, co-simulation,
  descriptor, readiness, and formal evidence. Every acceleration lane preserves
  0/2/3/6/11/26/100/250 events at I=0/0.333/0.5/1/2/5/20/50 over 1,000 steps
  and stays within `2e-12` of Python. Hand/TOML/JSON events remain exact with
  state error below `0.006`; both schema formats now reset inclusively at the
  configured `v_peak`, including exact equality. Q16.16 RTL preserves the
  enrolled cycle-level event vectors with voltage error below `0.011`, while
  the I=0.1 reset-timing boundary is recorded explicitly.
- Completed Perfect Integrator end-to-end fidelity across Python, Rust safety
  and engine, Go, Julia, Mojo, the paired schema formats, and generated Q8.8
  RTL. Julia/Go/Mojo now transport the complete numeric contract through real
  executable paths, while the Rust engine retains its explicit factory-default
  boundary; every path rejects invalid inputs before visible mutation.
- Added source-bound Perfect Integrator backend, benchmark, analytic-reference,
  co-simulation, descriptor, and formal evidence. All acceleration traces and
  events are bit-exact at seven enrolled currents; hand/schema/Q8.8 RTL preserve
  66 events at I=0.7 over 1,000 steps, while the I=0.333 quantisation boundary is
  stated explicitly as 32/32/31 rather than misreported as parity.
- Replaced the 1,378-line Bioware implementation and 1,139-line primary test
  file with a 129-line historical facade, nine responsibility modules, shared
  validation, and 13 focused test modules. The largest implementation and test
  files are 296 and 253 lines. Package-root and historical imports reuse the
  same objects, qualified names and pickle paths remain stable, and the
  responsibility graph is acyclic. The focused suite passes 200 tests with
  exact 100 percent coverage over 1,052 statements and 384 branches; strict
  MyPy, Ruff, NumPy docstrings, and architecture contracts pass.
- Hardened biological-interface contracts around finite MEA matrices, fixed
  16-bit AER epochs without silent wraparound, deterministic LFSR windows,
  area-aware optical-power accounting, deterministic spike sorting, ordered
  audit checksums, bounded Q8.8 state, and transactional session round counts.
  Documentation now distinguishes research heuristics from safety or clinical
  claims, records the pharmacology wash field as reserved, and identifies the
  legacy `energy_mw` metric as an optimisation proxy.
- Removed generated Bioware mirrors that did not implement executable parity:
  empty Go functions, non-parsing Julia syntax, Mojo string/zero placeholders,
  and Rust return-zero dead code. A source-bound 30-sample parent/candidate
  benchmark proves the maintained deterministic pipeline remains identical at
  6,865 bytes and SHA-256
  `2491dc73a2de93a45a1cc944539c170b151403e42b973b18806143f318b7d669`.
  Loaded-host timings regressed locally and are retained only as diagnostic
  context, not performance claims; independent Julia plasticity solvers remain
  separate from the Python-only orchestration.
- Completed Lapicque end-to-end fidelity across Python, Rust safety and engine,
  Go, Julia, Mojo, the paired schema formats, and generated Q16.16 RTL. Public
  simulation now follows the measured Mojo/Julia/Go/compatible-Rust/Python
  order, carries the complete numeric contract through Julia/Go/Mojo, retains
  the Rust engine's explicit factory-default boundary, and fails before state
  mutation on invalid inputs or non-finite candidates.
- Added source-bound Lapicque backend, benchmark, reference, co-simulation,
  descriptor, readiness, and formal evidence. The 100,000-step pinned run
  records 20,000 events in every lane and a maximum voltage difference of
  `4.44e-16`; the loaded, non-reserved host context is recorded and is not a
  throughput claim. Hand/TOML/JSON event vectors agree exactly and their state
  traces remain within `2e-15`; Q16.16 RTL preserves the complete event vectors
  at three operating points with voltage error below `0.04`, and the generated
  depth-20 Z3 job passes.
- Replaced the 1,414-line NIR hardware-graph implementation and three primary
  test files totalling 4,091 lines with an 84-line historical facade, seven
  acyclic responsibility modules, and 40 focused test/support modules. All 28
  established definitions retain their historical imports, qualified names,
  object identities, and pickle paths. The focused package passes 240 tests
  with two optional-runtime skips and covers all 651 statements and 324
  branches; strict touched MyPy, Ruff, NumPy docstrings, architecture, and
  package-boundary contracts pass.
- Added a dedicated Python 3.12 CI step that measures the complete
  `neuron_graph*.py` surface with branch coverage and an exact 100 percent
  threshold independently of the broader NIR coverage exclusions. The split
  test package preserves parser, framework-adapter, SC-NIR export, FPGA
  compilation, file-I/O, hierarchy, dense-lowering, delay, threshold, and
  metadata workflows through their maintained production boundaries.
- Added a source-bound 30-sample parent/candidate NIR-to-FPGA benchmark. The
  canonical two-population, two-connection graph still emits an identical
  22,860-byte compilation payload at SHA-256
  `32498fa1106229a4fe064862e20b86e0f0b1f0d42f8598d1988e06e68c13ef13`.
  Graph lowering and FPGA compilation medians changed by +7.09 and +24.98
  percent, while cold import and process wall medians changed by -23.26 and
  -16.98 percent. The CPU was affinity-pinned but not reserved, so timings are
  local regression diagnostics rather than throughput claims. Rust, Go,
  Julia, and Mojo are not applicable to this typed metadata/orchestration
  refactor; model-dynamics kernels remain unchanged.
- Replaced the historical ExpIF approximation with the Fourcaud-Trocmé (2003)
  exponential integrate-and-fire recurrence, candidate-first RK4 integration,
  the fitted `-65/-68/-59.9/3.48/10` mV/ms parameter set, the paper's
  `+30 mV` finite event cutoff, and its optional `1.7 ms` refractory
  protocol. Python, Rust safety, the Rust engine, Go, Julia, and Mojo now share
  the maintained event surface and fail closed on invalid state or candidates.
- Added source-bound five-backend ExpIF parity, benchmark, reference,
  descriptor, and Python-to-Verilog evidence. The compiled paths preserve
  `0/0/2` events at `I=0/5/20` over 1,000 steps, the committed 100,000-step
  run records 523 events in every lane within a `5e-8` voltage envelope, and
  hand/TOML/JSON/Q32.32 co-simulation preserves the enrolled event counts. The
  S5/H1 descriptor is linked to a generated depth-4 Z3 proof.
- Completed the AdEx baseline-Euler acceleration chain across Rust safety, the
  Rust engine, Go, Julia, and Mojo, and exposed Julia/Go/Mojo through the public
  `simulate()` dispatcher. The compiled paths preserve the `0/4/12` event
  goldens at `I=0/200/500`, remain within `5e-12` of Python, transport the full
  maintained numeric state and parameter surface where the backend supports it,
  and reject invalid inputs without partially committing state.
- Added a source-hashed, five-backend AdEx benchmark and regression contract.
  The single-logical-CPU, powersave-governor run measured Mojo, Julia, Go, Rust,
  and Python medians of 4.658, 5.040, 12.807, 58.988, and 2202.136 ms per
  100,000 steps, with 1065 events in every lane and maximum trace error
  `7.40e-13`. The loaded-host timings are local regression evidence only.
- Replaced the Hodgkin-Huxley Mojo parity note with an executable C-ABI kernel
  for the complete four-state, gate-first baseline-Euler model. The public
  `simulate(..., backend="mojo")` path transports every maintained numeric
  state and parameter, rejects invalid inputs without committing state,
  preserves the established `0/6/9` event counts at `I=0/10/20`, and keeps the
  enrolled 100-macro-step voltage trace within `2e-9` of Python.
- Added a source-hashed Hodgkin-Huxley closure benchmark with exact event
  parity, bounded trace evidence, final states, affinity, runtime versions,
  and loaded-host context. The single-logical-CPU run measured Mojo, Rust, and
  Python medians of 1.238, 1.543, and 187.566 ms per 100 macro-steps; the
  non-isolated, powersave-governor timings are local regression evidence only.
- Replaced the Connor-Stevens Mojo parity note with an executable C-ABI kernel
  for the complete six-state, candidate-first RK4 model. The public
  `simulate(..., backend="mojo")` path transports every maintained state and
  parameter, rejects invalid inputs without committing state, preserves the
  established `0/2/9` event counts at `I=0/10/20`, and keeps the enrolled
  100-macro-step trace within `2e-6` of Python.
- Added a source-hashed Connor-Stevens closure benchmark with exact event
  parity, bounded voltage-trace evidence, final states, affinity, runtime
  versions, and loaded-host context. The single-logical-CPU run measured Mojo,
  Rust, and Python medians of 5.327, 10.377, and 379.390 ms per 100 macro-steps;
  the non-isolated timings are local regression evidence only.
- Replaced the 1,451-line ASIC-flow GodFile and 814-line primary test file with
  a 123-line historical facade, nine acyclic responsibility modules, and 11
  focused test modules. The largest implementation and test files are 319 and
  234 lines. All 38 historical definitions retain object identity, qualified
  names, and pickle compatibility. The focused suite passes 100 tests with
  exact 100 percent coverage over 580 statements and 66 branches; strict
  touched MyPy, Ruff, NumPy docstrings, and architecture contracts pass.
- Removed nonfunctional generated ASIC-flow mirrors: Rust returned constants,
  Go exposed empty functions, Julia did not parse, and Mojo returned zeros.
  The maintained boundary is typed Python deck generation followed by external
  Yosys/OpenROAD/OpenSTA/KLayout/Magic/Netgen execution. A source-bound,
  30-sample parent/candidate benchmark retains raw samples and proves the
  canonical 10,465-byte output is unchanged at SHA-256
  `ae901f9b10bdc61f0997964d6143568994625bdf89080f02cd58efbc83099653`.
  Its non-exclusive, loaded-host timings are regression diagnostics, not
  physical-design or throughput claims.
- Replaced the 1,737-line evolutionary-substrate GodFile and 1,227-line
  primary test GodFile with a 156-line historical facade, 14 acyclic
  responsibility modules, and 18 focused test modules. No implementation file
  exceeds 304 lines and no owned test file exceeds 312 lines. All 56 public
  exports retain object identity, historical qualified names, and pickle
  compatibility. A deterministic parent/candidate probe remains byte-identical
  across genome, variation, fitness, safety, speciation, statistics, and
  three-generation replication workflows.
- Added fail-closed architecture, branch, runtime-fault, package, and polyglot
  contracts for the evolutionary substrate. The complete focused Python suite
  passes 160 tests with the real Mojo and four-runner parity paths enabled;
  135 core tests cover all 1,005 statements and 166 branches. Native Rust,
  Julia, Go, and Mojo checks pass 17, 17, 8, and 7 tests respectively, and the
  18-test cross-language runner suite has no skips. Strict touched MyPy, Ruff,
  NumPy docstrings, Bandit, SPDX, and staged-index wheel smoke checks are clean.
- Rebuilt both evolutionary benchmark producers as source-bound schema-v2
  evidence with raw samples, medians, runtime versions, affinity, governor,
  frequency, and host-load context. The 30-sample Python and five-language
  artefacts were captured without kernel-reserved cores under heavy concurrent
  load; their timings are diagnostic regression context and require an
  isolated-core rerun before publication.
- Replaced the 1,910-line quantum-annealing GodFile and its 1,001-line
  primary test GodFile with a 98-line historical facade, nine acyclic
  responsibility modules, and focused typed tests. Every production module is
  below 400 lines and every owned test file is below 500 lines. All 24 public
  imports retain object identity and historical pickle paths. Models now
  canonicalise terms and fail closed on invalid indices, labels, shapes,
  cardinalities, schedules, samples, and non-finite data. The exact QUBO to
  Ising mapping fixes diagonal offsets, explicit backend selectors never
  silently substitute a requested runtime, native responses are validated,
  exports are atomic, and decomposition reconstructs exact global indices.
  The linked focused cohort passes 250 tests with one optional `neal` skip and
  reaches 100 percent coverage over 1,040 statements and 442 branches.
- Removed false quantum-annealing mirrors: the generated Rust safety file
  returned constants, the Go service exposed empty functions, and the Julia
  and Mojo files did not build. The maintained native authority remains
  `engine/src/quantum.rs` with 12 focused tests. A source-hash-bound,
  30-sample local parent/candidate benchmark records median compile, cold
  import, Python solve, process wall, and RSS changes of +99.65 percent,
  -25.37 percent, -16.67 percent, -23.21 percent, and -3.48 percent. The
  non-exclusive workstation load rose during capture, so these are regression
  diagnostics rather than release throughput or quantum-speedup claims.
- Replaced the 1,926-line safety-certification GodFile and 2,262-line test
  GodFile with a 98-line historical facade, ten acyclic responsibility modules,
  and focused typed test modules, all below 600 lines. The established 32
  public evidence classes retain object identity and historical pickle paths.
  Package generation now fails closed: it does not invent implementation
  files, FIT/DC data, timing inputs, or addressed checklist rows. Full-field
  formal-evidence hashes, concrete traceability links, path-safe evidence
  manifests, and atomic six-file package materialisation are wired end to end.
  The focused safety/consumer cohort passes 467 tests with 100 percent
  statement and branch coverage over 1,437 production statements and 858
  branches; strict touched MyPy, Ruff, and public docstrings are clean.
- Removed unreachable generated safety_cert report mirrors: Julia and Mojo did
  not parse, Go exposed only empty functions, and Rust returned constants
  behind a test that always passed. The distinct SafetyMonitor chain remains.
  A source-hash-bound 30-sample local parent/candidate benchmark records median
  import, generation, process wall, and RSS changes of +16.62 percent,
  +37.31 percent, +6.18 percent, and -1.31 percent. Generation remains about
  0.172 ms while performing full-field hashing and fail-closed reporting. The
  non-exclusive workstation measurements are regression context, not
  publishable throughput claims.
- Replaced the 3,257-line molecular/DNA mapper GodFile with a 213-line
  compatibility façade and nine acyclic responsibility modules, each below 600
  lines. Historical documented imports, optional NUPACK reload/monkeypatch
  behavior, and pickle module identities remain stable; deterministic parent
  and candidate designs, exports, simulations, analyses, and serialized
  artefacts match. The focused cohort passes 316 tests, and the complete
  touched surface reaches 100% coverage over 1,220 statements and 406
  branches. Missing simulator outputs now fail closed in noise and
  concentration analysis. The bridge package no longer eagerly requires the
  source-only, wheel-excluded optics tree before DNA imports can succeed.
- Added a hash-bound 30-sample parent/candidate benchmark with raw samples and
  explicit workstation limitations. Median cold import, repeated one-gate
  compilation, wall time, and RSS changed by +8.25%, +4.06%, +6.94%, and
  -0.08%, respectively, while one-minute host load rose from 10.56 to 15.00.
  These are local regression observations requiring a reserved-core rerun, not
  release claims. The maintained Rust safety mirror compiles standalone
  and passes six tests. Removed generated Julia/Mojo DNA files that did not
  parse and a Go package containing only empty functions; none was reachable
  from canonical dispatch.

### Event-dataset manifests, group splits and declared encoders
- Replaced the 2,318-line command-line monolith with a stable
  `sc_neurocore.cli:main` package entry point, a 45-line parser composition
  root, and responsibility-specific compile, deploy, formal, SC-NIR, Studio,
  synthesis, mapping, serve, hub, status, and maintenance command modules.
  Top-level help now presents Model, Hardware, Studio, and Maintain modes with
  command-level progressive disclosure. The split preserves console and
  `python -m sc_neurocore.cli` invocation, makes `compile-nir` precision flags
  effective, and keeps SC-NIR upgrade/export destinations explicit. Focused
  command tests reach 100% of the package's 1,172 statements and 294 branches;
  real NIR/Icarus co-simulations use the post-reset membrane observable shared
  by generated RTL and the bit-true Python model. Removed unreferenced
  Rust/Julia/Mojo files that were generated comment/constant stubs rather than
  callable CLI counterparts. A path-free 30-run local benchmark records a
  22.27% lower median import cost, effectively unchanged process wall time,
  and unchanged RSS versus the parent flat module, with non-exclusive CPU and
  host-load limitations preserved in the evidence artefact.
- Replaced the unsupported Ibarz-Tanaka beta-based rational/linear hybrid with
  the four-branch fast/slow map from Ibarz, Tanaka, Sanjuan, and Aihara (2007),
  DOI `10.1103/PhysRevE.75.041902`, Eqs. 2–3. The stale review citation and
  non-source `beta`, threshold, and reset parameters were removed; an event is
  now execution of the source's fixed `-1` reset branch.
- Rebuilt the checked Python, Rust engine, Rust safety, Go, Julia, and Mojo
  paths. Rust, Julia, and Go are bit-exact on the enrolled 1,000-step envelope;
  Mojo preserves event vectors within its measured absolute-error band. A
  source-hashed, CPU-pinned 21-call benchmark at `I=0.2` records 33 events in
  every lane and discloses the host's high load and absent kernel isolation.
- Added paired schemas, an independent DOI reference contract, and bounded
  Q16.16 co-simulation. Hand/TOML/JSON trajectories are exact; the 30-step RTL
  run exercises all four branches and preserves all three reset events with
  `v/u` errors below `0.003/0.0001`. Q8.8 cannot represent `mu=0.001`.
- Promoted the corrected model to S5/H1 with a generated Q16.16 catalogue core
  and depth-4 SymbiYosys/Z3 safety proof. Schema coverage moves to 33 models;
  the catalogue moves to 26 jobs and the full HDL tree to 63 jobs / 211
  statements.
- Replaced the falsely attributed Medvedev tent map with the scalar
  slow-calcium first-return construction derived in Section 4 of Medvedev
  (2005), DOI `10.1016/j.physd.2005.01.021`. The three source regions follow
  equations 4.4/4.7, 4.8 with 4.13, and 4.15. The global return calibration,
  non-zero-current perturbation, and pre-state event convention are disclosed
  as maintained SC-NeuroCore choices rather than attributed to the paper.
- Rebuilt the checked Python, Rust engine, Rust safety, Go, Julia, and Mojo
  lanes around the same recurrence. Rust, Julia, and Go are trace-identical to
  Python on the enrolled envelope; Mojo preserves event vectors with a measured
  maximum trace difference below `5e-13`. The committed 500,000-iteration,
  five-repeat benchmark records 375,000 events in every lane and medians of
  8.230/10.799/19.524/20.959/242.929 ms for Julia/Rust/Mojo/Go/Python.
- Added paired `medvedev_map` TOML/JSON schemas and an independent Section 4
  reference contract. Hand model and both schemas agree exactly; at `I=2`,
  emitted Q16.16 RTL preserves the complete 75-event, 100-iteration vector with
  maximum state error below `0.007813`. Q16.16 is required because the disclosed
  `d=2271.1927977404063` calibration cannot fit signed Q8.8. Verilog and generated
  integer C/Rust now share a 256-entry positive-domain log LUT over
  `[1/256, 8 + 1/256)` at step `1/32`.
- Promoted the corrected model to S5/H1 with a generated Q16.16 catalogue core
  and depth-4 SymbiYosys/Z3 safety proof. The public fidelity count remains 21,
  schema-gap counts move to `32 / 121 / 123`, the formal catalogue moves to 25
  jobs, and the full HDL tree moves to 62 jobs / 210 statements.
- Refactored the 3,662-line Studio application godfile into a 147-line
  composition root, 16 responsibility-specific API routers, and dedicated
  runtime, security, schema, guard, preset-flow, and frontend support modules.
  The hermetic public contract is unchanged: all 113 backend HTTP routes, the
  progress WebSocket, and the 110-path OpenAPI document retain the
  pre-refactor semantics and canonical SHA-256
  `b1e698de8ce38e15f152a3bffdf86d24dc8f14c12747a927be320bcc25825cdb`.
  The conditional frontend document route remains outside OpenAPI, so a local
  ignored `dist/` build can no longer change the committed API reference.
  Added a deterministic OpenAPI generator, committed reference document,
  responsibility-ownership tests, normalized handler-body parity evidence,
  and route-level failure tests covering worker, identity, audit, export,
  training-weight, preset, precision, catalogue, and analysis boundaries.
- Re-enrolled the already acceleration-complete Hindmarsh-Rose model (DOI
  `10.1098/rspb.1984.0024`) on its maintained three-state RK4 path. The paired
  schemas now match the hand model's no-reset upward `x >= x_threshold`
  crossing semantics, replacing a stale explicit-Euler schema whose identity
  reset turned crossing detection into level counting. An independent RK4
  reference reproduces the 2,000-step `I=3` bursting trace without calling the
  hand model or schema runner.
- Hand model, TOML, JSON, and Q16.16 RTL produce exact 2,000-step crossing
  counts `0/0/26/40/52` at `I=0/2/3/4/5`. Dedicated tests also preserve the
  5,000-step chaotic boundary: Q16.16 produces one extra crossing at each
  bursting point `I=2` through `I=5`. The S5/H1 descriptor adds a generated
  Q8.8 port-only formal job whose depth-4 SymbiYosys/Z3 check passes. Existing
  model, acceleration, backend-parity, and benchmark artefacts are unchanged;
  schema-gap counts remain `31 / 122 / 124`, while the formal catalogue moves
  to 24 jobs and the whole HDL inventory to 61 jobs / 209 statements.
- Completed the source-verified Chialvo map unit (DOI
  `10.1016/0960-0779(93)E0056-H`) across its checked Python, Rust engine,
  Rust safety, Go, Julia, and Mojo simulation paths. All lanes implement the
  simultaneous `x*x*exp(y-x)+k+I`, `a*y-b*x+c` recurrence; the upward
  `x_threshold=1.0` event is documented as a maintained observation rather
  than a paper equation. The production `auto` path now consumes the
  host-matched committed benchmark order. A single-CPU-affinity,
  500,000-iteration result records identical 12,935-event counts and measured
  medians of 7.270/9.576/11.373/20.524/2,175.866 ms for
  Rust/Julia/Mojo/Go/Python, with environment and source hashes preserved in
  `benchmarks/results/bench_chialvo_map.json`.
- Added paired `chialvo_map` schemas, an independently re-derived 100-iteration
  DOI reference, and Q16.16 co-simulation. Hand/TOML/JSON states and events are
  exact at the five enrolled currents; RTL preserves event counts `0/2/3/0/1`
  and the declared stable-point error envelope while oscillatory event timing
  remains an explicit boundary. The S5/H1 promotion adds a generated Q8.8
  formal job whose depth-4 SymbiYosys/Z3 check passes. The public fidelity
  count moves to 21; schema-gap counts move to 31 schema models / 122 net
  missing / 124 source modules without a schema; the formal catalogue moves to
  23 jobs and the whole HDL inventory to 60 jobs / 208 statements.
- Enrolled the already acceleration-complete Ermentrout-Kopell hand class
  (`ermentrout_kopell_map_neuron`, DOI `10.1137/0146017`) with paired
  TOML/JSON schemas and an independent 2,000-step theta-flow reference. The
  source boundary is explicit: the paper supplies the continuous Type-I theta
  equation, while `dt=0.1`, gain, forward Euler, the upward `theta=pi` event,
  and modulo `2*pi` are maintained implementation choices. Hand model and both
  schemas agree exactly under varied negative/positive drive. Q16.16 RTL
  preserves 0/45/64 spikes at `I=-0.5/0.5/1.0` over 2,000 steps, with maximum
  circular phase error below 0.081/0.089/0.025 rad; event timing and full
  fixed-point trajectories are not claimed exact. The equation runtime,
  Verilog emitter, and generated integer C/Rust kernels now share explicit
  `<state>_prev` macro-boundary aliases and Python-compatible modulo by a
  finite positive literal, preventing a negative backward wrap from becoming
  a false event. Integer C/Rust and Verilog state/event words co-simulate
  cycle-exactly over the 240-step protocol for both current signs. The S5/H1
  descriptor adds a generated
  Q8.8 formal job whose depth-4 SymbiYosys/Z3 safety check passes. All 21
  pre-existing catalogue jobs regenerate byte-identically; the inventory moves
  to 22, and schema-gap counts move to 30 schema models / 123 net missing / 125
  source modules without a schema. Existing floating-point acceleration and
  benchmark artefacts remain unchanged.
- Enrolled the already acceleration-complete Courbage-Nekorkin-Vdovin
  discontinuous map (`courage_nekorkin_map`, DOI `10.1063/1.2795435`) with
  paired TOML/JSON schemas that reproduce the maintained simultaneous
  piecewise recurrence, Heaviside branch, and upward event crossing. The hand
  model and both schemas agree exactly at every enrolled point. Q16.16 RTL is
  event-exact at `I=-0.3/0/0.3` over bounded 30/20/30-iteration windows, with
  both state errors below `0.014`; Q32.32 is event-exact at all three inputs
  over 30 iterations with `x` error below `0.00003` and `y` error below
  `0.000001`. The autonomous 30-iteration Q16.16 trace is pinned separately at
  four float64 events, six RTL events, and six event-position mismatches rather
  than admitted through a loose tolerance. The model-scoped tests add a
  DOI-backed independent recurrence, S5/H1 descriptor and readiness evidence,
  source-bounded public documentation, and generated Q8.8 RTL. Its port-only
  depth-4 SymbiYosys/Z3 reset-spike safety job passes. The formal inventory
  moves to 21 models and schema-gap counts move to 29 schema models / 124 net
  missing / 126 source modules without a schema. The completed
  Rust/Go/Julia/Mojo chain and its committed benchmark artefact remain
  unchanged.
- Enrolled the already acceleration-complete Cazelles fast/slow map
  (`cazelles_map`, DOI `10.1209/epl/i2001-00548-y`) with paired TOML/JSON
  schemas that mirror the maintained simultaneous clipped logistic recurrence,
  slow-state update from the old fast coordinate, and committed-state level
  event. Hand model and both schemas agree exactly on every state and event at
  the enrolled points. Over 30 iterations, emitted Q16.16 RTL reproduces the
  complete 2/1/1 event vectors at `I=0.5/1.0/2.0`, with each state coordinate
  within `0.0004` of float64 while the operating set exercises the interior
  expression and both fast-state clip bounds. The sensitive `I=0.05` trace is
  pinned separately at seven float64 events, eight RTL events, and seven event
  mismatches rather than promoted through a loose tolerance. The Cazelles
  co-simulation and independent-reference tests live in model-scoped modules,
  not the legacy catalogue-wide test accumulators. The unit adds the DOI-backed
  independent map reference, S5/H1 descriptor and readiness entry, public
  fidelity documentation, and generated Q8.8 RTL. Its port-only depth-4
  SymbiYosys/Z3 reset-spike safety job passes. The formal inventory moves to
  20 models and schema-gap counts move to 28 schema models / 125 net missing /
  127 source modules without a schema. The completed Rust/Go/Julia/Mojo chain
  and its committed benchmark artefact remain unchanged.
- Corrected the already acceleration-complete Mihalas-Niebur co-simulation evidence after the
  shared candidate-reset/output fix. The paired TOML/JSON schemas reproduce every event and all
  four hand-model states exactly over a varied 1,600-step sequence with 168 adaptive resets. The
  former `I=3`, 300-step hand/schema/RTL claim is now exact at 36/36/36, superseding the stale
  36/36/35 record and its loose `<=2` guard. A 1,000-step Q16.16 contract now asserts exact
  three-way counts of 0/0/0/31/60/87/131/157/207/256 at
  `I=0/0.5/1/1.5/2/2.5/3.5/4/5/6`, while a separate regression fixes the isolated `I=3`
  boundary at 111/111/112 rather than hiding it inside a general tolerance. The S5/H1 descriptor,
  readiness index, fidelity page, model documentation, and co-simulation guide record the same
  boundary; the existing depth-3 SymbiYosys/Z3 formal job remains the structural safety evidence.
  Model, acceleration, schema, compiler, generated RTL, formal source, and benchmark artefacts are
  unchanged.
- Corrected the already acceleration-complete GLIF5 schema-to-RTL surface to match the maintained
  hand model: paired TOML/JSON four-state classical RK4, candidate-level `v >= theta` detection,
  and candidate-first adaptive reset replace the stale Euler / strict-greater-than contract. A
  varied 4,000-step drive gives exact hand/TOML/JSON state and event parity across 181 resets. The
  six-point 1,000-step Q16.16 contract spans silent through high-drive regimes with exact
  hand/schema/RTL counts of 0/0/23/54/86/95 at `I=0/15/22/30/45/50`. The equation compiler and
  bit-true C/Rust generators now evaluate reset expressions from the integrated candidate and expose
  identical post-reset state, removing the pre-step-reset mismatch that a one-spike band had masked.
  The DOI-backed reference is an independent, spike-bearing RK4 re-derivation; the S5/H1 descriptor
  and readiness evidence record the same observable. Deterministic regeneration changes all ten
  reset-using formal RTL models plus seven non-resetting single-step edge models so every spike-cycle
  output exposes the committed candidate. The Connor-Stevens and Hodgkin-Huxley macro-step outputs
  already exposed that candidate and remain byte-identical; all 19 jobs are therefore accounted for
  and the inventory count is unchanged. Acceleration kernels and benchmark artefacts are unchanged.
  The corrected reset exposes one honest legacy
  Izhikevich Q8.8 boundary (float64 25 spikes, RTL 24 at `I=50` over 200 steps); Q16.16 remains
  exact at 25/25 and now has a dedicated regression guard.
- Descriptor regeneration now reads `integration.dt` from an authoritative bundled map schema when
  a model class has no public `dt` field, preserving the unit-iteration metadata of pure discrete
  maps such as Rulkov without changing the fallback semantics of non-map models. Readiness facet
  application now leaves descriptors unchanged when their existing science and silicon evidence
  already meets or exceeds the indexed floor, preventing an H0 index entry from replacing Rulkov's
  trajectory-backed S5/H2 evidence.
- Class-correct schema-to-RTL re-enrolment of the Rulkov fast/slow map (`rulkov_map`,
  Rulkov 2002, DOI `10.1103/PhysRevE.65.041922`). The paired TOML/JSON schemas now mirror
  the maintained hand model's simultaneous three-branch map and rising `x >= 0` crossing
  semantics instead of counting every positive plateau step. At `I=1.5`, a bounded 30-iteration
  window exercises the rational, plateau, and hard-reset branches ten times each: hand model and
  both schema formats have exact state/event agreement, while the emitted Q16.16 RTL reproduces
  the exact ten-event vector with both state errors below `0.001`. This short-window trajectory is
  the class-appropriate map metric; no long-window spike-count identity is claimed. The descriptor
  now records faithful map dynamics, S5 trajectory evidence, and the documented H2 terminal backed
  by a raw Yosys 0.33 `synth_xilinx` report over the Q16.16 core. The resulting S5/H2 descriptor is
  registered with the formal catalogue through generated Q8.8 RTL, a port-only harness, and a
  depth-4 SymbiYosys/Z3 reset-spike safety job, bringing the inventory to 19 models. The existing
  Rust/Go/Julia/Mojo acceleration chain and committed benchmark artefacts remain unchanged.
- Faithful schema-to-RTL enrolment of the Wilson-HR two-state polynomial cortical model
  (`wilson_hr`, Wilson 1999, DOI `10.1006/jtbi.1999.1002`). The paired TOML/JSON schemas mirror
  the maintained hand model's simultaneous classical RK4 flow, polynomial membrane nullcline,
  level `v >= v_peak` decision, and hard `v = -0.7` reset that preserves the candidate recovery
  state. A varied 4,000-step drive produces 35 spikes and resets with exact hand/TOML/JSON state
  agreement. Over 5,000 constant-current steps the hand model, schema runner, and emitted Q16.16
  RTL agree exactly on 0, 1, and 4 spikes at `I=0.0`, `2.0`, and `10.0`. Added the DOI-backed
  `wilson_hr_driven_spiking_doi` trace with an independent two-state RK4 re-derivation, corrected
  the public model provenance, and marked the descriptor's RK4/co-sim facets. The resulting S5/H1
  descriptor is wired into the formal catalogue with generated Q8.8 RTL, a port-only harness, and
  a depth-4 SymbiYosys reset-spike safety job; the 18-model inventory and direct Z3 BMC pass.
  Schema-gap counts move to 27 schema models / 126 net missing / 128 source modules without a
  schema. The completed Rust/Go/Julia/Mojo acceleration chain and its committed benchmark
  artefacts are unchanged.
- Faithful schema-to-RTL enrolment of the Terman-Wang two-state LEGION relaxation oscillator
  (`terman_wang`, Terman & Wang 1995, DOI `10.1016/0167-2789(94)00205-5`). The paired TOML/JSON
  schemas mirror the maintained hand model's simultaneous classical RK4 flow, cubic fast
  nullcline, `tanh`-gated recovery, rising-edge `v >= v_peak` decision, and no-reset semantics.
  Over 8,000 steps the hand model, schema runner, and emitted Q16.16 RTL agree exactly on the
  silent/single/train crossing counts: 0 at `I=-1.0`, 1 at `I=0.0`, and 3 at `I=0.5`. Added the
  DOI-backed `terman_wang_legion_oscillation_doi` trace with an independent two-state RK4
  re-derivation, corrected the public model citation, and marked the descriptor's RK4/co-sim
  facets. The resulting S5/H1 descriptor is wired into the formal catalogue with generated Q8.8
  RTL, a port-only harness, and a depth-4 SymbiYosys reset-spike safety job; the 17-model inventory
  and direct Z3 BMC pass. Schema-gap counts move to 26 schema models / 127 net missing / 129 source
  modules without a schema. The completed Rust/Go/Julia/Mojo acceleration chain and its committed
  benchmark artefacts are unchanged.
- Faithful schema-to-RTL enrolment of the Pernarowski three-state pancreatic beta-cell burster
  (`pernarowski`, Pernarowski 1994, DOI `10.1137/S003613999223449X`). The new TOML/JSON schema
  mirrors the maintained hand model's simultaneous classical RK4 flow, exact `v * v * v`
  operation order, rising-edge `v >= v_threshold` decision, and no-reset semantics. The hand
  model, schema runner, and emitted Q16.16 RTL have exact spike-count parity at all four enrolled
  5,000-step operating points: 17 crossings at each of `I=-0.1`, `0.0`, `0.1`, and `0.2`.
  Added the DOI-backed `pernarowski_autonomous_bursting_doi` trace with an independent
  three-state RK4 re-derivation and marked the descriptor's RK4/co-sim facets. The resulting S5/H1
  descriptor is also wired into the formal catalogue with generated Q8.8 RTL, a port-only harness,
  and a depth-4 SymbiYosys reset-spike safety job; the 16-model inventory and direct Z3 BMC pass.
  Schema-gap counts move to 25 schema models / 128 net missing / 130 source modules without a
  schema. The already completed Rust/Go/Julia/Mojo acceleration chain and its committed benchmark
  artefacts are unchanged.
- Faithful schema-to-RTL enrolment of the FitzHugh-Rinzel three-state qualitative burster
  (`fitzhugh_rinzel`, Rinzel 1987, DOI `10.1007/978-3-642-93360-8_26`). The new TOML/JSON
  schema mirrors the maintained hand model's coupled classical RK4 flow, exact `v * v * v`
  operation order, rising-edge `v >= v_threshold` decision, and no-reset semantics. The hand
  model, schema runner, and emitted Q16.16 RTL have exact spike-count parity across the enrolled
  `I=0.4` to `I=0.6` band (seven, eight, and eight crossings over 3000 steps); `I=0.7` is recorded
  as an excluded marginal-crossing boundary rather than hidden by a tolerance. Added the
  DOI-backed `fitzhugh_rinzel_driven_bursting_doi` trace with an independent three-state RK4
  re-derivation, corrected the public model page's stale reset claim, and marked the descriptor
  RK4/co-sim facets. Schema-gap counts move to 24 schema models / 129 net missing / 131 source
  modules without a schema. The already completed Rust/Go/Julia/Mojo acceleration chain and its
  committed benchmark artefacts are unchanged.
- Same-name PyO3 engine binding for `StochasticLIFNeuron`. The Rust neuron
  (`engine/src/neurons/trivial.rs` — leaky integrate-and-fire with Euler-Maruyama membrane noise and
  an integer refractory clamp) already existed and was wired into the `NetworkRunner`, but carried no
  `#[pyclass]` wrapper, so once the model was enrolled in the public registry the coverage map in
  `tests/test_rust_python_neuron_parity.py` correctly flagged it as an uncovered non-Python-only
  entry. `PyStochasticLIFNeuron` now exposes the same `new(seed)` / `step` / `reset` / `get_state`
  surface as its `StochasticIFNeuron` sibling and is re-exported from the engine package root. It
  joins the RNG-dependent parity set — exact traces are not claimed because the Gaussian stream is
  `Xoshiro256++` Box-Muller rather than NumPy's PCG64 Ziggurat, so the spike count is the stated
  parity observable. The registry map is now 160 public Python registry names, 146 same-name Rust
  constructors, and 176 Rust PyO3 model wrappers.
- Sequential (Gauss-Seidel) integration mode (`[integration] method = "gauss_seidel"`) in the
  schema DSL, in both the Python runner (`EquationNeuron`) and the emitted Verilog. The state
  variables are advanced in declaration order, each derivative reading the already-committed
  earlier variables within the same sub-step (unlike the simultaneous `euler`/`rk4` modes, whose
  derivatives all read the pre-step state). This lowers a conductance hand model's
  gates-then-voltage update — the gating variables from the old membrane voltage, then the voltage
  from the new gates. The Verilog emitter renders each earlier variable as its freshly-committed
  `<var>_next` wire in a later variable's derivative (a commit-before-read dependency chain, no
  combinational cycle), matching the runner. Composes with `substeps`; the default `substeps = 1`
  simultaneous methods stay bit-for-bit unchanged.
- Faithful re-enrolment of the Wang-Buzsáki (1996) fast-spiking interneuron. The bundled schema was
  a single-step `method="euler"` re-derivation with a sigmoid-caricature `m_inf`, unfaithful gate
  initial conditions (`h=0.6`, `n=0.32`), a `v > -10` threshold, and a singular `n` rate; it is now
  `method="gauss_seidel"` with `substeps=50`, state ordered `h, n, v` (`h=0.8`, `n=0.1`, `v=-65`),
  the true instantaneous `m_inf = alpha_m/(alpha_m+beta_m)` (with `alpha_m` in the stable exprel
  form `1/exprel(-(v+35)/10)`), the exprel `n` rate `0.1/exprel(-(v+34)/10)`, a macro-boundary
  `v >= v_threshold` crossing (`v_threshold=-20`), and no reset — matching `WangBuzsakiNeuron`
  exactly (`hand == schema`, 3 action potentials at `I=10` over 20 macro steps). The Q16.16 RTL
  tracks the schema within one spike over the bounded window (three-way exact at `I=10`,
  `macro=20`; the residual is the `m_inf` fixed-point divide plus 256-entry exprel look-up, not a
  datapath-precision limit). The reference trace is re-derived as `wang_buzsaki_driven_spiking_doi`
  (`independent_macrostep_gauss_seidel_reference`, a 3-state helper bit-exact against the runner,
  spike_count 4, first_spike_step 4), replacing the deleted resting-gate trace; the descriptor
  dynamics are synced to the exprel form; the bundled 15%-band cosim test becomes a macro-step
  three-way parity test (`hand == schema` exact, `|schema - verilog| <= 1`). DOI
  `10.1523/JNEUROSCI.16-20-06402.1996` Crossref-verified.
- Schema/module/class alias registry (`neurons/schema_module_aliases.py`) so
  schema-DSL stems such as ``exp_if``, ``resonate_fire``, and ``lif`` join the
  correct hand modules and descriptor classes without fuzzy matching; schema-gap
  reporting consumes the same table.
- Catalogue registration for the public SC flagship ``StochasticLIFNeuron``
  (thin ``models/stochastic_lif`` re-export + curated descriptor) so Studio and
  dual-axis coverage include the package-root quickstart neuron. Schema stem
  ``lif`` remains Lapicque-lineage for silicon cosim — documented in the alias
  table and descriptor notes.
- Readiness evidence index (`tools/readiness_evidence_index.py`) that wires dual-axis
  validation and silicon facets onto descriptors for models already enrolled in
  schema→RTL co-simulation, and preserves those facets across descriptor corpus
  regeneration. Studio catalogue readiness now reports H0/H1 for that shortlist
  (Wang-Buzsaki was deferred here while its re-enrolment lane was open; that lane has since landed
  the faithful Gauss-Seidel enrolment above, so it can join the shortlist).
- Macro-step integration mode (`[integration] substeps = N`) in the schema DSL, in both the
  Python runner (`EquationNeuron`) and the emitted Verilog. One macro `step()` advances `N`
  inner integration sub-steps (any method — euler/rk4/exp_euler) before a single spike
  decision, and the rising-edge crossing is evaluated only on the macro boundary against the
  condition at the previous macro boundary. This lets the schema faithfully replicate the
  maintained conductance hand models whose `step()` is itself a fixed number of fine sub-steps
  per macro step (Hodgkin-Huxley / Connor-Stevens: 100 `dt=0.01` sub-steps per 1 ms; Wang-Buzsaki:
  50 per 0.5 ms), so a repetitively firing oscillator counts one spike per action potential
  rather than one per sub-step it stays above threshold. The RTL keeps one integration sub-step
  per clock and gates the crossing to the macro boundary with a sub-step counter; the lowering is
  bit-exact against the Python runner (proven on the polynomial FitzHugh-Nagumo at Q16.16, exact
  across sub-step groupings). Supported for the edge (crossing, non-resetting), non-pipelined
  datapath the conductance oscillators need; other combinations raise `NotImplementedError` rather
  than emit RTL that silently disagrees with the golden. With the default `substeps = 1` every
  existing model is bit-for-bit unchanged.
- Rising-edge (`detection = "crossing"`) threshold detection in the schema DSL, in both
  the Python runner (`EquationNeuron`) and the emitted Verilog. A non-resetting oscillator
  (FitzHugh-Nagumo, McKean) spikes once per upward threshold crossing rather than on every
  step it stays above threshold, so such models now co-simulate faithfully against their
  hand models — validated by the faithful FitzHugh-Nagumo hand model / schema / Q16.16 RTL
  three-way parity at exact spike counts (8 of 3000 steps, a sustained relaxation-oscillation
  train). Edge detection engages only for a crossing model that declares **no reset**: a
  reset that drops the state below threshold already clears the condition each spike, so
  reset-based integrate-and-fire models keep the identical (previously validated) level
  datapath whether they declare `level` or `crossing`. The RTL carries a 1-bit `_thr_prev`
  edge-history register seeded from the initial state to match the golden bit-for-bit. The
  folded `compile_to_datapath` path remains level-only (per-instance modules cover the
  co-simulation path); crossing support there is a separate follow-up.

- `build_manifest` records an N-MNIST, SHD or CIFAR10-DVS directory: every
  file by size and SHA-256, every sample by published split, label and
  group, and the publisher's citation, licence, sensor geometry and time
  unit. `verify_manifest` names missing, changed and unlisted files. Nothing
  is downloaded, and the data version is stated by the user.
- `group_split` divides a published split by whole groups — speakers in
  SHD, recordings where the publisher names no finer identity — from a
  seed, and records the manifest it was drawn from; `leaked_groups` checks
  a plan read back, and `group_overlap` reports groups the publisher's own
  splits share.
- `EventBinning`, `PoissonRates` and `FirstSpikeLatency` declare everything
  they do; `encoder_from_declaration` rebuilds the identical encoder and
  refuses a declaration the running version would not reproduce. Event
  binning drops late events and refuses events outside the sensor.
- `sc-neurocore dataset manifest | verify | split` runs the same from the
  command line.
- The datasets page described the N-MNIST format that the reader fixed
  earlier in this release no longer uses; it now gives the published 40-bit
  layout and the SHD time unit.

### Studio workers no longer import PyTorch for analyses

- Every process that imports the Studio routers imported PyTorch through the
  training job module, including the confined workers that run simulations
  and analyses. A PyTorch build with its GPU libraries maps several gigabytes
  of address space, which the worker's address-space limit counted, so a long
  analysis could fail with `MemoryError` before its own deadline. Whether
  PyTorch is installed is now found without importing it, and the training
  run imports it when it starts; a test holds that an analysis worker's
  imports leave PyTorch unloaded.

### Event dataset loaders read the real file formats

- N-MNIST `.bin` files were decoded as a 16-bit address with x and y in five
  bits each — too few for the 34 × 34 sensor — with the polarity bit inside
  the timestamp and microseconds scaled by the synthetic time step. They are
  now decoded as published: byte 0 x, byte 1 y, the top bit of byte 2 the
  polarity, 23 bits of microseconds, reported in milliseconds.
- SHD spikes after the requested `T`-step window were clipped into the last
  bin, inventing a burst at the end of long samples; they are now dropped.

### Cited notebooks generated from replay packs

- **Notebook** (`POST /api/export/replay-notebook`) downloads a Jupyter
  notebook built from a freshly sealed replay pack: it cites the catalogue
  model from its descriptor (or says custom equations have no published
  source), states that it runs the software model only, carries the pack
  inline, and replays it with runtime differences reported rather than
  refused. A test runs a generated notebook's code in a fresh process away
  from the author's files and gets the verdict `match`.

### Review comments bound to immutable workspace revisions

- A comment is made on one saved revision and records that revision's state
  digest; replies thread under a comment on the same revision. Listing checks
  each comment against its revision and marks one whose revision changed or is
  gone. Comments are appended, never rewritten; the author is the
  authenticated principal, or `local` in the single-user lab profile.
- `GET /api/project/{name}/comments` and
  `POST /api/project/{name}/revisions/{revision}/comments` serve them, and the
  **Review** tab shows, adds and threads the comments on the open revision. A
  live browser test comments, replies and finds the thread after reopening
  the project.

### The Studio's Fit tab

- **Fit** fits the selected catalogue model or the workspace's candidate
  draft to recordings imported as CSV files, each assigned to training or
  hold-out; it lists each fitted value with its standard error or *not
  stated*, names any unconstrained parameter combination, gives the hold-out
  error per recording and the optimiser's trial counts, and exports and
  replays the result. A live browser test fits a candidate's model, checks
  the recovered values and the replay, and fits the resistance–capacitance
  pair that the model sees only as a ratio and finds it reported
  unconstrained.

### Parameter fitting with held-out validation and identifiability

- `sc_neurocore.fitting` fits the parameters of a Universal DSL model — a
  catalogue model's canonical schema or a candidate's model — to a cohort of
  recordings split into training and hold-out sets; the objective sees the
  training set only, and data shared between the sets is refused.
- Seeded differential evolution with a local polish keeps the best loss of
  every generation and counts failed trials (non-finite state, or a residual
  too large to use) instead of hiding them; a fit with no finite trial does
  not claim convergence.
- The residual Jacobian at the optimum gives an identifiability diagnosis: a
  parameter combination the data do not constrain is reported with its
  direction, and then no standard error is reported. Otherwise standard errors
  and correlations come from the Gauss–Newton asymptotic covariance.
- Results carry the whole problem, versions and one digest, and replay.
  `POST /api/fits` and `POST /api/fits/replay` serve it, bounded to 3 000 000
  estimated model steps per synchronous fit.

### Simulation workbench: the plot in words, named sliders, bounded long series

- The plot canvas names itself for assistive technology: for the trace view, a
  sentence with the variables, the run's length and step, its spike count and
  each variable's range and final value; for other views, what the view is and
  where its numbers are. **Data table** shows the trace's numbers as a table.
  Both come from the server's display projection, which keeps every extreme
  and the last sample.
- Every parameter slider carries its name and unit and states its value; they
  had no accessible name at all, so a screen reader announced only "slider".
- A series with more samples than the plot has pixel columns is stroked through
  its per-column first, lowest, highest and last samples, keeping every visible
  extreme while bounding the path; result data are unchanged.
- A live browser test checks the slider names, the plot description, the data
  table by keyboard, and that the workbench does not scroll sideways at twice
  the zoom.

### Job custody on Python 3.10, and a refusal read instead of a broken pipe

- Purging a finished job failed on Python 3.10: clearing its directory called
  `shutil.rmtree(dir_fd=...)`, which exists from Python 3.11. The directory is
  now cleared by walking it relative to its open descriptor without following
  links, the same on every supported Python.
- When the storage authority refused an artefact frame and closed, a finishing
  API process that still had frames to send could hit a broken pipe, which
  closes the channel with the refusal unread. The client now reads an answer
  already waiting before it sends the next frame, so the refusal is reported.

### Candidate models: author, check and review a proposed model in the Studio

- A candidate package (`sc-neurocore.studio.candidate.v1`) carries one
  Universal DSL model with the unit of every state variable and parameter,
  its source, assumptions, authors, parent catalogue model and proposed
  reference tests. A candidate is never listed as a catalogue model and
  nothing writes a canonical file for it.
- `POST /api/candidates/validate` reports every problem at once, each located
  by a JSON pointer; units are parsed with pint, expressions pass the equation
  safety gate, unknown fields are refused, and the Universal DSL has the last
  word on the model.
- `POST /api/candidates/diff` compares a candidate with its parent's canonical
  schema; equations are converted from their syntax tree to SymPy (never
  evaluated) and reported as unchanged, equivalent, changed with the
  simplified difference, not comparable or undecided.
- `POST /api/candidates/simulate` runs a candidate under its own profile for at
  most 100 000 steps and reports divergence by step; `POST
  /api/candidates/review-packet` runs the reference tests and binds candidate,
  validation, diff, results, environment and what is not established under one
  digest.
- The Studio's **Candidate** tab imports, edits, validates, diffs, simulates and
  reviews a candidate and exports it byte for byte and its review packet; the
  draft is saved with the workspace in its `candidates` block. The `studio`
  extra and the hub requirement set now install `pint` and `sympy`.

### Route policies: ten routes classified, and a missing policy refused

- With route policies enforced, ten routes answered 500: the catalogue query,
  a model's capability matrix, a graph model's contract, and project
  revisions, fork, refused-edit branch, deleted list, restore, export and
  import. None had a registered policy, and looking one up failed inside the
  security middleware. Each now has a policy: the catalogue and graph-model
  reads are public like their neighbours, the project operations require an
  authenticated principal like saving and loading.
- The middleware refuses a route without a policy as `unclassified_route`
  (403) instead of failing.
- The test that every route has a policy walked only the application's top
  level, where FastAPI's included routers now hide every route, so it checked
  nothing. It walks the routes the middleware matches and requires more than a
  hundred of them.

### Network Canvas: the whole graph editable without a pointer

- The table view's rows carry **Edit** beside **Delete**, and each projection
  carries **Edit** and **Delete** at its source row. The editor opens beside
  the table and takes the focus; every control names what it acts on, a
  projection by both ends and what it carries. Before, the table could delete
  a population and connect two, but nothing in it could edit a population or
  touch a projection, and the editors were hidden with the canvas.
- The project list's open, delete and restore entries are buttons that name
  their project. They were clickable text no keyboard could reach, so a saved
  network could not be reopened without a mouse.
- A live browser test builds, connects, edits, deletes and undoes a network by
  keyboard alone, saves it, reloads the page and reopens it by keyboard; another
  drags a node and holds the run's `graph_sha256` unchanged. A backend test
  holds positions out of the resolved specification.
- The canvas guide lists the keyboard shortcuts.

### Identity store lock: unavailable lock named, one file beside the store

- A lock file that cannot be created or opened (a store directory the Studio
  account cannot write, a directory or foreign file at the lock path) now
  refuses startup with `StudioApiLockUnavailable`, naming the lock and the
  reason, instead of a bare SQLite error. Only a held lock is reported as
  another API process.
- The lock keeps its journal in memory, so the only file the API process adds
  beside the identity store is `<store>.api-lock`.

### Release gate: the installed Studio is accepted before publication

- `tools/studio_installed_acceptance.py` launches an installation's Studio
  through its own `sc-neurocore studio` command, after checking that its
  `sc_neurocore` is not the source checkout (the caller's `PYTHONPATH` and
  `PYTHONHOME` are not passed on, so they cannot lend it one), and requires the root redirect to
  the interface, every asset the interface names, and a catalogue listing
  exactly the models the installed registry maps. It stops the server however
  the checks end and reports the first check that failed.
- The publish workflow installs the built wheel into a clean environment
  (hash-pinned dependencies, the wheel without dependency resolution) and runs
  the acceptance before the wheel can reach PyPI. The distribution tests run
  the same acceptance on a wheel built from the source tree.

### The Studio tour names the catalogue it loaded

- The first-run tour's model-browser step states the number of models the
  catalogue delivered to the browser instead of a fixed 118, and states no
  number until the catalogue has arrived.

### Catalogue counts in the documentation follow the catalogue

- The capability manifest now counts the Studio catalogue: every model the
  registry maps, which is what the Studio model browser lists. The README
  inventory gains a "Studio catalogue models" row.
- A documentation page states a count by binding it: the number sits between
  an inline `<!-- count:NAME -->` and `<!-- /count -->` pair naming a manifest
  count. Regenerating the manifest rewrites the number, and `tools/capability_manifest.py --check`
  (already a CI step) fails on a stale bound number or a binding to a count
  the manifest does not have.
- The Studio guide, the Studio quickstart and the system map now bind the
  catalogue count. They had stated 118 and 152 models while the catalogue
  holds 185.

### Studio user interface in the wheel

- A published wheel now carries the built Studio interface in
  `sc_neurocore/studio/frontend_dist/`, so `pip install sc-neurocore[studio]`
  followed by `sc-neurocore studio` opens a working interface without a
  Node.js toolchain. Before this, only a source checkout with a local
  `npm run build` had an interface; an installed Studio served an empty root.
- The interface is mounted at `/studios/sc-neurocore/`, the path every asset of
  the production build names, and the root redirects there. It had been
  mounted at the root, where its assets did not resolve.
- The publish and release workflows build the interface before packaging and
  set `SC_NEUROCORE_STUDIO_UI=required`, so a release wheel without it fails
  to build. A source build without a built interface still succeeds; the
  launcher then says there is no interface and opens the API documentation.
- A rebuild in a reused build directory drops an interface removed from the
  source rather than shipping the stale copy.

### Studio identity store: private file, one serving process

- The identity store is owner-only: a store another account owns is refused,
  one readable by group or others is narrowed to `0600` before it is read (and
  the widening logged), and every write leaves it `0600`. Writes used to keep
  whatever mode the file had.
- Browser sessions and login throttles live in the API process, so one process
  serves an identity store: the first holds it for its lifetime and another API
  process on the same store is refused at startup with the reason, instead of
  keeping its own sessions and its own login-attempt count.

### Studio silicon operations per model

- `GET /api/models/{name}/capabilities` states which silicon operations this
  installation can run for a catalogue model — compile, co-simulate (per
  integrator and format, only where a bit-true kernel mirrors the RTL),
  synthesise, place and route (per target, with the tool that times it) and
  formal — and gives every disabled operation its reason. The model panel
  lists them. Every co-simulation combination enabled for a map model and an
  adaptive Euler model is executed in the test suite and is bit-exact; RK
  profiles compile and have co-simulation disabled by name.

### Studio network to hardware

- The Studio pipeline builds hardware for the network on the Canvas. It used to
  synthesise one hardcoded LIF neuron whatever the graph, and then refused every
  graph. `POST /api/pipeline/run` now validates, simulates, **lowers** the graph
  with each catalogue model's own step (`SCLapicqueLIFNeuron` profile `sc_lif`
  as its exact step, `PerfectIntegratorNeuron` with its profile's threshold
  comparison), **co-simulates** the compiled RTL in Icarus Verilog beside a C
  model built from the compiler's bit-true neuron kernels and beside the
  Studio's own run, and synthesises only when the RTL reproduces its model on
  every step. A graph it cannot reproduce is refused before any hardware with
  every reason: another model, a Poisson drive, a membrane that does not start
  at rest, an unrepresentable or vanishing fixed-point value, a delay over 1024
  steps, one model with two parameter sets. The result's `trace` binds the
  graph digest and format, the RTL, the model and the synthesised source.
  `q_format` selects `Q8.8` (default) or `Q16.16`; the first step at which the
  fixed-point hardware differs from the Studio run is reported.
- The hardware network compiler gains the Studio's catalogue profiles as neuron
  templates (`sc_lif`, `sc_if`); templates may name their integration method.
- Fixed: the NIR hardware-graph lowering dropped a `Delay` that follows a
  `Linear`, so such a file compiled with no delay; it is now composed into the
  connection, and a post-weight delay that differs between destination
  neurons is refused.

### Studio verified readiness in installations

- An installed Studio showed receipt-bound readiness as lost, because receipt
  subjects such as validator tests are not installed: Lapicque read S3 instead
  of S5, and four models read as not enrolled on silicon instead of H1. An
  installation now serves the verification sealed in the checkout it was built
  from (`studio/verified_readiness.json`, `tools/studio_readiness_seal.py`),
  and names the `source` of every verified block (`receipts`, `sealed` or
  `unsealed`). A test holds the seal equal to a fresh derivation; the
  distribution test holds an installed wheel's catalogue, including every
  model's verified tiers, equal to the checkout's.

### Studio catalogue: proven-readiness query and model links

- `GET /api/models/query` filters the catalogue on the server by text, family,
  behaviour, identity kind, metadata state and **verified** readiness
  (`min_verified_science`, `min_verified_silicon`, `verified_perfect_only`),
  and returns the matching models with drill-down facet counts. The readiness
  floors read only tiers bound to fresh facet receipts, never declared tiers.
  Each catalogue entry now carries `is_perfect_verified`.
- The model browser uses that route for search and filtering. Its readiness
  filters were declared-tier filters (one of them labelled "verified" for a
  declared tier 3); they are now "proven" filters on the verified tiers,
  including "perfect, proven", and every filter is a keyboard-reachable
  button with its pressed state announced.
- `#model=<ClassName>` opens one model; **Copy link** in the model panel builds
  it. A link naming a model the catalogue no longer holds says which name
  failed.

### Studio network NIR export and import

- `POST /api/graph/export-nir` now writes a real NIR graph (HDF5, through the
  reference `nir` package of the `nir` extra). It used to return the Studio's
  own JSON graph envelope, whose node types were catalogue model names, under
  the NIR name. `SCLapicqueLIFNeuron` (profile `sc_lif`) maps to `nir.LIF` and
  `PerfectIntegratorNeuron` to `nir.IF` with `r = 1 / c_m`; a network with any
  other model is refused, naming the population. Projections become
  `nir.Linear` with the realised connectivity, followed by `nir.Delay`
  including the runtime's one-step latency. The response
  (`sc-neurocore.studio.nir-export.v1`) carries the file as base64 and notes on
  what NIR does not carry: the drive, the `>=` threshold comparison, and a
  non-resting initial membrane. Run through the NIR bridge, an exported network
  spikes exactly as the Studio runs it.
- `POST /api/graph/import-nir` reads `{"content_base64": ...}`. A file this
  Studio wrote is rebuilt in its original order and refused if any tensor
  differs from the network its metadata describes. A file another tool wrote
  is read where the graph can hold it (uniform LIF and IF populations,
  all-to-all single-weight projections, whole-step delays) and refused by name
  otherwise. The legacy JSON envelope still reads and is reported as
  `studio-envelope`. Refusals answer 422 with `detail.reason`.
- The Network Canvas downloads `network.nir` and gains **Import NIR**
  (`.nir` files, or `.json` envelopes saved by earlier builds). Both show what
  the file did not carry exactly or what the reading assumed.
- Breaking: the export route's response shape changed as above, and
  `sc_neurocore.studio.network_graph.graph_to_nir` / `nir_to_graph` are
  renamed `graph_to_envelope` / `envelope_to_graph`, since they never produced
  or read NIR.

### Go alpha C ABI hygiene

- Removed the pre-existing `unsafe.Pointer` to integer and back conversion in
  alpha-kernel region assembly. Direct pointer indexing retains the exported C
  ABI and behaviour while satisfying whole-module `go vet` and Go tests.

### DPI source-bound complete-state reclosure

- Re-audited `DPINeuron` directly against Indiveri, Stefanini, and Chicca
  (2010), equations (2)-(3), plus the described threshold feedback, reset,
  refractory pulse, and spike-driven adaptation. Normalised defaults,
  `I_in=i_rest+current`, explicit Euler, sampled event ordering, and the
  refractory counter remain explicitly maintained choices.
- Added failure-atomic, aligned membrane/adaptation/refractory/event packets
  across Python, production and safety Rust, PyO3, Julia, Go, Mojo, and both C
  ABIs. Every lane carries the configurable 18-field contract; explicit native
  failures leave caller-visible state and output buffers untouched.
- Added a primary-PDF-bound independent 5,000-step receipt, complete state and
  event digests, NetworkRunner execution, and a controlled five-runtime
  benchmark that hashes the runtime, ABI, schema, descriptor, receipt, RTL,
  formal, synthesis, and owning-test surfaces.
- Preserved the 13-event Q16.16 co-simulation envelope, completed 112,953-cell
  Yosys coarse synthesis, and strengthened the formal lane to a depth-8
  deterministic protocol that reaches a spike and checks its public reset and
  following refractory packet. This closes H2 only; timing, PPA, device,
  board, transistor mismatch, physical silicon, and universal formal
  equivalence remain open.

### Theta source-bound complete-packet reclosure

- Re-audited `ThetaNeuron` directly against Ermentrout and Kopell (1986),
  equation (2.5), the frozen-drive reduction in equation (3.3), and the stated
  spike passage through `theta=pi`. The catalogue implements the constant-
  parameter Type-I normal form; it does not claim the paper's full coupled
  slow-oscillator parabolic-bursting system.
- Added failure-atomic, sample-aligned phase/event packets across Python,
  production and safety Rust, PyO3, Julia, Go, Mojo, and both native C ABIs.
  Arbitrary finite initial phase and timestep now reach every lane, while a
  held step capable of more than one complete rotation is rejected rather than
  silently compressed into one binary event.
- Added an author-PDF-bound source receipt with independent circle-flow oracle,
  complete input/phase/event digests, and explicit solver/source boundary. The
  controlled five-runtime benchmark now hashes all runtime, ABI, receipt,
  descriptor, schema, RTL, formal, synthesis, and readiness inputs and records
  exact event vectors; production Rust reaches 12.06x the Python reference in
  the pinned local diagnostic run.
- Regenerated the source-labelled Q16.16 Euler RTL and passed the complete
  co-simulation vector, declared phase/timing envelopes, 6,203-cell Yosys
  coarse synthesis, and depth-110 receipt-drive phase/event safety. This closes H2 only;
  timing, PPA, device, board, physical silicon, and universal exact-flow
  equivalence remain open.

### Quadratic IF source-normalisation correction and dual-profile reclosure

- Restored Latham et al.'s 2000 isolated scalar QIF identity from equations
  (1), (2), and (5a). Their numerical `V_r/V_t/V_apex/V_repol/tau/dt` values
  normalize to `-1/+1/(31/3)/-3/.05`; the unstable `+1` threshold is no longer
  mislabeled as the source spike apex. The former `-1/+1/.01` recurrence remains
  fully available as count-neutral `SCSymmetricQuadraticIFNeuron` and through
  the legacy zero-argument constructor.
- Added failure-atomic complete voltage/event packets across Python, production
  and safety Rust, Julia, Go, Mojo, PyO3, and both native C ABIs. Paired
  source/SC schemas, explicit NetworkRunner routing, an independent DOI/PDF-
  bound receipt, and a source-hashed controlled benchmark retain complete
  parameter, state, event, final-state, and identity custody.
- Added dedicated source Q16.16 Euler RTL using the paper's numerical timestep
  and finite apex/reset boundaries. It is cycle-exact against the source schema
  at `eta=0/2/4/8`, explicitly records the `eta=1` one-cycle quantization
  boundary, synthesizes to 9,379 Yosys coarse cells, and passes depth-20
  reset/event safety. The original `sc_quadratic_if` RTL remains assigned to
  the SC profile; timing, PPA, device, board, physical silicon, and universal
  real-valued equivalence remain open.

### Perfect Integrator source-boundary correction and dual-profile reclosure

- Restored Naud and Gerstner's 2012 perfect-integrator equation with its
  written strict `V > V_T` reset boundary and exact piecewise-constant-current
  integral. The former inclusive comparator remains fully available through
  the legacy zero-argument constructor and the explicit count-neutral
  `SCInclusivePerfectIntegratorNeuron`; no SC schema or RTL variant was removed.
- Added failure-atomic complete voltage/event packets across Python,
  production and safety Rust, Julia, Go, Mojo, PyO3, and both native C ABIs.
  Canonical NetworkRunner aliases now select the source profile, explicit SC
  aliases select the retained profile, and an independent DOI-bound receipt
  pins the equality-sensitive voltage and event-vector digests.
- Added dedicated rational-`dt` Q8.8 source RTL that preserves the strict
  equality protocol exactly, synthesizes to 3,034 Yosys coarse cells, and
  passes depth-20 reset/event safety. The source-hashed five-runtime benchmark
  records 33,333 bit-exact events over 100,000 steps; timing, PPA, device,
  board, physical silicon, and universal fixed-point equivalence remain open.

### Lapicque 1907 source-identity correction and dual-profile reclosure

- Corrected the catalogue identity to Lapicque's leaky-capacitor polarization
  experiment and strength-duration law. The source profile emits one latched
  threshold-attainment event and does not invent an automatic reset or
  repetitive spike generator. The previous exact-flow hard-reset recurrence
  remains fully available as count-neutral `SCLapicqueLIFNeuron` and through
  the legacy zero-argument constructor.
- Added failure-atomic complete polarization/event packets across Python,
  production and safety Rust, Julia, Go, Mojo, PyO3, and both native C ABIs.
  Paired source/SC schemas, explicit NetworkRunner routing, an independent
  source receipt, and a controlled source-hashed five-runtime benchmark retain
  the complete parameter, state, event, and identity boundary.
- Added a dedicated source-specialized Q32.32 core: four-drive co-simulation
  preserves exact event vectors with state error below `7e-8`, Yosys reports
  11,511 coarse cells, and depth-20 Z3 proves reset, permanent latch, and no
  repeated post-latch event. The historical SC Q16.16 core and evidence remain
  separate and intact.

### ExpIF dual-profile full-fidelity reclosure

- Restored the Fourcaud-Trocmé fitted source protocol as an explicit factory:
  deterministic zero-noise Heun RK2 below the paper's -30 mV numerical
  handoff, the derived analytical exponential-only tail duration, and the fitted 1.7 ms
  refractory interval. The former +30 mV candidate-first RK4 behavior remains
  exactly available as the separately declared SC compatibility profile.
- Added failure-atomic complete voltage/refractory/event packets across Python,
  production and safety Rust, Julia, Go, Mojo, and both native batch ABIs.
  Paired schemas, direct NetworkRunner coverage, an independent 20,000-step
  source receipt, and a controlled source-hashed five-runtime benchmark retain
  the complete numerical and event contract.
- Promoted the compatibility RTL boundary to honest H2: the Q32.32 packet is
  bound to complete event/state digests and preserves its enrolled aggregate
  event count, the tracked core synthesizes to 484,938 Yosys coarse cells, and
  depth-4 bounded checks prove reset and public event/reset safety. Timing, PPA, device, board,
  physical silicon, and universal fixed-point equivalence remain open.

### AdEx complete-state full-fidelity reclosure

- Separated the Brette–Gerstner 2005 regular-spiking fit from the maintained
  compatibility defaults and bound the as-published equations, exact fit
  parameters, three-event source trajectory, and complete-state digests in an
  independent receipt.
- Added failure-atomic complete-state/event batches across Python, production
  and safety Rust, Julia, Go, Mojo, and both native C ABIs. Every compiled lane
  now accepts the full maintained numeric contract; the source-hashed benchmark
  retains both state traces, exact event vectors, final state, and complete
  packet digests.
- Promoted the honest hardware boundary to H2: Q16.16 co-simulation preserves
  exact event counts across four drives, the committed design synthesizes to
  52,014 Yosys coarse cells, and depth-6 Z3 proves exact reset plus public
  event/reset safety. Timing, PPA, device, physical silicon, and universal
  numerical equivalence remain open.

### Chialvo map complete-state full-fidelity reclosure

- Bound the DOI-sourced Eq. 1 recurrence to an independent 1,000-step receipt
  containing complete binary64 `(x,y)` and maintained event digests; the
  threshold crossing remains explicitly separated from the paper's dynamics.
- Added failure-atomic complete-state batches across Python, production and
  safety Rust, Julia, Go, Mojo, and both native C ABIs. PyO3 exports both state
  traces, NetworkRunner directly exercises the Chialvo identity, and the
  source-hashed five-runtime benchmark now retains state, event, final-state,
  and complete output-packet custody.
- Promoted the honest hardware boundary to H2: Q16.16 co-simulation retains
  its declared event/state envelope, tracked Q8.8 RTL synthesizes to 117 coarse
  Yosys cells, and depth-4 Z3 proves exact reset and public upward-crossing
  event safety. Timing, PPA, device, physical-silicon, and universal numerical
  equivalence remain open.

### Ibarz-Tanaka analysis-profile full-fidelity reclosure

- Preserved the public `IbarzTanakaMapNeuron` compatibility identity while
  correcting provenance: Shilnikov and Rulkov (2004) introduced the
  four-branch recurrence, and Ibarz et al. (2007) restated and analysed the
  maintained parameter profile. A paired-source receipt now binds every
  `(v,u)` state and reset decision over the 1,000-step reference protocol.
- Corrected a branch-order bug that could report a false reset event under a
  sufficiently low upper guard while the earlier `v<=0` branch executed.
  Python, production and safety Rust, Julia, Go, and Mojo now agree on actual
  fourth-branch execution; stateful and C-ABI batches validate completely
  before caller-state or output mutation, and NetworkRunner has direct identity
  coverage.
- Re-ran the source-hashed five-runtime benchmark with complete trace/output
  packet custody. Two Q16.16 co-simulation protocols now genuinely cover all
  four branches, tracked RTL passes 33-cell Yosys coarse synthesis, and a
  depth-4 Z3 proof establishes exact ordered event logic and
  `event => v_next=-1`. The descriptor advances from H1 to its honest H2
  terminal boundary; timing and higher silicon claims remain open.

### Wilson-HR dual-identity full-fidelity closure

- Restored `WilsonHRNeuron` to Wilson's `C=0.8` continuous polynomial flow and
  separated the repository's sampled event observation from the source
  dynamics. The former unit-capacitance hard-reset recurrence remains intact as
  count-neutral `SCResettingWilsonHRNeuron`, without publication attribution.
- Completed both identities through Python, production and safety Rust, Julia,
  Go, Mojo, PyO3, NetworkRunner, paired schemas, executable source-bound
  benchmarks, Q16.16 co-simulation, Yosys synthesis, bounded formal safety, CI
  backend construction, and native/public documentation. The existing source
  catalogue count remains unchanged.

### Hindmarsh-Rose full-fidelity reclosure

- Rebound Model 7 descriptor validation and co-simulation custody to collected
  tests, completed state semantics and source derivation metadata, and promoted
  the bounded RTL packet to H2 with an executable Yosys synthesis receipt.
- Made Python, Rust-engine, Go, Julia, and Mojo batch execution fail explicitly
  before Python object state commits when an RK4 stage becomes non-finite.
- Bound the five-runtime benchmark to implementation, descriptor, paired
  schemas, independent receipt, and native sources; enrolled it in the
  fail-closed benchmark gate and replaced the public temporary evidence marker
  with durable artefact paths.

### Descriptor and benchmark evidence fix-forward

- Completed unit, valid-range, and semantic custody for every source and
  retained-SC descriptor parameter, replaced ignored legacy reproducibility
  keys with active receipt digests, and enrolled class-correct trajectory
  validation without attributing the SC recurrence to the publication.
- Bound the dual-identity five-runtime benchmark to both descriptors, paired
  schemas, receipts, and runtimes; enrolled it in the fail-closed benchmark
  evidence gate and replaced the public `this commit` marker with durable
  artifact paths. No RTL or silicon claim was added.
- Completed the 15 missing valid ranges in the retained SC Wang/NMDA descriptor
  and enrolled its existing dual-identity five-runtime benchmark, paired
  schemas, and receipts in the same fail-closed evidence gate.

### McKean dual-identity full-fidelity closure

- Re-certified the catalogue-counted Tonnelier/McKean 2003 space-clamped
  Heaviside system without altering the separate count-neutral
  `SCTriangularMcKeanNeuron` recurrence. Corrected the journal provenance,
  parameter envelopes, failure-atomic Python validation, durable public
  evidence anchors, and source-bound five-runtime benchmark packets; the SC
  identity now also carries a SHA-256-locked 3,000-row project receipt.
- Routed `McKeanNeuron` exclusively to its curated signed-Q32.32 formal lane so
  the catalogue emitter cannot create a competing Q8.8 identity. Both retained
  RTL cores preserve their enrolled event/state traces, pass Yosys coarse
  synthesis, and pass bounded reset-safety BMC at the honestly declared H2
  terminal tier; timing, PPA, device, and universal equivalence remain open.

### Hodgkin-Huxley full-fidelity closure

- Recorded the original rest-relative to modern absolute-voltage coordinate
  transformation and separated the maintained default gate-first Euler
  production profile from the simultaneous-RK4 schema/compiler profile. Added
  a replayable complete four-state/event receipt, failure-atomic Python
  boundaries, and source-bound execution across Python, production Rust,
  Julia, Go, and Mojo with independent safety Rust custody.
- Replaced the stale committed Q8.8 catalogue object with the current
  signed-Q16.16 RK4 lowering used by bounded co-simulation. The committed
  object retains the honest one-event LUT-limited envelope, passes Yosys coarse
  synthesis, and passes depth-4 reset-safety BMC under the enrolled receipt
  protocol without claiming unrestricted numerical equivalence or silicon.

### Connor-Stevens full-fidelity closure

- Corrected direct provenance from the 1971 experimental lineage to the exact
  Connor-Walter-McKown (1977) Appendix A parameterization while preserving the
  established Connor-Stevens catalogue identity. Added a replayable complete
  six-state/event receipt and expanded the source-bound packet from three
  executed runtimes to Python, production Rust, Julia, Go, and Mojo with exact
  event custody and explicit non-production measurement boundaries.
- Replaced the stale committed Q8.8 catalogue object with the current
  compiler's signed-Q16.16 lowering used by the behavioral evidence. The
  committed object retains the honest one-event LUT-limited bounded envelope,
  passes Yosys coarse synthesis, and passes depth-4 public-port reset-safety
  BMC. This reaches H2 without claiming unrestricted numerical equivalence,
  timing, PPA, a target device, or physical silicon.

### FitzHugh-Nagumo full-fidelity closure

- Reconciled the maintained two-state recurrence with FitzHugh's (1961)
  equations (1)-(3) under the explicit coordinate/time transformation and kept
  RK4, the parameter profile, and sampled upward crossings clearly identified
  as repository numerical specializations. Added a replayable 3,000-step
  complete-state/event receipt and a source-hashed 2,000,000-step packet across
  Python, production Rust, Julia, Go, and Mojo; standalone safety Rust and the
  Go service remain independently exercised.
- Replaced the stale committed Q8.8 catalogue object with the current
  compiler's signed-Q16.16 lowering used by the behavioral evidence. The
  committed object preserves all eight receipt events under Icarus/VVP, passes
  Yosys coarse synthesis, and passes depth-4 public-port reset-safety BMC. This
  reaches the declared H2 terminal tier; timing, PPA, device, formal real-number
  equivalence, and physical silicon remain explicitly open.

### Morris-Lecar full-fidelity closure

- Bound the existing Morris-Lecar source identity to a replayable 3,000-step
  full-state/event receipt and its executed source-hashed 200,000-step packet
  across Python, production Rust, Go, Julia, and Mojo. The descriptor and public
  model page now expose the production engine, safety Rust, Go, Julia, and Mojo
  truth instead of reporting only Python/Rust.
- Replaced the stale committed Q8.8 catalogue object with the current compiler's
  signed-Q16.16 lowering used by the behavioral evidence. The committed object
  preserves all seven receipt events under Icarus/VVP, passes Yosys coarse
  synthesis, and passes depth-4 public-port reset-safety BMC under the explicit
  initial-reset and fixed-current receipt protocol. This reaches the declared H2
  terminal tier; timing, PPA, device, formal equation equivalence, and physical
  silicon remain explicitly open.

### Wang-Buzsaki full-fidelity closure

- Closed the existing Wang-Buzsaki (1996) source identity through a replayable
  mixed-drive receipt and a source-hashed 20,000-step measured packet across
  Python, production Rust, Go, Julia, and Mojo. The descriptor now records the
  complete runtime, reproducibility, validation, and silicon boundary; the
  catalogue remains 60/155 polyglot-complete because this model was already
  counted by the strict runtime inventory.
- Committed the current equation compiler's signed-Q16.16 Gauss-Seidel RTL,
  bounded event-count co-simulation, Yosys synthesis gate, and depth-4
  public-port reset-safety proof. This reaches the declared H2 terminal tier;
  timing, PPA, device, universal real-number equivalence, and silicon claims
  remain explicitly open.

### Wang NMDA-autapse dual-identity closure

- Restored `NMDANeuron` as the Wang (1999) pyramidal LIF plus two-stage
  saturating NMDA-autapse equations, including the Jahr–Stevens magnesium
  block. Midpoint RK2 at `0.05 ms` with sampled threshold handling is the
  declared scalar implementation specialization.
- Retained the former WB membrane plus input-driven NMDA project recurrence
  count-neutrally as `SCWBNMDAMagnesiumBlockNeuron`. Both identities have
  separate descriptors, paired schemas, independent receipts, measured
  behavior, Python/Rust/Go/Julia/Mojo custody, production Rust bindings, and a
  source-hashed benchmark. Source-default signed-Q16.16 RTL is bit-exact to an
  independent integer oracle, preserves the four-event receipt and bounded
  state envelopes, synthesizes in Yosys, and passes depth-4 CVC5 bounded
  safety. The retained identity has its own 50-cycle signed-Q32.32 FSM,
  bit-exact integer oracle, exact enrolled quiet/spiking event vectors, Yosys
  coarse synthesis, and bounded handshake/output proof. The catalogue is now
  60/155 polyglot-complete; network, interpolated spike-time, configurable-
  parameter RTL, SC technology gate mapping, timing, PPA, device, and higher-
  silicon claims remain outside scope.

### Larter-Breakspear dual-identity closure

- Restored `LarterBreakspearNeuron` to the source cortical population equations,
  including excitatory and inhibitory firing-rate feedback, NMDA modulation,
  external coupling balance, and complete three-state RK4 dynamics.
- Retained the former decoupled-adaptation recurrence count-neutrally as
  `SCDecoupledAdaptationIonMassNeuron`. Both identities have independent
  receipts, paired schemas, Python/Rust/Go/Julia/Mojo custody, production Rust
  bindings, and source-hashed benchmark evidence. The catalogue is now 59/155
  complete; no RTL or higher-silicon claim is made.

### Butera Model 1 dual-identity closure

- Restored `ButeraRespiratoryNeuron` as the Butera-Rinzel-Smith (1999)
  three-state Model 1 current balance with the source `C = 21 pF` membrane
  capacitance, persistent-sodium inactivation, optional tonic conductance, and
  observational spike crossing without reset. The repository's candidate-first
  RK4 at `0.1 ms` is an explicit numerical specialization of the continuous
  source equations.
- Retained the former omitted-capacitance recurrence as count-neutral
  `SCUnitCapacitanceRespiratoryNeuron` without paper attribution. It has its own
  public registration, descriptor, measured behavior, paired schemas,
  project-spec receipt, native Python/Rust/Go/Julia/Mojo custody, and
  five-runtime benchmark. The source identity preserves 954 exact events over
  200,000 steps in every runtime. The numerically sensitive SC profile binds
  one-step state parity and an explicit 4–5-event native envelope over 20,000
  steps rather than claiming false long-run bit identity. The catalogue is now
  58/155 complete; RTL and higher silicon rungs remain unclaimed.

### Hill-Tononi dual-identity closure

- Restored `HillTononiNeuron` as the Hill and Tononi (2005) hybrid model
  neuron using the cortical-excitatory waking profile, dynamic threshold,
  depolarisation-dependent potassium current, finite post-spike pulse, and
  source `0.25 ms` RK4 integration. The former six-state HH/Na-pump project
  recurrence remains count-neutrally as `SCSixStateThalamocorticalNeuron`
  without paper attribution.
- Closed the source identity through Python, Rust safety and engine, Julia,
  Go, Mojo, paired schemas, an independent mixed-drive receipt, native parity,
  and a source-bound five-runtime benchmark. The source catalogue is now
  57/155 complete; full-network and silicon evidence remain explicitly
  unclaimed.

### Bertram phantom dual-identity closure

- Restored `BertramPhantomBurster` as the four-state Bertram et al. (2000)
  source system with dynamic `n`, equations 1–10, and the authors' `BJ_00`
  defaults. The former instantaneous-`n_inf` project recurrence remains
  count-neutrally as `SCThreeStatePhantomBurster` without paper attribution.
- Closed source and compatibility identities through Python, Rust safety and
  engine, Julia, Go, Mojo, paired schemas, independent receipts, native parity,
  and a source-bound five-language benchmark. The source catalogue is now
  56/155 complete.

### Descriptor corpus alias and inherited-drift repair

- Made descriptor generation idempotent by traversing unique canonical model
  identities, preventing the `KilincBhattMapNeuron` compatibility alias from
  overwriting `SCAdaptiveThresholdMapNeuron` metadata.
- Regenerated the inherited drifted corpus and completed the missing parameter
  units, ranges, and meanings exposed by the prior dual-identity closures.

### Benda-Herz dual-identity closure

- Restored `BendaHerzNeuron` as the deterministic universal rate-adaptation
  and phase-generator source identity, and retained the former stochastic
  project recurrence as explicit, count-neutral
  `SCStochasticRateAdaptationNeuron` without paper attribution.
- Closed both contracts through the five runtime surfaces, paired schemas,
  receipts, RTL/Yosys/bounded formal evidence, focused benchmarks,
  NetworkRunner, and language-native documentation. The source catalogue is
  now 55/155 complete.

### McKean dual-identity closure

- Restored `McKeanNeuron` as the source McKean/Tonnelier space-clamped
  Heaviside system and retained the former three-branch project recurrence as
  explicit, count-neutral `SCTriangularMcKeanNeuron` without paper attribution.
- Closed both identities through five runtimes, NetworkRunner, paired schemas,
  independent receipts, signed-Q32.32 RTL/Yosys/bounded formal evidence, local
  source/binary-bound benchmarks, and language-native documentation. The source
  catalogue is now 54/155 complete.

### EnergyLIF dual-identity closure

- Restored `EnergyLIFNeuron` as the Fardet-Levina coupled voltage-energy RK4
  source identity and retained the former exact-flow normalized recurrence as
  the explicit, count-neutral `SCNormalizedEnergyLIFNeuron` project identity.
- Closed both contracts through five runtimes, paired schemas, independent
  receipts, Q32.32 RTL/Yosys/bounded safety, local source/binary-bound
  benchmarks, and language-native documentation. The source catalogue is now
  53/155 complete.

### Sigma-delta dual-identity closure

- Restored `SigmaDeltaNeuron` as the sampled Yoon APSDM source identity and
  retained the former bipolar one-quantum recurrence as the explicit,
  count-neutral `SCSigmaDeltaAccumulatorNeuron` project identity.
- Closed both contracts through five runtimes, paired schemas, independent
  receipts, Q32.32 RTL/Yosys/CVC5, local source/binary-bound benchmarks, and
  language-native documentation. The source catalogue is now 52/155 complete.

### Non-resetting LIF dual-identity closure

- Corrected `NonResettingLIFNeuron` to source MAT(1) with non-resetting
  forward-Euler voltage, one exact 50 ms threshold history, and a 2 ms
  absolute refractory interval. Preserved the former exact-relaxation behavior
  as `SCNonResettingAdaptiveLIFNeuron` rather than overwriting it.
- Closed both identities across Python, Rust engine/PyO3 and safety, Julia, Go,
  Mojo, paired schemas, independent receipts, Q32.32 RTL/Yosys/formal checks,
  source/binary-bound benchmarks, and native documentation. The source count is
  now 51/155; the SC compatibility identity is not counted as literature.

- Completed Aihara Model 43 from primary Eqs. 10–12 after separating the
  original SC two-state recurrence into the preserved `SCChaoticMapNeuron`:
  one-state source dynamics,
  Eq. 12 level events, all maintained language backends, paired schemas,
  independent trace, Q8.24 bounded co-simulation, depth-6 formal proof, and a
  pinned Figure 4 periodic benchmark. The preserved SC model independently
  has measured Python/Rust/Julia/Go/Mojo trajectory and event parity. The
  public polyglot-complete count is 43.

### Native engine environment fidelity

- Made the checkout-shadow loading regression exercise both clean editable
  `maturin develop` layouts and installed-wheel layouts. A stale engine package
  left outside the active distribution record can no longer be mistaken for a
  required part of the supported editable installation.

### Adaptive-threshold IF source-to-co-simulation closure

- Reconstructed `AdaptiveThresholdIFNeuron` as an explicitly composite
  reduced adaptive-threshold LIF: exact constant-input LIF membrane
  relaxation, the Mihalas–Niebur (2009) threshold equation at zero voltage
  coupling, and the Platkiewicz–Brette (2010) fixed post-spike threshold
  shift. Corrected the "Platkiewicz & Bhatt" typographical attribution and
  the descriptor's `integration.method = "euler"` mismatch; the
  voltage-dependent threshold equilibrium, voltage coupling, and adaptation
  current are recorded as the reduction boundary. Defaults are documented as
  catalogue/model-family choices.
- Closed the complete contract end to end: validating atomic Python model
  with a five-runtime measured-order dispatch; modular Rust engine and PyO3
  batch binding; standalone Rust safety mirror; Julia, Go, and Mojo kernels
  with real C ABIs; paired TOML/JSON exact-map schemas; an independent DOI
  reference trace; Q32.32 co-simulation at the enrolled grid-exact operating
  point (complete event vector identical, state error below `1.22e-8`); a
  depth-4 Z3 reset-safety job; a source/binary-hashed five-runtime
  200,000-step benchmark; and descriptor, catalogue, fidelity, validation,
  and generated documentation surfaces.

### Alpha-synapse LIF source-to-co-simulation closure

- Reconstructed `AlphaNeuron` from the Rall (1967) alpha kernel and the
  Gerstner-Kistler (2002) LIF framing, correcting the public Rall (1962)
  misattribution. The five-state exact piecewise-constant-input map now runs
  consistently through Python, modular Rust/PyO3, standalone Rust safety,
  Julia, Go and Mojo, with somatic-only reset and atomic failure semantics.
- Added paired TOML/JSON schemas, an independent dual-drive reference trace,
  Q32.32 generated-RTL co-simulation with the full event vector preserved at
  both enrolled inhibitory levels, a depth-4 reset-safety job, and an honest
  source/binary-bound 200,000-step five-runtime benchmark.

### Wilson-Cowan generated-RTL co-simulation

- Added paired-schema and generated Q32.32 Verilog trajectory co-simulation
  for `WilsonCowanUnit`. The 96-sample mixed-drive RK4 trace keeps both public
  E/I rates within `0.021` absolute error and inside the declared state
  envelope while the event output remains silent. Lookup-table quantisation
  and the absence of formal, synthesis, timing, device, or PPA evidence are
  explicit.

### Threshold-linear-rate generated-RTL co-simulation

- Added paired-schema and generated Q16.16 Verilog co-simulation for
  `ThresholdLinearRateNeuron`. The configured subtract-compare-multiply
  datapath preserves all 193 public rate words cycle-exactly across below,
  equality, and above-threshold inputs and keeps the event output silent. The
  absence of formal, synthesis, timing, device, or PPA evidence is explicit.

### Sigmoid-rate generated-RTL co-simulation

- Added paired-schema and generated Q32.32 Verilog trajectory co-simulation
  for `SigmoidRateNeuron`. The 256-step sign-changing drive stays within
  `0.016` absolute rate error through the public RTL outputs, keeps every rate
  in `[0, 1]`, and keeps the event output silent. The lookup-table
  quantisation and the absence of formal, synthesis, timing, device, or PPA
  evidence are explicit.

### Resonate-and-fire source-to-co-simulation closure

- Reconstructed the Izhikevich (2001) complex resonator across Python, the
  modular Rust engine, independent Rust safety, Julia, Go C-shared, and Mojo
  shared-library execution. Every lane preserves the current-like `x`,
  voltage-like `y`, exact constant-input flow, sampled upward voltage crossing,
  and generalised source reset `z=i*threshold`.
- Added public atomic batch dispatch, paired schemas, an independently derived
  DOI trace, configured five-runtime trajectory parity, native invalid-contract
  tests, source/binary-bound local benchmark evidence, bounded generated
  Q32.32 H1 co-simulation, and a depth-4 reset-safety job. Sampled event timing,
  non-monotone high-drive behaviour, the fixed lookup-grid envelope, and the
  absence of formal equivalence or higher-silicon evidence are explicit.

### Montbrió–Pazó–Roxin source-to-co-simulation closure

- Corrected the legacy `ErmentroutKopellPopulation` attribution and promoted
  the actual Montbrió, Pazó, and Roxin (2015) equations (12a–b) through Python,
  the modular Rust engine, independent Rust safety, Julia, Go C-shared, and
  Mojo shared-library execution. All lanes restore the maintained variables
  through `R=tau*r` and `t'=t/tau` and use one simultaneous two-state Euler
  update.
- Added public atomic batch dispatch, paired schemas, an independently derived
  DOI-pinned trace, configured five-runtime trajectory parity, native invalid-
  contract tests, source/binary-bound local benchmark evidence, bounded
  generated Q32.32 co-simulation, and a depth-4 reset-and-event-silence
  catalogue job. The compatibility name, continuous-rate output, Euler solver
  boundary, and absence of formal-equivalence, higher-silicon, or
  production-speed evidence are explicit.

### Jansen–Rit source-to-co-simulation closure

- Promoted the Jansen and Rit (1995) equation-(6) neural mass through Python,
  the modular Rust engine, independent Rust safety, Julia, Go C-shared, and
  Mojo shared-library execution. All lanes use the published
  `C2*S(C1*y0)`/`C4*S(C3*y0)` connectivity placement and simultaneous six-state
  explicit-Euler update.
- Added public atomic batch dispatch, paired schemas, an independent DOI and
  Brian2-commit-pinned trace, configured five-runtime trajectory parity,
  native invalid-contract tests, source/binary-bound local benchmark evidence,
  and bounded generated Q32.32 co-simulation. The default solver step is
  documented as implementation scope; no higher silicon or production-speed
  result is claimed.

### Wong-Wang source-to-co-simulation closure

- Promoted the Wong and Wang (2006) Appendix decision circuit through Python,
  the modular Rust engine, independent Rust safety, Julia, Go C-shared, and
  Mojo shared-library execution. All lanes use pre-update firing rates,
  simultaneous explicit-Euler NMDA gating, AMPA Ornstein-Uhlenbeck current
  states, and caller-supplied Gaussian samples.
- Added public atomic batch dispatch, paired serial-input schemas, an
  independent DOI and author-code-pinned trace, configured five-runtime
  trajectory parity, native invalid-buffer tests, source/binary-bound local
  benchmark evidence, and bounded generated Q32.32 co-simulation. The paper's
  0.1 ms timestep is retained and the author-script 0.5 ms discrepancy is
  explicit; no higher silicon or production-speed result is claimed.

### Wilson-Cowan polyglot closure

- Promoted the declared normalised Wilson-Cowan coupled E/I equations through
  Python, the modular Rust engine, independent Rust safety, Julia, Go C-shared,
  and Mojo shared-library execution. Every runtime advances both continuous
  rates with the same shifted-sigmoid, candidate-first RK4 contract.
- Added public atomic batch dispatch, paired TOML/JSON schemas, bounded
  complete-trajectory parity, configuration-preserving reset, native invalid-
  contract buffer tests, and source/binary-bound local benchmark evidence.
  Availability/refractory factors and independent inhibitory external drive
  remain explicitly outside this reduction; no spike, RTL, or hardware claim
  is made.

### Schema and descriptor contract reconciliation

- Unified schema validation and execution around the same exact event-only
  predicate: empty state and dynamics are valid only for a non-empty
  deterministic level threshold or a Poisson probability expression. Added
  paired JSON schemas for both rate models and moved the live schema inventory
  baseline to 38 unique stems.
- Bound descriptor generation to the canonical schema/module alias table,
  public stochastic replay state, and the semantic integration labels for
  state-free maps and Poisson intervals. Regenerated only the nine descriptors
  affected by those rules and serializer normalisation.
- Reworked the Tier-3 evidence runner to retain integer currents, apply declared
  constructor configuration, accept both spiking tuples and continuous-rate
  arrays, and verify explicitly declared source-text digest encodings. This
  keeps IQIF and McCulloch-Pitts source-oracle hashes distinct from ordinary
  float-trace hashes while exercising every model's maintained public API.

### Threshold-linear rate polyglot closure

- Promoted `ThresholdLinearRateNeuron` to the explicit algebraic contract
  `r = gain * max(0, I - theta)` across the Python reference, modular Rust
  engine, independent Rust safety, Julia, Go C-shared, and Mojo shared-library
  lanes. The cached rate is not integrated state and is never counted as a
  binary spike.
- Added real five-runtime build/loading and bit-exact parity tests, failure-
  atomic native buffers, configuration-preserving reset, a map schema, curated
  Gerstner-et-al. provenance, and source/binary-bound local benchmark evidence.
  No production-speed, fixed-point RTL, or hardware result is claimed.

### Sigmoid-rate polyglot closure

- Added a configurable `SigmoidRateNeuron.simulate` dispatcher across Python,
  the modular Rust-engine batch, Julia, generated Go C-shared ABI, and exported
  Mojo C ABI. The independent Rust-safety reset now restores only the dynamic
  rate and preserves `tau`, `beta`, `theta`, and `dt`.
- Added executed complete-trace parity, empty/large-step/invalid-contract
  boundaries, atomic native-output tests, and a source- and binary-hashed
  five-backend benchmark. The local timings are non-exclusive regression
  evidence and make no production, cross-host, hardware, or universal ranking
  claim.
- Clarified the scientific boundary: Wilson and Cowan (1972) motivate the
  population-rate sigmoid motif, while this class is a reduced scalar
  relaxation and does not claim the paper's full coupled excitatory/inhibitory
  system.

### McCulloch-Pitts source-to-silicon closure

- Replaced the later real-weighted threshold abstraction with McCulloch and
  Pitts' 1943 all-or-none rule: a fixed positive count of simultaneously
  active excitatory afferents is required, while any active inhibitory
  afferent absolutely vetoes output. One synaptic delay remains a network
  scheduling boundary; the formal cell has no evolving membrane state.
- Added exact Python, Rust-engine, independent Rust-safety, Julia, Go, and Mojo
  varying-input execution with strict signed-ABI validation and atomic native
  failures. Paired stateless TOML/JSON schemas and an independent primary-paper
  truth table preserve the same OR, AND, equality, maximum-count, and
  inhibitory-veto rows.
- Kept the new PyO3 batch implementation in its per-neuron binding module and
  registered it through the existing neuron registry, leaving the Rust crate
  root unchanged relative to the accepted parent.
- Added registered and folded signed-Q32.0 Python-to-Verilog co-simulation and
  a depth-4 Z3 catalogue safety job. The signed value `-1` is reserved as the
  hardware inhibition sentinel; non-negative values carry excitatory counts.
- Added a source- and binary-hashed five-backend benchmark. Its timings are
  single-logical-CPU, non-exclusive local regression evidence, not production,
  cross-host, hardware, or universal performance claims.

### Predictive-model responsibility modularisation

- Replaced the 1,018-line mixed linear-Gaussian implementation with a 63-line
  historical facade over six bounded responsibility modules for contracts,
  native dispatch, filtering, smoothing, controlled EM, and planning-facing
  forecasts. Public imports, class identities, constructor names, and pickle
  paths remain stable.
- Corrected controlled EM sufficient statistics, lag-one covariance
  orientation, covariance validation, and fail-closed native result shapes.
  Filtering uses Cholesky solves and Joseph covariance updates without explicit
  matrix inverses. Every M-step candidate is likelihood-checked before
  acceptance, including the candidate produced by the final allowed iteration.
  Architecture contracts pin exclusive ownership, an acyclic import graph, the
  definition-free facade, and per-module size ceilings.
- Removed an out-of-bounds Mojo gain-workspace read for observation dimensions
  wider than the latent state. A real `d=1, p=3` parity case now exercises
  Python, Rust, Julia, Go, and Mojo through the maintained loading path.
- The source- and binary-bound five-backend artifact records 25 interleaved
  samples, raw timing distributions, parity deltas, runtime versions, hashes,
  affinity, governor, load, and an explicit loaded-host disclosure. It is local
  regression evidence, not a cross-host or exclusive-core performance claim.

### IQIF source-to-silicon closure

- Replaced the unsupported square-law approximation with the exact signed
  integer recurrence in the Wu et al. (2021) coauthor implementation, pinned
  to `twetto/iq-neuron` commit
  `a8752eba49dba9ba43a64be74090b91a51044b2f`. The source is piecewise linear,
  uses a C++-truncated branch point and Q0.3 arithmetic shift, emits only above
  `v_max`, hard-resets, and otherwise applies the lower clamp.
- Added complete-contract Python, Rust-engine, independent Rust-safety, Julia,
  Go, and Mojo execution with atomic failure semantics and configured-parameter
  parity. Paired TOML/JSON schemas reproduce the independent 400-tick source
  trace exactly.
- Added registered and folded signed-Q32.0 Python-to-Verilog co-simulation and
  a depth-4 Z3 catalogue job. IQIF relies only on the signed compiler path; the
  evidence makes no unsigned-mode claim.
- Added a source-hashed five-backend benchmark with zero trajectory, event, or
  final-state mismatches. The committed timings are non-exclusive local
  regression evidence, not a production, cross-host, hardware, or universal
  speed claim.
- Expanded the public formal inventory to 50 proof jobs and 180 statements
  (150 assert, 7 assume, 23 cover), split across 18 non-catalogue and 32
  catalogue jobs.

### Poisson source-to-silicon closure

- Replaced the process-local random generator with the canonical seeded LFSR16
  threshold contract shared by Python, the Rust engine and safety module,
  Julia, Go, Mojo, the schema interpreters, and generated Q24.24 RTL.
- The maintained binary-bin recurrence uses
  `p = 1 - exp(-rate_hz * dt_ms / 1000)`, advances exactly one trial per
  accepted bin, preserves deterministic reset replay, and rejects invalid work
  before committing the caller-visible RNG state.
- Added full-contract backend parity and loading tests, a complete-period
  statistical reference artifact, registered/folded RTL co-simulation, a
  depth-4 Z3 safety job, descriptor/readiness evidence, and a source-hashed
  five-backend benchmark. Recorded timings are labelled non-exclusive local
  regression evidence rather than a universal performance claim.

### Neuromorphic profile responsibility modularisation

- Replaced the 1,065-line mixed registration module with a 65-line stable
  facade over bounded event-driven (194 lines), programmable/aerospace (202),
  heterogeneous-accelerator (271), memory-compute (225), and physical-compute
  (249) modules.
- Preserved every field of all 70 profiles at SHA-256
  `c21e2075bb98787db31afbaa0c41cb6300e6707663de49b6630d6b5dbb3d1d41`,
  their historical registry insertion order, the facade's public
  `HardwareProfile` identity, and duplicate-registration failure semantics.
- Architecture contracts pin exact responsibility ownership, registry-only
  private dependencies, the 18-step facade composition order, fully typed and
  documented registrars, side-effect-free private reloads, public class
  filters, and module-specific ceilings. The 206-test focused platform cohort
  covers all 108 statements and two branches exactly.
- Refreshed the generated API and capability inventory on accepted parent
  `33d74f904`. The generated delta includes that commit's EscapeRate API, one
  validation page, and four tests plus this lane's architecture test; the nine
  unrelated uncommitted Rust safety files are not represented.
- Registered the already-landed COBA-LIF and EscapeRate evidence pages under
  the public Validation navigation, closing their inherited policy gap.
- Strict MyPy, Ruff lint/format, NumPy docstrings, Bandit, generated API and
  capability checks, SPDX, navigation policy, and strict MkDocs pass. A clean
  `33d74f904` overlay wheel contains all five private modules, preserves the
  194-profile registry, exact 70-profile digest/order, and 349-line platform
  CLI output, and has SHA-256
  `5bfb86a0c65ccb2c394fbededbb6992368a8b3825842b4dc7654f8ebd579e185`.
- The split only reorganises static Python catalogue metadata. It does not
  validate vendor specifications, alter a numerical runtime kernel, introduce
  Rust/Go/Julia/Mojo ownership, or create a throughput claim.

### Photonic emitter responsibility modularisation

- Replaced the 1,109-line mixed photonic emitter with a 64-line definition-free
  compatibility facade over types (106 lines), bitstream conversion (117), FDTD
  (297), compilation/GDSII (281), Meep (218), crosstalk (321), and bitstream
  emission (83).
- Preserved the historical public exports, callable signatures, class identity,
  pickle paths, monkey-patchable engine boundary, and deterministic valid-output
  digests. Architecture contracts pin exclusive ownership, the acyclic import
  graph, facade purity, and per-module ceilings.
- Hardened finite and physical validation, fixed zero-length and 2D waveguide
  edge cases, and made unavailable Meep execution raise ``ImportError`` instead
  of returning synthetic results. The 102-test focused cohort covers every one
  of the 702 statements and 214 branches in the eight governed Python modules.
- Replaced crosstalk placeholders with executable Rust, Go, Julia, and Mojo
  pair/bank/batch contracts. These mirrors cover crosstalk only; Python owns
  FDTD, Meep, compilation, netlist, GDSII, and filesystem effects. The committed
  4,096-pair benchmark is local, loaded, non-isolated evidence and is explicitly
  ineligible for a universal performance claim.

### UVM generator responsibility modularisation

- Replaced the 1,138-line mixed UVM generator with a 55-line historical facade
  over RTL-contract/parser (183 lines), configuration (103), generated-artifact
  packaging (57), UVM component emitters (509), simulator/formal harness
  emitters (268), and orchestration (125).
- Preserved the exact 13 package exports, signatures, object identities,
  pickle-qualified paths, private emitter methods, and six representative
  default/configured parent payload SHA-256 values. Those payloads cover every
  generated artifact, file-list entry, and formal link.
- Architecture contracts pin exclusive class ownership, the acyclic dependency
  graph, facade and module ceilings, and deterministic payloads. The 82-test
  focused package cohort covers all 422 statements and 112 branches exactly;
  strict MyPy, Ruff, NumPy docstrings, and Bandit pass on the split surface.
- This is a behaviour-preserving Python text-generator refactor. The
  architecture map still labels emitted UVM/formal execution as draft; no
  vendor-UVM compile, polyglot, or throughput promotion is claimed. Existing
  generated Go/Rust/Julia/Mojo placeholders remain explicit adjacent debt.

### CMOS profile responsibility modularisation

- Replaced the 1,145-line CMOS profile registration GodFile with a 67-line
  stable facade over bounded FPGA (562 lines), conventional-accelerator (295),
  architecture-paradigm (181), embedded-processor (140), and ASIC/simulation
  reference (90) modules.
- Preserved every field of all 72 profiles at SHA-256
  `0d72e74b9a779dd3a98b70817a12fac902528e741cea3772a7dc6bf37be20fc9`,
  their historical registry insertion order, the facade's public
  `HardwareProfile` identity, and duplicate-registration failure semantics.
- Architecture contracts pin exact responsibility ownership, a registry-only
  private import graph, facade composition order, side-effect-free private
  module reloads, and module-specific ceilings. The 197-test focused platform
  cohort covers all 111 statements and two branches exactly.
- Strict MyPy, Ruff lint/format, NumPy docstrings, Bandit, public-docstring and
  generated-doc checks, capability-manifest checks, and strict MkDocs pass.
  The installed wheel contains all five private modules, preserves the exact
  digest/order and CLI surface, and has SHA-256
  `60092f90bb902fa57e5c8f41e7adbc5ee22536c5fa726685729e650446544e01`.
- The generated API/capability refresh is based on accepted parent
  `f431710c3`: it includes that commit's COBA-LIF API, validation page, and five
  tests plus this lane's architecture test. No uncommitted peer work is
  represented in those generated surfaces.
- Refreshed the hardware-profile guide against the live registry: 194 profiles,
  38 classes, and 121 vendors. The split only reorganises static Python
  metadata; no numerical kernel, Rust/Go/Julia/Mojo mirror, or throughput claim
  applies.

### Studio training responsibility modularisation

- Replaced the 1,208-line Studio training GodFile with a 311-line historical
  facade and bounded execution, parent-control, weight-attach, and event-codec
  modules. Architecture contracts enforce the one-way dependency graph and
  module-specific size ceilings.
- Preserved the 14-name public export set, callable signatures,
  stream-generator contract, pickle identity, and all 12 composed HTTP routes.
  A seeded real-Torch run retains the established metrics, `64->8->10`
  architecture, 610 parameters, and learned tensor-state digest.
- The 131-test focused training cohort covers all 562 statements and 128
  branches across the five split modules. Live attach now fails closed for a
  thread-backed target or a target that stops during control delivery instead
  of acknowledging a command that cannot be consumed.
- Strict MyPy, Ruff lint/format and NumPy docstrings, Bandit, scoped public
  docstrings, SPDX, generated API/capability and OpenAPI checks, and strict
  MkDocs pass. The installed wheel contains every responsibility module,
  preserves facade and pickle identity, and completes a real Torch training
  smoke; its SHA-256 is
  `10ffc95f29937cc34783ad9e7ac478622d5a41708390522700508c4c0b2dcdd0`.
- This is a behaviour-preserving Python/PyTorch service refactor. No functional
  Rust, Julia, Go, or Mojo Training Monitor counterpart is wired, and no
  cross-language or throughput claim is introduced.

### DPI coupled-circuit source-to-silicon closure

- Replaced the historical one-state DPI surrogate with the coupled
  current-domain silicon-neuron equations from Indiveri, Stefanini, and Chicca
  (2010), Eqs. (2)–(3). The maintained Python, Rust engine and safety, Go,
  Julia, and Mojo paths now share nonlinear positive feedback, membrane and
  after-hyperpolarisation states, simultaneous explicit Euler updates,
  post-update reset, and spike-driven refractory-pulse semantics.
- Added source-bound parity and fail-closed ABI contracts over the enrolled
  current vector. All five runtime lanes preserve `0/0/0/0/1/3/6/11/21`
  events at `I=-0.1/0/1/2/3/5/10/20/50` over 1,000 steps, with floating-state
  differences below `5e-13` and invalid work rejected before visible state or
  buffer mutation.
- Enrolled paired TOML/JSON schemas, generated Q16.16 RTL, co-simulation,
  reference-trace, descriptor, benchmark, and formal evidence. The fixed-point
  path preserves 13 events at `I=5` over 5,000 steps within its declared state
  and timing envelopes, and the generated depth-4 Z3 proof passes. Controlled
  benchmark timings remain explicitly local, loaded-host evidence rather than
  isolated throughput claims.
- Corrected clean-checkout native loading and regeneration. The source bridge
  now resolves the extension installed by `maturin develop`, while the Go build
  hint compiles the package so the generated C header retains its committed
  package identity and source digest.

### Bus-interface responsibility modularisation

- Replaced the 1,221-line bus-interface GodFile with a 179-line historical
  compatibility facade, a 393-line AXI4-Lite/Wishbone wrapper renderer, a
  608-line AXI4-Lite live-parameter-bank core renderer, and a 175-line PCIe
  register-window adapter. Architecture contracts enforce bounded modules,
  the one-way import graph, the exact public export set, and private renderer
  ownership.
- Split the 1,012-line mixed live-control test surface into static contract,
  AXI4-Lite execution, PCIe execution, wrapper/register-map, and architecture
  suites. The 119-test focused integration cohort passes; the four production
  modules cover all 184 statements and 66 branches exactly. Strict MyPy, Ruff,
  Bandit, NumPy docstrings, public-docstring policy, SPDX, generated API and
  capability checks, strict MkDocs, and installed-wheel smoke checks pass.
- Preserved the historical function identities and signatures and the exact
  parent-generated bytes for representative AXI4-Lite, Wishbone, empty-wrapper,
  AXI live-bank, PCIe live-bank, and register-map payloads. The installed wheel
  contains every responsibility module and has SHA-256
  `0fb62c38bf9eeb26fa8a291880a335864d90f17de8bb53bae289bcaa35ae26ce`.
- Refreshed the Python/SystemVerilog live-control evidence with executable AXI
  trap and PCIe commit simulations. The affinity-only, powersave-governor
  timings remain local diagnostic context, not a production throughput claim;
  no Rust, Julia, Go, or Mojo counterpart exists for this HDL adapter surface.

### Cortical-column responsibility modularisation

- Replaced the 1,259-line cortical-column GodFile with a 733-line historical
  public runtime, a 146-line optional-backend discovery boundary, a 395-line
  sparse-connectivity builder, and a 121-line published-parameter module.
  Architecture contracts enforce the acyclic dependency direction and bounded
  file sizes.
- Preserved the exact public constructor signature, class module and qualified
  name, direct constant re-exports, backend capability flags, reload patch
  points, and RNG draw order. Parent and candidate fingerprints match for the
  single-delay per-pair, distributed-delay per-pair, and distributed-delay
  block-CSR modes.
- The 76 focused passing tests plus one optional-backend skip cover all 553
  statements and 170 branches exactly. Strict MyPy is clean across the four
  source and two strict contract files; Ruff, Bandit, scoped NumPy-docstring,
  SPDX, generated API-reference, and capability-manifest checks pass.
- This is a behaviour-preserving Python responsibility split. The existing
  Rust, Julia, Go, and Mojo sparse-kernel contracts are unchanged, and no new
  throughput or scientific-fidelity claim is introduced.

### Studio jobs responsibility modularisation

- Replaced the 1,311-line Studio jobs GodFile and its 1,496-line test GodFile
  with a 95-line historical facade, nine focused implementation modules,
  shared architecture support, seven responsibility-focused behavior suites,
  an architecture suite, and a compact adversarial sentinel. The largest
  implementation and test files are 295 and 284 lines. The exact 18-name public
  export contract preserves object identity, qualified names, and pickle paths
  through an acyclic runtime import graph.
- The 64-test focused suite covers all 650 statements and 120 branches exactly,
  and the full 1,174-test Studio prefix remains green. A dedicated Python 3.12
  workflow measures the jobs modules independently of the repository-wide
  Studio coverage exclusion. Adversarial cases cover malformed control
  payloads, seed-size boundaries, corrupt stored job paths, symlink escape,
  absent work directories, invalid seed types, and cross-drive path handling.
- Parent and candidate retain the same 18-export, five-route jobs contract at
  SHA-256
  `8bc27f16c8d97e33cc0cb5d21c15a775c5c070354282c12e84e52732a3fefd5c`.
- This is a Python-only Studio control-plane and filesystem-confinement
  refactor. It preserves thread and isolated-process execution behavior and
  makes no numerical-kernel, cross-language-mirror, or throughput claim.

### Studio policy responsibility modularisation

- Replaced the 1,319-line Studio policy GodFile and its 1,135-line test file
  with a 78-line compatibility facade, nine focused implementation modules,
  shared test support, eight focused test modules, and an import-only sentinel.
  The largest implementation and test files are 249 and 272 lines. Historical
  package and module imports preserve object identity, qualified names, and
  pickle compatibility through an acyclic responsibility graph.
- The 53-test focused suite covers all 412 statements and 96 branches, and the
  full 1,165-test Studio suite remains green. A dedicated Python 3.12 workflow
  enforces exact coverage independently of the repository-wide Studio
  exclusion. The parent and candidate retain the same duplicate-free 114-route
  registry at SHA-256
  `452152885f8f5dbf97b4a76b2d30868c3fbded2984d2828dfe273e57834d5084`.
- Verified parent-produced pickle compatibility and an isolated installed-wheel
  import of every split module. This is a Python-only Studio control-plane and
  security contract, not a numerical kernel, so no cross-language mirror or
  throughput benchmark is claimed.

### Theta exact-flow source-to-silicon closure

- Exposed the tangent-half-angle exact constant-current flow through Python,
  the Rust engine and safety module, Julia, Go, and Mojo. Julia, Go, and Mojo
  carry the complete phase/timestep contract; the Rust engine retains its
  explicit factory-default boundary, and an executable safety probe covers the
  standalone Rust implementation.
- Added an executable 11-point current vector for every compiled lane, bounded
  circular phase parity at `2e-12`, fail-closed C ABI buffer contracts, focused
  100% coverage for the Python dispatcher and backend wrapper, and a controlled
  single-CPU benchmark with source hashes and host-load evidence.
- Preserved the existing Euler schema/compiler boundary and upgraded its proof:
  paired TOML/JSON schemas and generated Q16.16 RTL retain the complete event
  count vector, moderate-regime circular phase error stays below `0.17` rad,
  the I=1 one-cycle timing displacement is explicit, and the curated depth-110
  Z3 job reaches the first receipt-drive event while proving the declared
  phase/event envelope.

### Hierarchical partitioner responsibility modularisation
- Replaced the 1,324-line hierarchical partitioner and its two oversized test
  surfaces with a 104-line compatibility facade, ten focused implementation
  modules, shared test support, and nine focused test modules. The largest
  implementation and test files are 258 and 291 lines. All 21 established
  public definitions preserve their package and historical imports, identities,
  qualified names, and pickle paths through an acyclic responsibility graph.
- The exact 111-test owning suite covers all 664 statements and 186 branches.
  Backend decoding fails closed on incomplete, duplicate, or out-of-range
  partition maps; architecture tests enforce one public owner, dynamic runtime
  diagnostics, file-size limits, the CI gate, and the distinction between real
  kernels and generated mirrors.
- Retained the maintained Rust, Julia, Go, Mojo, and Python KL-refinement chain
  and removed four non-executable generated mirrors. The source-bound parent and
  candidate benchmarks produce the same canonical partition SHA-256
  `258b5c0c54ea33758f60090e68b7ce2a7657800ec16a6c90bc0eb52fbbd3585f`
  across every workload and backend. Affinity-only, powersave-governor timings
  are explicitly diagnostic and do not claim isolated throughput.

### Portable Tier-3 golden traces
- Replaced the false assumption that NumPy transcendental traces are byte-
  identical across heterogeneous x86 runners with an explicit exact-hash
  variant contract. The Chialvo and Hodgkin-Huxley descriptors record the
  measured AVX-512 and portable-kernel digests; no arbitrary digest is
  accepted.
- Added a regression that reruns every variant-bearing model with AVX-512
  disabled, confirms both the native and portable digests are declared, and
  enforces the existing `1e-9` numeric-parity ceiling. The observed boundaries
  are one ULP for Chialvo and `1.902e-11` maximum absolute error for Hodgkin-
  Huxley, with unchanged events.

### Quadratic IF exact-flow source-to-silicon closure
- Exposed the exact constant-current Riccati flow through Python, the Rust
  engine, Julia, Go, and Mojo. Julia, Go, and Mojo carry the complete numeric
  contract through real executable ABIs; the Rust engine retains its explicit
  factory-default boundary, and an independent executable probe exercises the
  actual Rust-safety module. Dedicated tests preserve exact events at eight
  currents and bound every compiled voltage trace to `2e-12` from Python.
- Replaced the raw mirror-style timing record with a source-hashed public-API
  benchmark and executable Rust-safety gate. The CPU-10 affinity-pinned,
  non-reserved powersave-governor run records 2,631 events and zero observed
  trace difference in every lane. Its heavy host load and empty isolated-CPU
  set make the timings local regression evidence rather than a throughput
  claim.
- Added dedicated hand/schema/RTL fidelity evidence. The paired Euler schemas
  preserve the exact-flow hand model's event sequence over a varied 1,000-step
  drive with state error below `0.006` and reset inclusively at the configured
  `v_peak`, including an exactly-equal Euler candidate. Q16.16 RTL retains the
  complete enrolled event vectors at I=0/0.333/0.5/1/2/5/20/50 with voltage
  error below `0.011`. The I=0.1 one-cycle reset displacement is declared
  rather than hidden, and the regenerated depth-20 Z3 job proves the inclusive
  threshold RTL.

### Perfect Integrator source-to-silicon fidelity closure
- Exposed the candidate-first non-leaky Euler recurrence through Python, the
  Rust engine, Julia, Go, and Mojo. Julia, Go, and Mojo carry the complete
  numeric contract; the Rust engine keeps its explicit factory-default
  boundary. Dedicated source-golden tests preserve bit-exact traces and
  0/32/66/200/250/500/1,000 events at I=0/0.333/0.7/2/3/5/20 over 1,000 steps.
- Added a source-hashed five-backend public-dispatch benchmark. The CPU-10
  affinity-pinned, non-reserved powersave-governor run records 50,000 events
  and zero trace difference in every lane. Its high host load and empty
  isolated-CPU set make the timings local regression evidence rather than a
  production throughput claim.
- Strengthened the existing schema-to-RTL evidence from a saturated-only point
  to 66-event hand/schema/Q8.8 parity at I=0.7 over 1,000 steps. The I=0.333
  quantisation boundary is recorded honestly as 32/32/31; the existing analytic
  sawtooth reference and generated depth-20 Z3 proof remain cross-wired.

### Bioware responsibility modularisation
- Replaced the 1,378-line biological-interface implementation and 1,139-line
  test GodFile with a 129-line compatibility facade, nine focused
  responsibility modules, shared validation, and 13 focused test modules. The
  largest implementation and test files are 296 and 253 lines. All established
  package and historical imports preserve object identity, qualified names,
  and pickle compatibility through an acyclic dependency graph.
- The 200-test focused suite covers all 1,052 statements and 384 branches.
  Contracts now reject invalid MEA shapes and values, AER overflow, out-of-window
  events, inconsistent payload counts, unsafe optical-power arithmetic, invalid
  Q8.8 state, audit reordering, and partial session round advancement.
- Removed non-executable generated Go, Julia, Mojo, and Rust Bioware mirrors.
  The maintained full orchestration is Python-only; independent Julia
  plasticity solvers remain a separate numerical surface.
- Added a source-bound 30-sample parent/candidate benchmark. Both trees produce
  the same 6,865-byte canonical MEA-to-opto payload at SHA-256
  `2491dc73a2de93a45a1cc944539c170b151403e42b973b18806143f318b7d669`.
  The affinity-only, loaded-host timing regression is documented as local
  diagnostics and is not a throughput or hardware claim.

### Lapicque exact-flow source-to-silicon closure
- Exposed the maintained exact constant-current RC recurrence through Python,
  the Rust engine, Julia, Go, and Mojo. The public dispatcher follows the
  committed measured Mojo/Julia/Go/compatible-Rust/Python order; Julia, Go, and
  Mojo transport the complete numeric contract, while the Rust engine retains
  its explicit factory-default boundary. Dedicated native-path tests preserve
  0/0/71/200/500 events at I=0/0.5/2/5/20 over 1,000 steps and bound every
  compiled trace to `2e-15` from Python.
- Replaced the earlier mirror-style benchmark with a source-hashed public-API
  harness and committed five-backend result. The single-logical-CPU,
  powersave-governor run records 20,000 events in every lane and a maximum
  voltage difference of `4.44e-16`; its high host load and absent isolated-CPU
  set make the timings local regression evidence rather than a throughput
  claim.
- Aligned the paired schemas and generated RTL with exponential-Euler lowering
  and inclusive candidate thresholding. Hand, TOML, and JSON event vectors are
  exact and their state traces remain within `2e-15`; Q16.16 RTL retains
  complete event vectors at I=0.333/2.3/20.25 over 1,000 steps with voltage
  error below `0.04`. The translation DOI, independent closed-form trace,
  S5/H1 descriptor, readiness evidence, and depth-20 Z3 proof are cross-wired.

### NIR hardware-graph responsibility modularisation
- Replaced the 1,414-line hardware-graph implementation and three primary test
  files totalling 4,091 lines with an 84-line compatibility facade, seven
  acyclic responsibility modules, and 40 focused test/support modules. All 28
  established definitions preserve their historical imports, qualified names,
  identities, and pickle paths.
- The focused package passes 240 tests with two optional-runtime skips and
  covers all 651 statements and 324 branches. A dedicated Python 3.12 workflow
  enforces exact statement and branch coverage over every `neuron_graph*.py`
  module independently of broader NIR exclusions. Parser, optional-framework,
  SC-NIR, FPGA, file-I/O, hierarchy, dense, delay, threshold, and metadata
  boundaries remain exercised through their maintained entry points.
- Added a source-bound 30-sample parent/candidate benchmark. The maintained
  two-population graph produces the same 22,860-byte FPGA compilation payload
  at SHA-256
  `32498fa1106229a4fe064862e20b86e0f0b1f0d42f8598d1988e06e68c13ef13`.
  Affinity-only, non-reserved CPU timing medians are local regression
  diagnostics, not throughput claims. Cross-language kernels do not apply to
  this typed NIR metadata and FPGA-orchestration refactor; neuron-dynamics
  implementations are unchanged.

### ExpIF source-to-silicon fidelity closure
- Bound `ExpIFNeuron` to Fourcaud-Trocmé et al. (2003), Eqs. 6 and 10, DOI
  `10.1523/JNEUROSCI.23-37-11628.2003`. Python, Rust safety, the Rust engine,
  Go, Julia, and Mojo now share the candidate-first RK4 recurrence, fitted
  defaults, `+30 mV` finite cutoff, and fail-closed event handling. The
  optional `1.7 ms` fitted refractory protocol remains explicit rather than
  changing the zero-refractory catalogue default.
- Dedicated production-boundary tests preserve `0/0/2` events at
  `I=0/5/20` over 1,000 steps. Compiled traces stay within `5e-8` of Python;
  Rust, Julia, Go, and Mojo expose real executable paths, while the Rust engine
  boundary remains factory-default-only.
- Replaced the resting approximation trace with an independently integrated
  DOI-bound driven trace. Hand, TOML, JSON, and generated Q32.32 RTL preserve
  the enrolled event counts; Q16.16 is explicitly excluded. The S5/H1
  descriptor, readiness index, catalogue RTL, and depth-4 Z3 proof are
  cross-wired, moving the formal catalogue from 26 to 27 jobs.
- Added a source-hashed, single-logical-CPU five-backend benchmark. The
  committed powersave-governor run records 523 events in every lane and a
  maximum voltage difference below `5e-8`; its loaded-host timings are local
  regression evidence rather than a throughput claim.

### AdEx five-backend baseline-Euler closure
- Added executable Rust-safety, Go, Julia, and Mojo recurrence coverage around
  the maintained Python AdEx model and Rust engine path. Public dispatch carries
  the complete numeric contract through Julia/Go/Mojo, keeps the factory-default
  Rust boundary explicit, and fails closed on invalid state or candidates.
- Dedicated parity tests preserve `0/4/12` events at `I=0/200/500` over 1,000
  steps and bound every compiled voltage trace to `5e-12` from Python. The
  existing DOI-backed independent reference and Q16.16 Python-to-Verilog
  co-simulation remain the declared source and H1 hardware evidence.
- Added a source-hashed, single-logical-CPU benchmark for all five public paths.
  The committed powersave-governor run records 1065 events in every lane and a
  maximum trace error of `7.40e-13`; its loaded-host timings are local regression
  evidence, not a hardware throughput claim.

### Hodgkin-Huxley executable Mojo lane
- Replaced the non-executable parity note with a compiled C ABI for the
  complete four-state Hodgkin-Huxley recurrence. The Python dispatcher carries
  the maintained numeric state and parameter surface, mirrors the historical
  gate-first baseline-Euler schedule, and fails closed on invalid inputs or
  candidates without partially committing state.
- Dedicated production-boundary tests preserve `0/6/9` events at `I=0/10/20`
  over 100 macro-steps. Mojo's exp/FMA trace stays within the measured `2e-9`
  envelope, including non-default state and parameter transport; an
  RK4-configured instance is rejected rather than silently changing its
  integrator.
- Added a source-hashed, single-logical-CPU benchmark for Python, the Rust
  engine, and Mojo. The recorded timings are loaded-host local regression
  evidence; event parity, trace bounds, source digests, final states, affinity,
  governor, load, and runtime versions are the maintained contracts.

### Connor-Stevens executable Mojo lane
- Replaced the non-executable Mojo parity note with a compiled C ABI for the
  complete six-state Connor-Stevens recurrence. The public Python dispatcher
  carries the maintained state and parameter surface, uses the same 100
  candidate-first RK4 sub-steps per macro-step, and fails closed on invalid
  inputs or candidates.
- Dedicated production-boundary tests preserve the established `0/2/9` event
  counts at `I=0/10/20` over 100 macro-steps. Mojo's transcendental/FMA trace
  remains within the measured `2e-6` absolute envelope, including non-default
  state and parameter transport.
- Added a source-hashed, single-logical-CPU benchmark for Python, the Rust
  engine, and Mojo. The recorded timings are loaded-host local regression
  evidence; event parity, trace bounds, source digests, final states, affinity,
  load, and runtime versions are the maintained contracts.

### ASIC-flow GodFile modularisation
- Replaced the 1,451-line implementation and 814-line primary test file with a
  123-line compatibility facade, nine acyclic responsibility modules, and 11
  focused test modules. The largest implementation and test files are 319 and
  234 lines. All 38 historical definitions retain identity, qualified names,
  and pickle compatibility.
- The focused suite passes 100 tests and covers all 580 statements and 66
  branches. Strict touched MyPy, Ruff, NumPy docstrings, architecture, package,
  and benchmark-source contracts pass.
- Removed nonfunctional Rust, Go, Julia, and Mojo ASIC-flow mirrors. The real
  boundary remains Python deck generation followed by external EDA execution.
  A source-bound 30-sample benchmark proves the 10,465-byte parent/candidate
  output is identical. Its affinity-only, loaded-host timings are local
  regression context rather than throughput or physical-design evidence.

### Evolutionary-substrate GodFile modularisation
- Replaced the 1,737-line implementation and 1,227-line primary test file with
  a 156-line compatibility facade, 14 acyclic responsibility modules, and 18
  focused test modules. The largest implementation and test files are 304 and
  312 lines. All 56 historical exports retain object identity, qualified names,
  and pickle compatibility; deterministic parent/candidate Python-fallback
  workflows are byte-identical.
- The focused suite passes 160 tests with real Mojo and four-runner parity
  paths. The 135-test core covers all 1,005 statements and 166 branches. Native
  Rust, Julia, Go, and Mojo checks pass 17, 17, 8, and 7 tests, while the
  18-test cross-language runner suite completes without skips. Strict touched
  MyPy, Ruff, NumPy docstrings, Bandit, SPDX, and wheel smoke checks pass.
- Rebuilt the Python and five-language benchmark producers around 30-sample
  schema-v2 evidence with raw samples, source digests, runtime versions,
  affinity, governor, frequency, and load context. The committed run had no
  kernel-reserved cores and heavy concurrent load, so its values are local
  regression diagnostics rather than publishable throughput claims.

### Ibarz-Tanaka four-branch source-to-silicon correction
- Replaced the unsupported beta-based rational/linear hybrid with Ibarz,
  Tanaka, Sanjuan, and Aihara (2007), DOI `10.1103/PhysRevE.75.041902`, Eqs.
  2–3: four fast-map branches and the simultaneous slow update
  `u_next = u - mu*(v + 1 - sigma)`. Events now identify execution of the
  source's fixed `-1` reset branch; the stale 2011 review citation, `beta`,
  configurable threshold, and configurable reset were removed.
- Rebuilt the fail-closed Python, Rust engine, Rust safety, Go, Julia, and Mojo
  paths around the same recurrence. Rust, Julia, and Go are bit-exact across
  the 1,000-step `I=0/0.2/1` envelope; Mojo preserves event vectors below a
  measured `1.5e-8` absolute band. The source-hashed 21-call pinned benchmark
  at `I=0.2` records 33 events in every lane. The host had no kernel-isolated
  CPU set and high load, so its timings are local regression evidence.
- Added paired TOML/JSON schemas and an independently re-derived DOI reference.
  Hand/TOML/JSON agree exactly. Over the bounded 30-step `I=0.2` co-simulation,
  Q16.16 RTL preserves all three reset events with maximum `v/u` errors below
  `0.003/0.0001` while exercising every source branch. Q8.8 is invalid because
  `mu=0.001` quantises to zero.
- The S5/H1 descriptor, readiness evidence, generated Q16.16 catalogue core,
  and depth-4 Z3 proof are cross-wired. The polyglot-complete model count is
  unchanged; schema-gap counts move to `33 / 120 / 122`, the catalogue
  inventory moves to 26, and the whole HDL inventory moves to 63 jobs / 211
  statements.

### Chiplet responsibility modularisation
- Replaced the 1,744-line chiplet implementation and 1,223-line primary test
  file with a 98-line compatibility facade, seven focused public-responsibility
  modules, one private SystemVerilog helper, and focused tests below 500 lines.
  All 36 historical imports, qualified names, and pickle identities remain
  stable. Canonical three-die EMIB generation remains byte-identical.
- Added finite physical-boundary validation and direct topology, routing,
  thermal, RTL, link-protocol, power-domain, partition, and architecture
  contracts. The focused package cohort passes 172 tests; exact-file coverage
  reaches all 680 statements and 202 branches, with strict MyPy and Ruff clean.
- Removed nonfunctional Rust, Go, Julia, and Mojo chiplet-control mirrors and
  their Rust registry entries. The source-bound 30-repeat benchmark now records
  the maintained Python path, including complete generation for 2, 8, and 32
  dies. Non-exclusive host load and validation overhead make the old/new values
  diagnostic evidence rather than promotion claims.

### Quantum-annealing GodFile modularisation
- Replaced the 1,910-line quantum-annealing implementation and 1,001-line
  primary test GodFile with a 98-line compatibility facade, nine acyclic
  responsibility modules below 400 lines, and focused typed tests below 500
  lines. All 24 historical imports and pickle identities remain stable. The
  linked cohort passes 250 tests with one optional `neal` skip, while 216
  exact-file tests cover all 1,040 statements and 442 branches.
- Corrected the exact QUBO-to-Ising diagonal offset, made backend selection
  explicit, validated native and QPU boundaries, hardened compiler/model/
  schedule/sample inputs, made JSON exports atomic, and retained exact global
  indices through overlapping decomposition.
- Removed nonfunctional generated Rust safety, Go, Julia, and Mojo mirrors.
  The maintained native authority remains `engine/src/quantum.rs` with 12
  focused tests. The source-bound 30-sample local benchmark records +99.65%
  compile, -25.37% cold import, -16.67% Python solve, -23.21% process wall,
  and -3.48% RSS median changes. Rising non-exclusive host load makes these
  regression diagnostics, not release or quantum-speedup evidence.

### Medvedev slow-calcium first-return source-to-silicon correction
- Replaced the repository's falsely attributed tent map with the scalar
  slow-calcium first-return construction derived in Section 4 of Medvedev
  (2005), DOI `10.1016/j.physd.2005.01.021`. Its three source regions follow
  equations 4.4/4.7, 4.8 with 4.13, and 4.15. The paper does not tabulate one
  unique global pair of return functions, so the complete calibration is
  disclosed. Non-zero current and the pre-state event label remain explicit
  SC-NeuroCore conventions.
- Rebuilt the fail-closed Python, Rust engine, Rust safety, Go, Julia, and Mojo
  paths around the same recurrence. Rust, Julia, and Go are trace-identical on
  the enrolled envelope; Mojo retains exact event vectors below a `5e-13`
  trace band. The source-hashed 500,000-iteration benchmark records 375,000
  events in every lane and medians of 8.230 ms Julia, 10.799 ms Rust,
  19.524 ms Mojo, 20.959 ms Go, and 242.929 ms Python on its recorded host.
- Added paired TOML/JSON schemas and an independently re-derived Section 4
  reference contract. At `I=2`, hand/TOML/JSON agree exactly and Q16.16 RTL
  preserves the complete 75-event vector over 100 iterations with maximum
  state error below `0.007813`. Q8.8 cannot represent the calibrated
  `d=2271.1927977404063`, so both behavioural RTL and the generated catalogue
  core use Q16.16. The shared log LUT has 256 positive-domain entries over
  `[1/256, 8 + 1/256)` at step `1/32`.
- The S5/H1 descriptor, readiness evidence, generated Q16.16 catalogue core,
  and depth-4 Z3 proof are cross-wired. The public fidelity count remains 21,
  schema-gap counts move to `32 / 121 / 123`, the catalogue inventory moves to
  25, and the whole HDL inventory moves to 62 jobs / 210 statements.

### Studio responsibility-router and OpenAPI contract split
- Reduced `src/sc_neurocore/studio/app.py` from 3,662 lines to a 147-line
  composition root. Sixteen routers now own system, jobs, audit, identity,
  catalogue, presets, simulation, compiler, co-simulation, synthesis, design,
  deployment, training, training weights, export, and adaptive-precision
  responsibilities; shared runtime, security, schema, guard, preset-flow, and
  frontend helpers live beside them.
- Preserved the complete hermetic API surface: 113 backend HTTP routes, 110
  OpenAPI paths, and `/ws/progress`. Canonical OpenAPI JSON remains
  byte-semantically equal to the clean pre-refactor document at SHA-256
  `b1e698de8ce38e15f152a3bffdf86d24dc8f14c12747a927be320bcc25825cdb`.
- Kept the frontend root route conditional on the ignored `dist/` build while
  excluding that UI document route from OpenAPI. Runtime and generated API
  contracts are therefore identical in local, CI, and clean-worktree builds.
- Added deterministic `tools/generate_studio_openapi.py` generation, a
  committed JSON reference, MkDocs API guidance, route-ownership assertions,
  normalized handler parity checks, and public-route coverage for terminal,
  validation, isolation, policy, audit, identity, frontend, and artifact
  failure boundaries.

### Hindmarsh-Rose RK4 schema-to-RTL enrolment
- Re-enrolled the already acceleration-complete Hindmarsh-Rose model against
  Hindmarsh and Rose (1984), DOI `10.1098/rspb.1984.0024`. Paired schemas now
  reproduce the maintained three-state RK4 hand flow, no-reset state, and
  upward `x >= x_threshold` observation. The former explicit-Euler schema's
  identity reset had disabled crossing-edge history and counted every
  above-threshold timestep as an event.
- The hand model and both schema formats agree across a varied 1,200-step
  sequence. Over 2,000 steps, hand/TOML/JSON/Q16.16 RTL crossing counts are
  exactly `0/0/26/40/52` at `I=0/2/3/4/5`. The longer 5,000-step chaotic
  boundary is explicit: Q16.16 reports `10/49/86/115` crossings at
  `I=2/3/4/5`, one more than float64's `9/48/85/114`.
- The DOI-backed reference is independently re-derived with classical RK4.
  The S5/H1 descriptor adds a generated Q8.8 port-only formal job whose
  depth-4 SymbiYosys/Z3 check passes. Existing model, acceleration,
  backend-parity, and benchmark artefacts remain unchanged; schema-gap counts
  stay at `31 / 122 / 124`, while the formal catalogue moves to 24 jobs and
  the whole HDL inventory to 61 jobs / 209 statements.

### Chialvo source-to-silicon fidelity enrolment
- Bound `ChialvoMapNeuron` to Chialvo (1995), DOI
  `10.1016/0960-0779(93)E0056-H`. The Python and compiled lanes now share the
  paper's simultaneous `x*x*exp(y-x)+k+I`, `a*y-b*x+c` recurrence and reject
  non-finite candidates without committing them. The paper permits a constant
  or time-dependent additive perturbation; `k` and `current` represent those
  roles. The upward `x_threshold=1.0` crossing is kept separate as a maintained
  observation convention.
- Wired checked Rust-engine, Rust-safety, Go, Julia, and Mojo batch paths into
  measured-order `auto` dispatch. Cross-language tests enforce one-step
  source-equation envelopes and identical 1,000-step event counts at six
  currents. The committed 500,000-iteration, five-repeat benchmark was pinned
  to one logical CPU and records 12,935 events in every lane. Its medians are
  7.270 ms Rust, 9.576 ms Julia, 11.373 ms Mojo, 20.524 ms Go, and 2,175.866 ms
  Python; the JSON records that the host had no kernel-isolated CPU set.
- Added paired TOML/JSON map schemas and an independent 100-iteration DOI
  reference. Hand/TOML/JSON states and events are exact at
  `I=-0.05/0/0.01/0.1/1.0`; Q16.16 retains event counts `0/2/3/0/1` and keeps
  stable-point `x/y` errors below `0.055/0.093`. Four and six oscillatory event
  positions shift at `I=0/0.01`, so timing identity is excluded explicitly.
  The S5/H1 descriptor adds a generated Q8.8 port-only formal job whose
  depth-4 SymbiYosys/Z3 check passes. The public fidelity count moves to 21;
  schema-gap counts move to 31 / 122 / 124; the formal catalogue moves to 23
  jobs and the whole HDL inventory to 60 jobs / 208 statements.

### Ermentrout-Kopell theta-Euler schema-to-RTL enrolment
- Enrolled `ermentrout_kopell_map_neuron` against Ermentrout and Kopell
  (1986), DOI `10.1137/0146017`. Paired schemas reproduce the maintained hand
  recurrence exactly while separating the paper's continuous theta equation
  from the implementation's `dt=0.1`, gain, forward-Euler, `theta=pi` event,
  and modulo `2*pi` choices. The independent `I=0.5` reference records 45
  events over 2,000 steps; varied-drive tests require exact hand/TOML/JSON
  states and events.
- Q16.16 RTL preserves 0/45/64 spikes at `I=-0.5/0.5/1.0` over 2,000 steps,
  with maximum circular phase error below 0.081/0.089/0.025 rad. The cosine LUT
  can shift event positions, so event-vector and full-trajectory identity are
  withheld. Generated integer C/Rust kernels match generated Verilog state and
  event words cycle-for-cycle over the 240-step protocol for both current signs.
- The equation runtime and generated backends now expose `<state>_prev` at the
  macro boundary and lower modulo only for a representable finite positive
  literal, with the negative-remainder correction required to match Python.
  The S5/H1 descriptor adds a Q8.8 port-only formal job; its depth-4 Z3 BMC
  passes. All 21 earlier jobs regenerate byte-identically, the inventory moves
  to 22, and schema-gap counts move to 30 / 123 / 125. Existing acceleration
  and benchmark artefacts remain unchanged.

### Courbage-Nekorkin-Vdovin bounded schema-to-RTL enrolment
- Enrolled the already acceleration-complete 2007 discontinuous map
  (`courage_nekorkin_map`, DOI `10.1063/1.2795435`). The paired TOML/JSON
  schemas reproduce equations 3–5, including simultaneous state commits, all
  three fast-map branches, the Heaviside discontinuity, and the maintained
  upward `x_threshold` crossing. The hand model and both schemas agree exactly
  across the enrolled operating set.
- Q16.16 RTL is event-exact at `I=-0.3/0/0.3` over bounded
  30/20/30-iteration windows, with both state errors below `0.014`. Q32.32 RTL
  is event-exact at all three inputs over 30 iterations, with fast-coordinate
  error below `0.00003` and recovery-coordinate error below `0.000001`. A
  separate regression fixes the autonomous 30-iteration Q16.16 boundary at
  four float64 events, six RTL events, and six event-position mismatches.
- The model-scoped co-simulation and independent-reference tests add a
  DOI-backed recurrence, S5/H1 descriptor/readiness facets, source-bounded
  public documentation, and generated Q8.8 RTL. The port-only depth-4
  SymbiYosys/Z3 reset-spike safety job passes. The formal inventory moves to
  21 models and schema-gap counts move to 29 schema models / 124 net missing /
  126 source modules without a schema. The completed acceleration chain and
  its committed benchmark artefact remain unchanged.

### Cazelles map bounded schema-to-RTL enrolment
- Enrolled the already acceleration-complete Cazelles, Courbage, and Rabinovich
  (2001) fast/slow map (`cazelles_map`, DOI
  `10.1209/epl/i2001-00548-y`). The paired TOML/JSON schemas mirror the
  maintained hand model's simultaneous clipped logistic fast recurrence, slow
  update from the old fast coordinate, and committed-state `x >= x_threshold`
  level event. Hand model and both schemas agree exactly on every state and
  event across the enrolled operating set.
- Over 30 iterations, emitted Q16.16 RTL reproduces the complete 2/1/1 event
  vectors at `I=0.5/1.0/2.0`, with both coordinates within `0.0004` absolute
  error of float64. The three points exercise the interior expression and both
  clip bounds. A separate regression fixes the sensitive `I=0.05` boundary at
  seven float64 events, eight RTL events, and seven event-position mismatches,
  so the bounded evidence cannot imply long-window chaotic identity.
- The Cazelles co-simulation and independent-reference tests live in dedicated
  model-scoped modules rather than the legacy catalogue-wide test accumulators.
  The unit adds the DOI-backed independent map reference, S5/H1
  descriptor/readiness facets, public fidelity and model documentation, and
  generated Q8.8 RTL. Its port-only depth-4 SymbiYosys/Z3 reset-spike safety
  job passes. The formal inventory
  moves to 20 models and schema-gap counts move to 28 schema models / 125 net
  missing / 127 source modules without a schema. The completed acceleration
  chain and its committed benchmark artefact remain unchanged.

### Mihalas-Niebur co-simulation evidence correction
- Replaced the stale loose Q16.16 guard with exact operating-point contracts after the shared
  candidate-reset/output correction. The hand model and paired TOML/JSON schemas agree exactly on
  every event and all four states over a varied 1,600-step sequence containing 168 resets. At
  `I=3` over 300 steps, hand/schema/RTL now report 36/36/36 rather than the former 36/36/35.
- Over 1,000 steps, hand/schema/Q16.16 RTL agree exactly on
  0/0/0/31/60/87/131/157/207/256 spikes at
  `I=0/0.5/1/1.5/2/2.5/3.5/4/5/6`. A separate test pins the isolated `I=3` boundary at
  111/111/112, preserving the marginal fixed-point crossing as an explicit exclusion rather than
  a hidden tolerance. The S5/H1 descriptor, readiness index, public fidelity row, model page, and
  co-simulation guide now state that same contract. The existing depth-3 SymbiYosys/Z3 job remains
  the structural safety proof; all model, acceleration, schema, compiler, generated RTL, formal
  source, and benchmark artefacts are unchanged.

### GLIF faithful RK4 schema-to-RTL correction
- Corrected the already enrolled `glif` TOML/JSON schemas from stale Euler and `v > theta`
  semantics to the maintained Allen Institute GLIF5 contract: simultaneous four-state classical
  RK4, candidate-level `v >= theta` detection, and candidate-first adaptive reset. The hand model
  and both schema formats agree exactly on all states and 181 reset events over a varied
  4,000-step drive.
- Replaced the silent Euler reference with an independent 54-spike RK4 re-derivation. Across the
  six 1,000-step Q16.16 operating points `I=0/15/22/30/45/50`, hand model, schema runner, and RTL
  agree exactly on 0/0/23/54/86/95 spikes. The compiler and bit-true C/Rust generators now evaluate
  reset expressions from the integrated candidate and expose identical post-reset state, removing
  the pre-step-reset mismatch that a one-spike band had masked. The S5/H1 descriptor/readiness
  facets record the exact observable. Deterministic regeneration changes all ten reset-using formal
  RTL models plus seven non-resetting single-step edge models so spike-cycle outputs expose the
  committed candidate. Connor-Stevens and Hodgkin-Huxley already expose that candidate in their
  macro-step branches and remain byte-identical, accounting for all 19 jobs without changing the
  inventory. The completed acceleration chain and benchmark artefacts are unchanged. The reset
  correction exposes one honest legacy
  Izhikevich Q8.8 boundary (float64 25 spikes versus RTL 24 at `I=50` over 200 steps), while a new
  Q16.16 guard proves exact 25/25 parity at the same point.

### Rulkov map class-correct schema-to-RTL enrolment
- Descriptor regeneration now inherits `integration.dt` from a bundled map schema when the hand
  class intentionally has no timestep parameter, so the Rulkov map retains its one-iteration
  metadata without adding a non-functional public field or changing non-map fallbacks. The
  readiness facet writer also preserves any descriptor whose recorded evidence already meets or
  exceeds the indexed floor, preventing regeneration from replacing Rulkov's S5 trajectory and H2
  synthesis evidence with H0 facets.
- Re-enrolled the Rulkov 2002 fast/slow map (`rulkov_map`, DOI
  `10.1103/PhysRevE.65.041922`) with paired TOML/JSON schemas that mirror the maintained
  hand model's simultaneous rational/plateau/hard-reset branches and rising `x >= 0`
  crossing decision. The old schema used level detection and therefore counted every
  positive plateau step rather than the hand model's upward crossings.
- At `I=1.5`, the bounded 30-iteration validation window executes every fast-map branch
  ten times. Hand model and both schema formats agree exactly on all post-step states and
  the ten-event vector; emitted Q16.16 RTL reproduces that event vector exactly with
  absolute `x`/`y` error below `0.001`. The evidence uses the class-appropriate short-window
  trajectory metric and explicitly withholds long-window spike-count identity.
- Corrected the descriptor's stale equations and timestep, regenerated the DOI-backed
  crossing reference, and recorded S5/H2: the Q16.16 core passes Yosys 0.33
  `synth_xilinx` with a raw committed report. The now-perfect descriptor is registered
  with the formal catalogue through generated Q8.8 RTL, a port-only harness, and a
  depth-4 SymbiYosys/Z3 safety job, bringing the inventory to 19 models. Schema-gap
  counts are unchanged because the paired schema already existed. The completed
  acceleration chain and its benchmark artefacts are unchanged.

### Wilson-HR faithful schema-to-RTL enrolment
- Enrolled the Wilson-HR two-state polynomial cortical model (`wilson_hr`, Wilson 1999, DOI
  `10.1006/jtbi.1999.1002`) with paired TOML/JSON schemas that mirror the maintained hand model's
  simultaneous classical RK4 flow, polynomial membrane nullcline, level `v >= v_peak` decision,
  and hard `v = -0.7` reset that preserves the candidate recovery state. A varied 4,000-step drive
  produces 35 spikes and resets with exact hand/TOML/JSON state agreement. Over 5,000
  constant-current steps the hand model, schema runner, and Q16.16 RTL agree exactly on 0, 1, and
  4 spikes at `I=0.0`, `2.0`, and `10.0`.
- Added the DOI-backed `wilson_hr_driven_spiking_doi` trace with an independent two-state RK4
  re-derivation, corrected the public model provenance, marked the descriptor's RK4 and co-sim
  facets, and updated the public fidelity row. The S5/H1 descriptor is registered with the formal
  catalogue: generated Q8.8 RTL, a port-only harness, and a depth-4 SymbiYosys reset-spike safety
  job bring the committed inventory to 18 models. Schema-gap counts move to 27 schema models / 126
  net missing / 128 source modules without a schema. The completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### Terman-Wang faithful schema-to-RTL enrolment
- Enrolled the Terman-Wang two-state LEGION relaxation oscillator (`terman_wang`, Terman & Wang
  1995, DOI `10.1016/0167-2789(94)00205-5`) with paired TOML/JSON schemas that mirror the
  maintained hand model's simultaneous classical RK4 flow, cubic fast nullcline, `tanh`-gated
  recovery, rising-edge `v >= v_peak` decision, and no-reset semantics. Over 8,000 steps the hand
  model, schema runner, and emitted Q16.16 RTL agree exactly on the silent/single/train crossing
  counts: 0 at `I=-1.0`, 1 at `I=0.0`, and 3 at `I=0.5`.
- Added the DOI-backed `terman_wang_legion_oscillation_doi` trace with an independent two-state RK4
  re-derivation, corrected the public model citation, marked the descriptor's RK4 and co-sim
  facets, and updated the public fidelity row. The S5/H1 descriptor is registered with the formal
  catalogue: generated Q8.8 RTL, a port-only harness, and a depth-4 SymbiYosys reset-spike safety
  job bring the committed inventory to 17 models. Schema-gap counts move to 26 schema models / 127
  net missing / 129 source modules without a schema. The completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### Pernarowski faithful schema-to-RTL enrolment
- Enrolled the Pernarowski three-state pancreatic beta-cell burster (`pernarowski`, Pernarowski
  1994, DOI `10.1137/S003613999223449X`) with a TOML/JSON schema that mirrors the maintained hand
  model's simultaneous classical RK4 flow, exact `v * v * v` operation order, rising-edge
  `v >= v_threshold` decision, and no-reset semantics. The hand model, schema runner, and emitted
  Q16.16 RTL have exact spike-count parity at all four enrolled 5,000-step operating points: 17
  crossings at each of `I=-0.1`, `0.0`, `0.1`, and `0.2`.
- Added the DOI-backed `pernarowski_autonomous_bursting_doi` trace with an independent three-state
  RK4 re-derivation, marked the descriptor's RK4 and co-sim facets, and updated the public fidelity
  row. The S5/H1 descriptor is also registered with the formal catalogue: generated Q8.8 RTL, a
  port-only harness, and a depth-4 SymbiYosys reset-spike safety job bring the committed inventory
  to 16 models and pass a direct Z3 BMC. Schema-gap counts move to 25 schema models / 128 net missing
  / 130 source modules without a schema. The already completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### FitzHugh-Rinzel faithful schema-to-RTL enrolment
- Enrolled the FitzHugh-Rinzel three-state qualitative burster (`fitzhugh_rinzel`, Rinzel 1987,
  DOI `10.1007/978-3-642-93360-8_26`) with a TOML/JSON schema that mirrors the maintained hand
  model's coupled classical RK4 flow, exact `v * v * v` operation order, rising-edge
  `v >= v_threshold` decision, and no-reset semantics. The hand model, schema runner, and emitted
  Q16.16 RTL have exact spike-count parity across the enrolled `I=0.4` to `I=0.6` band (seven,
  eight, and eight crossings over 3000 steps); `I=0.7` is recorded as an excluded
  marginal-crossing boundary rather than hidden by a tolerance.
- Added the DOI-backed `fitzhugh_rinzel_driven_bursting_doi` trace with an independent three-state
  RK4 re-derivation, corrected the model page's stale reset claim, marked the descriptor's RK4 and
  co-sim facets, and updated the public fidelity row. Schema-gap counts move to 24 schema models /
  129 net missing / 131 source modules without a schema. The already completed Rust/Go/Julia/Mojo
  acceleration chain and its committed benchmark artefacts are unchanged.

### StochasticLIF same-name Rust engine binding
- `StochasticLIFNeuron` was enrolled in the public registry
  (`sc_neurocore.neurons.models.__all__`) for catalogue readiness but had no same-name PyO3
  constructor, so the registry-parity coverage map flagged it as an uncovered non-Python-only model.
  The Rust neuron already existed in the engine (`engine/src/neurons/trivial.rs`) and was wired into
  the `NetworkRunner`; this adds the missing `PyStochasticLIFNeuron` wrapper (`new(seed)` / `step` /
  `reset` / `get_state`, mirroring `StochasticIFNeuron`), re-exports it from the engine package root,
  and enrols it in the RNG-dependent parity set. Exact traces are not claimed — the Rust Gaussian
  stream is `Xoshiro256++` Box-Muller rather than NumPy's PCG64 Ziggurat — so the spike count is the
  stated parity observable. The registry map is now 160 public Python registry names, 146 same-name
  Rust constructors, and 176 Rust PyO3 model wrappers.

### Connor-Stevens polyglot kernels pinned to the Python golden spike count
- The Go (`accel/go/services/connor_stevens.go`), Rust (`accel/rust/safety/connor_stevens.rs`) and
  Julia (`accel/julia/neurons/connor_stevens.jl`) Connor-Stevens kernels already carried the real
  six-state macro-step RK4 dynamics (100 sub-steps) but were only self-consistency / smoke tested;
  each now asserts the Python golden counts — silent at zero drive, two action potentials at `I=10`
  over 100 macro steps, nine at `I=20`. Connor-Stevens gating is `exp`-based, so the trace is not
  bit-exact across C libraries; the spike count is the stated parity observable and all three
  languages reproduce it.
- Added a native Julia parity test (`accel/julia/connor_stevens_parity_test.jl`), a Go golden-parity
  test plus an honest `BenchmarkConnorStevensStep` (262.7 µs/macro-step), and the idiomatic
  `Default` impl on the Rust side. The Mojo kernel remains an honest parity note pending the Mojo
  neuron-kernel lane's promotion to a build target. Connor-Stevens's Python model dispatches only to
  the Rust engine, so the Go/Julia kernels are language-native; no Python runtime path, FFI
  dispatch, or committed benchmark artefact changed.

### Morris-Lecar polyglot kernels pinned to the Python golden spike count
- The Go (`accel/go/services/morris_lecar.go`), Rust (`accel/rust/safety/morris_lecar.rs`) and Julia
  (`accel/julia/neurons/morris_lecar.jl`) Morris-Lecar kernels already carried the real RK4 dynamics
  but were only self-consistency / smoke tested; each now asserts the Python golden counts — silent
  at zero drive, three action potentials at `I=50` over 2000 steps, five at `I=100`. Morris-Lecar
  gating is `tanh`/`cosh`, so the trace is not bit-exact across C libraries; the spike count is the
  stated parity observable and all four languages reproduce it.
- Added a native Julia parity test (`accel/julia/morris_lecar_parity_test.jl`) and an executable
  `simulate`/`main` parity harness to the Mojo kernel (`accel/mojo/kernels/morris_lecar.mojo`, so
  `mojo run` prints `PARITY OK`); removed a dead `k4_v` assignment in that kernel's `next_w` path.
  On the Rust side, dropped a vestigial `#![allow(unused_variables, dead_code, non_snake_case)]`
  (the code is real, not a masked stub) and added the idiomatic `Default` impl. Morris-Lecar's
  Python model dispatches only to the Rust engine, so the Go/Julia/Mojo kernels are language-native;
  no Python runtime path, FFI dispatch, or committed benchmark artefact changed.

### FitzHugh-Nagumo Go accel-services kernel (services test suite unblocked)
- Added `accel/go/services/fitzhugh_nagumo.go`, a real RK4 `SimulateFitzHughNagumoNeuron` in
  parity with `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron` (the cube written
  `v*v*v`, exact arithmetic, fail-closed on a non-finite input, state, or candidate). The Go
  services test suite could not compile because `services_test.go` referenced this function while
  the FitzHugh-Nagumo Go kernel existed only as a `package main` cgo shared library under
  `accel/go/neurons/fitzhugh_nagumo/`. A golden-parity test pins the kernel to the Python reference
  (one action potential at `I=10` over 100 steps and a five-spike partial train at `I=0.5` over
  2000 steps, final `v` bit-identical to NumPy) and an honest per-step benchmark records the timing.
- Strengthened the `accel/rust/safety` FitzHugh-Nagumo test from a `spike is 0 or 1` smoke check to
  the same Python-golden spike count, dropped a vestigial `#![allow(dead_code)]`, and added the
  idiomatic `Default` impl. The services surface is Go-native, not FFI-dispatched, so no Python
  runtime path, FFI dispatch, cross-language `simulate` benchmark, or committed benchmark artefact
  changed.

### Sequential (Gauss-Seidel) integration mode + faithful Wang-Buzsáki re-enrolment
- Added a sequential (Gauss-Seidel) integration mode (`[integration] method = "gauss_seidel"`) to
  the schema DSL, in both the Python runner (`EquationNeuron`) and the emitted Verilog. The state
  variables advance in declaration order, each derivative reading the already-committed earlier
  variables within the same sub-step — lowering a conductance hand model's gates-then-voltage
  update (gates from the old voltage, voltage from the new gates). The emitter renders each earlier
  variable as its `<var>_next` wire in a later variable's derivative (a commit-before-read chain,
  no cycle). Composes with `substeps`; the simultaneous `euler`/`rk4` methods stay bit-for-bit
  unchanged (default `substeps = 1`).
- Re-enrolled the Wang-Buzsáki (1996) fast-spiking interneuron (`wang_buzsaki` schema, DOI
  `10.1523/JNEUROSCI.16-20-06402.1996`) faithfully. The bundled schema was a single-step
  `method="euler"` re-derivation with a sigmoid-caricature `m_inf`, unfaithful gate initial
  conditions (`h=0.6`, `n=0.32`), a `v > -10` threshold, and a singular `n` rate; it is now
  `method="gauss_seidel"`, `substeps=50`, state ordered `h, n, v` (`h=0.8`, `n=0.1`, `v=-65`), the
  true instantaneous `m_inf = alpha_m/(alpha_m+beta_m)` (`alpha_m = 1/exprel(-(v+35)/10)`), the
  exprel `n` rate `0.1/exprel(-(v+34)/10)`, a macro-boundary `v >= v_threshold` crossing
  (`v_threshold=-20`), no reset — matching `WangBuzsakiNeuron` exactly (`hand == schema`, three
  action potentials at `I=10` over 20 macro steps). The Q16.16 RTL tracks the schema **within one
  spike** over the bounded window (three-way exact at `I=10`, `macro=20`); the residual is the
  `m_inf` fixed-point divide plus a 256-entry exprel look-up, not a datapath-precision limit.
  Re-derived reference trace `wang_buzsaki_driven_spiking_doi`
  (`independent_macrostep_gauss_seidel_reference`, a 3-state helper bit-exact vs the runner,
  spike_count 4), replacing the deleted resting-gate trace; descriptor dynamics synced to the
  exprel form; the 15%-band cosim test becomes a macro-step three-way parity test. Wang-Buzsáki is
  the last of the four schema-compiled conductance oscillators (Morris-Lecar, Connor-Stevens, Hodgkin-Huxley,
  Wang-Buzsáki), all now faithfully enrolled; the polynomial edge-crossing oscillators
  (FitzHugh-Nagumo, McKean) were enrolled earlier.

### Faithful Hodgkin-Huxley re-enrolment (macro-step RK4)
- Re-enrolled the Hodgkin-Huxley (1952) membrane (`hodgkin_huxley` schema,
  DOI `10.1113/jphysiol.1952.sp004764`) as a driven repetitive-spiking oscillator using the
  macro-step mode. The bundled schema was a single-step `method="euler"` resting-gate re-derivation
  compared schema-vs-verilog under a 5% band; it is now `method="rk4"`, `substeps=100`,
  macro-boundary `v >= v_threshold` crossing — matching `HodgkinHuxleyNeuron(integrator="rk4")` (the
  simultaneous RK4, not the Gauss-Seidel `baseline_euler` default the DSL cannot reproduce).
  `hand == schema` exact (five action potentials at `I=20` over 60 macro steps). The Q16.16 RTL
  tracks the schema **within one spike** over the bounded window (`I=15`, 20 macro steps, three-way
  exact); like Connor-Stevens the residual is genuine conductance-LUT quantisation
  (LUT-resolution-limited, identical at Q16.16 / Q24.24 / Q32.32). Re-derived reference trace
  `hodgkin_huxley_driven_spiking_doi` (`independent_macrostep_rk4_reference`, bit-exact vs the
  runner); descriptor integration updated (euler → rk4). Also fixed a pre-existing
  `hodgkin_huxley.json` toml/json drift (singular `a*(V-V0)/(1-exp(...))` rate form → the stable
  `exprel` rewrite the `.toml` already used) and a stale wrong Connor-Stevens DOI (`sp009368` →
  `sp009366`) in the changelog. Schema-gap counts unchanged; no polyglot / benchmark change (the
  hand model already carries the RK4 path).

### Macro-step integration mode + faithful Connor-Stevens re-enrolment
- Added a macro-step integration mode (`[integration] substeps = N`) to the schema DSL — in both
  the Python runner (`EquationNeuron`) and the emitted Verilog. One macro `step()` advances `N`
  inner integration sub-steps before a single spike decision, with the rising-edge crossing taken
  only on the macro boundary. This lets the schema faithfully replicate the maintained conductance
  hand models whose `step()` is a fixed number of fine sub-steps per macro step (HH / Connor-Stevens:
  100 `dt=0.01` sub-steps per 1 ms; Wang-Buzsaki: 50 per 0.5 ms), so a repetitively firing
  oscillator counts one spike per action potential, not one per sub-step above threshold. The RTL
  keeps one sub-step per clock and gates the crossing to the macro boundary via a sub-step counter;
  the lowering is bit-exact against the runner (proven on the polynomial FitzHugh-Nagumo at Q16.16,
  exact across sub-step groupings). Supported for the edge (crossing, non-resetting), non-pipelined
  datapath; other combinations raise `NotImplementedError`. `substeps = 1` (default) leaves every
  existing model bit-for-bit unchanged.
- Re-enrolled Connor-Stevens (1971) faithfully with the macro-step mode: the bundled schema is now
  `method="rk4"`, `substeps=100`, macro-boundary `v >= v_threshold` crossing — matching the
  maintained `ConnorStevensNeuron`. `hand == schema` exact (ten action potentials at `I=100` over
  60 macro steps), which the earlier single-step Euler schema could not achieve. The Q16.16 RTL
  tracks the schema **within one spike** over the bounded window; the residual is genuine
  conductance-LUT quantisation (LUT-resolution-limited, identical at Q16.16 / Q24.24 / Q32.32),
  three-way exact over a bounded window and accumulating beyond it — an honest per-model
  hardware-fidelity band. Re-derived reference trace `connor_stevens_driven_spiking_doi`
  (`independent_macrostep_rk4_reference`, bit-exact vs the runner); descriptor integration updated
  (euler → rk4). Also fixed a pre-existing `connor_stevens.json` toml/json drift (singular
  `a*(V-V0)/(1-exp(...))` rate form → the stable `exprel` rewrite the `.toml` already used).
  Schema-gap counts unchanged; no polyglot / benchmark change (the hand model was already RK4).

### Morris-Lecar faithful re-enrolment (RK4, no reset, rising-edge crossing)
- Re-enrolled the Morris-Lecar (1981) calcium-potassium oscillator (`morris_lecar` schema,
  DOI `10.1016/S0006-3495(81)84782-0`) as the first **conductance** edge-crossing oscillator
  in the schema-driven Python↔Verilog co-simulation set. The prior schema was `method="euler"` with a
  no-op `[reset]` (`v -> v`, `w -> w`) that disabled edge detection and over-counted every
  above-threshold step — a caricature that only "passed" a ~15% band because both sides
  over-counted identically. The faithful schema mirrors `MorrisLecarNeuron`'s maintained
  defaults: four-stage RK4, **no reset**, rising-edge (`v >= v_threshold`) crossing, and
  `phi = 1/15`.
- At the sustained depolarising regime (`I=100`, 3000 steps) the hand `MorrisLecarNeuron`,
  the schema runner, and the emitted Q16.16 RTL report the same seven upward crossings.
  Because the sigmoidal gating lowers to 256-entry cosh/tanh look-up tables and the hand
  model integrates with `math` transcendentals through a distinct RK4 driver, this is an
  exact **spike-count** parity (robust across the whole `I in [90, 110]` band), not the
  bit-identical state the polynomial FitzHugh-Nagumo / piecewise-linear McKean oscillators
  achieve; `I=120` is a knife-edge that splits a marginal crossing between the paths.
- Re-derived the reference trace as `morris_lecar_driven_oscillation_doi`
  (`independent_rk4_reference`, `I=100`, 3000 steps, seven crossings, first at step 141) via a
  new `_morris_lecar_rk4_features` helper verified bit-exact against the runner. The
  `MorrisLecarNeuron` descriptor now carries the schema's RK4 integration. Schema-gap counts
  are unchanged (a re-enrolment, not a new schema).
- The schema-DSL runner is Python-only; the hand `MorrisLecarNeuron` and its Rust/Julia/Go/Mojo
  mirrors were already RK4 / no-reset, so no polyglot counterpart or benchmark artefact changed.

### McKean piecewise-linear oscillator enrolment (RK4, no reset, rising-edge crossing)
- Enrolled the McKean (1970) piecewise-linear FitzHugh-Nagumo caricature (`mckean` schema,
  DOI `10.1016/0001-8708(70)90023-X`) into the model-schema corpus and Python↔Verilog
  co-simulation. The bundled schema is RK4, no reset, rising-edge (`v >= v_peak`) detection,
  with the three-branch piecewise-linear membrane `f(v) = min(max(-v, v - a), 1 - v)` — the
  second edge-crossing oscillator after FitzHugh-Nagumo. The min/max branch selection lowers
  to a fixed-point comparison + select (no look-up table), so at the sustained
  relaxation-oscillation operating point (`epsilon=0.2`, `gamma=0.5`, `I=0.6`) the hand
  `McKeanNeuron`, the schema runner, and the emitted Q16.16 RTL report the same 16-crossing
  train over 3000 steps **bit-exactly** — a genuine three-way parity, not a tolerance band.
- The default hand-model regime (`epsilon=0.01`) is a single-transient knife-edge; the
  enrolled regime is a robust limit cycle whose upward crossings survive fixed-point rounding.
- Committed an independent RK4-parity reference trace (`independent_rk4_reference`, `I=0.6`,
  3000 steps, 16 crossings, first at step 12) via a new `_mckean_rk4_features` helper verified
  bit-exact against the runner. Schema-gap counts move to 23 schema models / 129 net missing /
  131 source modules without a schema; the `McKeanNeuron` descriptor now carries the schema's
  RK4 integration and min/max dynamics (golden-trace SHA unchanged).
- The schema-DSL runner is Python-only; the hand `McKeanNeuron` and its Rust/Julia/Go/Mojo
  mirrors were already RK4 / piecewise-linear / no-reset, so no polyglot counterpart or
  benchmark artefact changed.

### FitzHugh-Nagumo faithful re-enrolment (RK4, no reset, rising-edge crossing)
- Replaced the bundled `fitzhugh_nagumo` schema's explicit-Euler + `v = -1` reset
  caricature with the genuine FitzHugh (1961) relaxation oscillator: four-stage RK4,
  **no reset**, rising-edge (`v >= v_threshold` upward crossing) spike detection, and
  the exact IEEE cube `v * v * v`. The schema now matches `FitzHughNagumoNeuron`
  bit-for-bit in float64, and over 3000 steps at `I=0.5` the hand model, the schema
  runner, and the emitted Q16.16 RTL report the same eight-crossing partial train
  **exactly** — a genuine three-way parity, not a tolerance band, because the
  right-hand side is polynomial (no look-up table). The earlier Euler+reset "parity"
  only held because both sides shared the same unfaithful reset dynamics; the RK4
  distinctness demonstration therefore moved to models whose spike count is genuinely
  integrator-sensitive (theta for RK4, resonate-and-fire for exponential Euler), since
  a faithful relaxation oscillator counts the same crossings under any integrator.
- The committed reference trace is re-derived with an independent RK4 recurrence
  (`independent_rk4_reference`, `I=0.5`, 3000 steps, eight crossings, first at step 29).
- The schema-DSL runner now fails closed on a non-finite state (`FloatingPointError`,
  matching the hand neuron models' `_validate_candidate` contract) rather than silently
  propagating `inf`/`nan` into the threshold decision, so the unbounded oscillator's
  large-step divergence is a controlled error, not a corrupt trace. FitzHugh-Nagumo also
  moved from the "transcendental (LUT)" compile group to the polynomial group in the
  DSL→Verilog tests — its cube lowers to plain fixed-point multipliers.
- The schema-DSL runner is Python-only; no benchmark dispatch or benchmark artefact
  changed. The hand `FitzHughNagumoNeuron` and its Rust/Julia/Go/Mojo mirrors were
  already RK4 / `v*v*v` / no-reset, so no polyglot counterpart changed.

### Rising-edge (`crossing`) threshold detection for non-resetting oscillators
- Made the schema DSL's `[threshold] detection = "crossing"` field functional in both the
  Python runner (`EquationNeuron`) and the schema→Verilog emitter. A non-resetting
  oscillator now spikes once per **upward** threshold crossing (matching the hand models'
  `v >= thr and v_prev < thr` edge test) instead of on every step it stays above threshold.
  This unlocks faithful enrolment of the biophysical oscillator family (FitzHugh-Nagumo,
  McKean, and the conductance oscillators), which the previous level-only path could only
  over-count. Validated end-to-end by a faithful FitzHugh-Nagumo hand-model / schema /
  Q16.16 RTL three-way parity at exact spike counts on a sustained relaxation-oscillation
  train (8 of 3000 steps). Edge detection engages **only** for a crossing model with no
  reset — a reset that clears the condition makes `level` and `crossing` identical, so every
  existing reset-based integrate-and-fire model keeps its exact prior behaviour (the field
  was previously decorative, and several reset models declared `crossing` harmlessly). The
  emitted RTL carries a 1-bit `_thr_prev` edge-history register, seeded from the initial
  state to bit-match the golden. The folded `compile_to_datapath` PE stays level-only for
  now (the co-simulation path uses the per-instance module); crossing support in the folded
  datapath is a separate follow-up.

### Mihalas-Niebur RK4 co-simulation enrollment
- Enrolled `mihalas_niebur` (Mihalaş & Niebur 2009 generalised linear
  integrate-and-fire) into the model-schema corpus and Python↔Verilog
  co-simulation as the first `method="rk4"` bundled model: a four-state RK4 flow
  (membrane, adaptive threshold, two spike-triggered currents) with a
  state-to-state `v >= theta` threshold and a `max(theta, theta_reset)`
  adaptive-threshold reset that floors on every spike. The schema runner
  reproduces the hand model bit-for-bit in float64, and a committed independent
  RK4-parity reference trace covers the new bundled schema. Unlike the exact Euler
  anchors, the Q16.16 RTL tracks the float spike train to within a single spike
  (35 vs 36 over 300 steps) rather than bit-exactly: the state-vs-state threshold
  compares two quantised states and the four-stage RK4 update injects four times
  the per-step rounding, so a marginal crossing shifts by one step under
  fixed-point rounding (timing jitter, not LUT coarseness — the right-hand side is
  linear). The co-simulation claim is therefore an honest tolerance band, not
  exact parity. The schema-DSL runner has no polyglot mirror; no benchmark
  dispatch or benchmark artefact changed.

### DYNAP-SE differential-pair integrator co-simulation enrollment
- Enrolled `dpi_neuron` (Chicca et al. 2014 DYNAP-SE differential-pair
  integrator, current-mode subthreshold LIF) into the schema-driven Python↔Verilog
  co-simulation campaign: the explicit-Euler discretisation co-simulates at exact
  Q16.16 spike-count parity (three-way hand-model / schema / RTL) on a partial
  spike train, with a committed independent Euler-parity reference trace covering
  the new bundled schema. The drive is non-negative, so the source model's
  `max(i_mem, 0)` current rectification is inert and the linear schema is a
  faithful discretisation. The schema-DSL runner has no polyglot mirror; no
  benchmark dispatch or benchmark artefact changed.

### Izhikevich 2007 co-simulation enrollment and Verilog parameter-collision fix
- Enrolled `izhikevich2007` (Izhikevich 2007 biophysical quadratic IF, NeuroML
  `izhikevich2007Cell`) into the schema-driven Python↔Verilog co-simulation campaign: the
  explicit-Euler discretisation co-simulates at exact Q16.16 spike-count parity
  (three-way hand-model / schema / RTL), with a committed independent
  Euler-parity reference trace covering the new bundled schema. The model's RK4
  default remains a separate RK4-emitter candidate.
- Fixed the schema→Verilog emitter to keep the parameter port map injective:
  `str.upper()` collapsed case-distinct names (Izhikevich `C` capacitance vs `c`
  reset voltage) onto one `P_C` port that iverilog rejected. Single-case
  parameter names keep the canonical `P_TAU`-style ports. The schema-DSL runner
  has no polyglot mirror; no benchmark dispatch or benchmark artefact changed.

### Discrete-map integration mode and Rulkov 2002 map fix
- Added a `method="map"` discrete-map integration mode to `EquationNeuron`
  (iterates `state_{n+1} = f(state_n)` directly), plus Verilog `IfExp` lowering
  and a discrete-map RTL datapath so piecewise maps compile to
  Icarus-Verilog-valid RTL. The `rulkov_map` schema adopts the DOI-verified
  Rulkov (2002) piecewise fast/slow map with `method="map"`, replacing the smooth
  variant that was integrated with explicit Euler and diverged; its reference
  trace is a bounded driven-spiking trace with exact independent map-iteration
  parity, so all 17 deterministic reference traces now hold independent parity.
  The schema-DSL runner has no polyglot mirror; no benchmark dispatch or
  benchmark artefact changed.

### Compiler HDL e2e CI
### Changed
- Split the 2,690-line NIR FPGA compiler into a 400-line composition and
  resource-policy boundary plus focused modules for connection validation,
  result contracts, neuron RTL, weight ROMs, SC-NIR hierarchy boundaries, and
  direct, address-event, and folded interconnects. Valid direct, address-event,
  folded, and hierarchical compilations retain byte-identical generated
  artefacts; public result types keep their historical import and pickle path.
  Empty graphs and malformed connection matrices now fail before SC-NIR
  conversion or partial HDL emission.
- Re-enrolled the Connor-Stevens A-current oscillator (`connor_stevens` schema, Connor &
  Stevens 1971, DOI `10.1113/jphysiol.1971.sp009366`) faithfully using the new macro-step mode.
  The bundled schema was single-step `method="euler"` — neither the maintained
  `ConnorStevensNeuron`'s RK4 integrator nor its 100-sub-step-per-millisecond macro-stepping — so
  it could only be compared schema-vs-verilog and over-counted crossings. The faithful schema is
  now `method="rk4"` with `substeps=100` and a macro-boundary `v >= v_threshold` crossing, so the
  schema reproduces the hand model's action-potential count exactly (`hand == schema`, ten
  crossings at `I=100` over 60 macro steps). The Q16.16 RTL tracks the schema **within one spike**
  over the bounded window (`I=100`, 20 macro steps): Connor-Stevens is a stiff six-state model
  whose exprel / cube-root gating lowers to 256-entry look-up tables, and the residual drift is
  **LUT-resolution-limited, not datapath-precision-limited** (identical spike count at Q16.16 /
  Q24.24 / Q32.32) — an honest per-model hardware-fidelity band, three-way exact over a bounded
  window and accumulating beyond it. The reference trace was re-derived as
  `connor_stevens_driven_spiking_doi` (`independent_macrostep_rk4_reference`, bit-exact against the
  runner) and the descriptor's integration updated (euler → rk4). Also fixed a pre-existing
  toml/json drift: the `connor_stevens.json` `m`/`n` rate functions were the singular
  `a*(V-V0)/(1-exp(...))` form (a 0/0 at `V=V0`) while the loaded `.toml` used the stable `exprel`
  rewrite; the JSON now matches. Schema-gap counts unchanged (a re-enrolment). The schema-DSL
  runner is Python-only; the hand `ConnorStevensNeuron` and its polyglot mirrors were already RK4,
  so no polyglot counterpart or benchmark artefact changed.
- Re-enrolled the Morris-Lecar calcium-potassium oscillator (`morris_lecar` schema,
  Morris & Lecar 1981, DOI `10.1016/S0006-3495(81)84782-0`) faithfully as the first
  **conductance** edge-crossing oscillator in the schema co-simulation set. The bundled
  schema was `method="euler"` with a no-op `[reset]` (`v -> v`, `w -> w`) that disabled
  edge detection, routed to the level datapath, and over-counted every above-threshold
  step; both Python and Verilog over-counted identically so a ~15% tolerance band passed
  while validating a caricature. The faithful schema mirrors `MorrisLecarNeuron`'s
  maintained defaults — four-stage RK4, **no reset**, rising-edge (`v >= v_threshold`)
  crossing, and `phi = 1/15` — so it counts one spike per action potential. At the
  sustained depolarising regime (`I=100`, 3000 steps) the hand model, the schema runner
  and the emitted Q16.16 RTL report the same seven upward crossings. Because the sigmoidal
  gating lowers to 256-entry cosh/tanh look-up tables and the hand model uses `math`
  transcendentals through a distinct RK4 driver, this is an exact **spike-count** parity
  (robust across the whole `I in [90, 110]` band), not the bit-identical state the
  polynomial FitzHugh-Nagumo / piecewise-linear McKean oscillators achieve. The reference
  trace was re-derived as `morris_lecar_driven_oscillation_doi` (`independent_rk4_reference`,
  bit-exact against the runner) and the `MorrisLecarNeuron` descriptor now carries the
  schema's RK4 integration. Schema-gap counts are unchanged (a re-enrolment, not a new
  schema). The schema-DSL runner is Python-only; the hand `MorrisLecarNeuron` and its
  Rust/Julia/Go/Mojo mirrors were already RK4 / no-reset, so no polyglot counterpart or
  benchmark artefact changed.
- Added a path-filtered `Compiler HDL E2E` workflow for pull requests touching
  `src/sc_neurocore/compiler/`, `src/sc_neurocore/hdl_gen/`, or `tests/e2e/`.
  A focused contract test locks the trigger paths and narrow e2e selector. No
  runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Deterministic reference-trace corpus
- Completed the deterministic bundled-schema reference-trace corpus for the reference models:
  all 17 deterministic schema-DSL models now have committed package data
  entries validated through `sc_neurocore.neurons.reference_traces`, while
  stochastic schemas and external simulator traces remain explicitly outside
  this deterministic corpus. No runtime package code, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Systematic audit rerun
- Added a systematic-audit rerun contract that locks the concrete 2026-07-04
  audit findings to repeatable gitignore, internal-TODO, SPDX-header, and SNN
  memory-discipline checks. No runtime package code, polyglot mirror, benchmark
  dispatch, or benchmark artefact changed.

### Strict typing and docstring policy
- Added a strict typing and NumPy docstring policy contract test that locks the
  2026-06-17 broadcast wiring across `pyproject.toml`, CI, preflight, the
  scoped docstring policy, and public maintenance docs. No runtime package code,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Rust/Python neuron binding boundary
- Documented the durable Python-only boundary for the five registry names
  without same-name PyO3 neuron constructors and locked each boundary to source
  evidence in the Rust/Python neuron parity map. No neuron runtime, PyO3
  implementation, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Optional-extra CI matrix
- Added optional-extra CI matrix lanes for annealing, ONNX protobuf export, and
  real MPI. The lanes install their focused extras, run the import-skipped
  production test selectors, and are locked by workflow/docs contract tests. No
  runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Audit cadence
- Added a monthly/manual `Audit Cadence` workflow that runs pytest
  collect-only, validates tracked test inventory with
  `tools/test_inventory_audit.py`, and uploads collection/audit artefacts. A
  focused contract test keeps the workflow, MkDocs navigation, and public
  development guide in sync. No runtime package code, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Training device fallback
- Made `sc_neurocore.training.auto_device()` skip CUDA devices whose compute
  capability is not supported by the installed PyTorch build, avoiding noisy
  local GTX 1060 / `sm_61` warnings while preserving fallback to MPS or CPU.
  Public training and GPU install docs now state that explicit `device="cuda"`
  requires a matching PyTorch CUDA architecture. No training algorithm,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Optics optional-extra CI coverage
- Added a dedicated CI optics-extra lane that installs `.[dev,optics]` and runs
  `tests/test_optics -q -rs`, so the `gdsfactory`-gated GDSII round-trip tests
  are exercised outside the default CPU jobs. Public optional-dependency docs
  now name that CI boundary. No optics runtime algorithm, package promotion
  boundary, polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Surrogate compiler warning hygiene
- Migrated the surrogate custom-op compiler regression to
  `torch.compile(..., backend="eager")` and added a guard against deprecated
  TorchScript `script_method` usage in the touched training lane. Public
  surrogate docs now state that this is graph-capture evidence, not a throughput
  benchmark. No training algorithm, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Annealing optional extra
- Added an `annealing` optional extra for `dwave-neal` and `dimod`, updated the
  optional dependency matrix and install-profile docs, and locked the metadata
  contract for the quantum-annealing `neal` parity selector. No base dependency,
  annealing runtime algorithm, benchmark dispatch, or benchmark artefact changed.

### Go autonomous-learning parity
- Fixed the autonomous-learning Go CGO setup contract: the parity test now runs
  from the Go module root with the `github.com/anulum/sc-neurocore/accel`
  import path, the Go bridge passes the Rust C-FFI timestep argument while
  preserving existing convenience calls, and public docs record the local
  `LD_LIBRARY_PATH` setup. No learning-rule dynamics, benchmark dispatch, or
  benchmark artefact changed.

### Rust/Python neuron binding coverage
- Documented the Rust/Python neuron binding coverage boundary and turned
  `tests/test_rust_python_neuron_parity.py` into a registry-level coverage map
  for 159 public Python registry names, including same-name Rust constructors,
  Rust-prefixed/core-only constructors, and current Python-only entries. No
  neuron runtime, PyO3 implementation, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Vivado CI gates
- Added a public Vivado CI gate guide for the `MIF_VIVADO_CI=1` ZU3EG
  synthesis-flow tests. A focused contract test now discovers the live
  Vivado-gated pytest files and keeps the guide plus MkDocs navigation in sync.
  No HDL logic, runtime package code, dependency pins, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Optional dependency matrix
- Added a public optional-dependency matrix for the audited `gdsfactory`,
  `neal`/`dimod`, ONNX, Lava, snnTorch, SpikingJelly, CuPy, and MPI surfaces.
  A focused contract test now checks the matrix against `pyproject.toml`, the
  relevant skip-gated test paths, install-profile docs, and MkDocs navigation.
  No runtime package code, dependency pins, polyglot mirror, benchmark dispatch,
  or benchmark artefact changed.

### Performance-gate CI
- Added a scheduled/manual `Performance Benchmarks` workflow lane for the
  `SC_NEUROCORE_PERF=1` pytest selector and documented the perf-gate contract.
  A focused contract test now discovers perf-gated files from the live test tree
  and keeps the workflow selector plus public guide in sync. No runtime package
  code, polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Schema-gap reporting
- Added `tools/schema_gap_report.py` and focused tests for schema-DSL
  coverage planning. The report scans live model/schema files without importing
  optional backends, distinguishes the net schema gap from exact source-module
  alias coverage, and ranks missing-schema rows by source-evidence enrolment
  priority. No neuron runtime, HDL logic, polyglot mirror, benchmark dispatch,
  or benchmark artefact changed.
### Fixed
- Made the DPI developer build path reproducible from a clean checkout. The
  source bridge now discovers the extension installed by `maturin develop`
  instead of shadowing it, and the Go backend hint builds the package rather
  than the single source file so cgo preserves the committed header identity.
- Made Tier-3 golden-trace verification portable across NumPy's AVX-512 and
  non-AVX-512 transcendental kernels. Chialvo and Hodgkin-Huxley now declare
  the two measured, byte-exact SHA-256 variants; the gate accepts only those
  hashes and separately proves their traces remain inside the existing
  `1e-9` numeric-parity bound. This covers the two digests observed across the
  Python 3.10-3.14 CI matrix without weakening deterministic models to an
  unconstrained tolerance check.
- Removed the stale Rust safety-module registry entry left after the generated
  CLI stub was deleted. Both Rust registries now resolve every declared module,
  and a focused source-registry contract prevents future file/declaration drift.

### Q4.12 co-simulation range classification
- Restored a green `main` after inherited tooling drift left three required checks red, with no
  behavioural change. The `pre-commit` and `lint` jobs reformatted 21 model/test/tool files to the
  pinned `ruff` 0.15.16 style — adjacent-string joins and single-line calls a staler formatter had
  wrapped — and trimmed a trailing newline the SymbiYosys catalogue commit left in
  `hdl/formal/catalogue/INVENTORY.md`. The `test` job's `tools/readiness_evidence_index.py` imports
  `tomli_w` (the TOML *writer*, which has no standard-library equivalent) but the dependency was
  never declared, so its whole test module errored at collection with `ModuleNotFoundError`;
  `tomli-w>=1.0` is now in the `dev` extra. The `v3-engine` `rust-lint` job tracks the latest stable
  Rust (1.97.0), whose new default `clippy` lints rejected two pre-existing engine sites — a
  redundant `seed: _` beside a `..` rest-pattern (`unneeded_wildcard_pattern`) and a
  `[b'A', b'C', b'G', b'T']` array literal (`byte_char_slices`); both are cleared (the latter as
  `*b"ACGT"`).
- Pinned the Connor-Stevens polyglot kernels to the Python golden spike count. The Go
  (`accel/go/services/connor_stevens.go`), Rust (`accel/rust/safety/connor_stevens.rs`) and Julia
  (`accel/julia/neurons/connor_stevens.jl`) kernels already carried the real six-state macro-step
  RK4 dynamics (100 sub-steps) but were only self-consistency / smoke tested against their own
  arithmetic; each now asserts the Python golden counts — silent at zero drive, two action
  potentials at `I=10` over 100 macro steps, nine at `I=20`. Connor-Stevens gating is `exp`-based,
  so the trace is not bit-exact across C libraries; the spike count is the stated parity observable,
  and all three languages reproduce it. Added a native Julia parity test
  (`accel/julia/connor_stevens_parity_test.jl`), a Go golden-parity test plus an honest
  `BenchmarkConnorStevensStep` (262.7 µs/macro-step), and the idiomatic `Default` impl on the Rust
  side. The Mojo kernel remains an honest parity note pending the Mojo neuron-kernel lane's
  promotion to a build target. Connor-Stevens's Python model dispatches only to the Rust engine, so
  the Go/Julia kernels are language-native; no Python runtime path, FFI dispatch, or committed
  benchmark artefact changed.
- Pinned the Morris-Lecar polyglot kernels to the Python golden spike count. The Go
  (`accel/go/services/morris_lecar.go`), Rust (`accel/rust/safety/morris_lecar.rs`) and Julia
  (`accel/julia/neurons/morris_lecar.jl`) kernels already carried the real RK4 dynamics but were
  only self-consistency / smoke tested against their own arithmetic; each now asserts the Python
  golden counts — silent at zero drive, three action potentials at `I=50` over 2000 steps, five at
  `I=100`. Morris-Lecar gating is `tanh`/`cosh`, so the trace is not bit-exact across C libraries;
  the spike count is the stated parity observable, and all four languages reproduce it. Added a
  native Julia parity test (`accel/julia/morris_lecar_parity_test.jl`) and an executable
  `simulate`/`main` parity harness to the Mojo kernel (`accel/mojo/kernels/morris_lecar.mojo`, so
  `mojo run` prints `PARITY OK`), and removed a dead `k4_v` assignment in that kernel's `next_w`
  path. On the Rust side, dropped a vestigial `#![allow(unused_variables, dead_code,
  non_snake_case)]` (the code is real, not a masked stub) and added the idiomatic `Default` impl.
  Morris-Lecar's Python model dispatches only to the Rust engine, so the Go/Julia/Mojo kernels are
  language-native; no Python runtime path, FFI dispatch, or committed benchmark artefact changed.
- Completed the FitzHugh-Nagumo Go accel-services surface so the `accel/go/services`
  test suite compiles and runs again. `services_test.go` called
  `SimulateFitzHughNagumoNeuron`, which no source file defined (the FitzHugh-Nagumo Go
  kernel existed only as a `package main` cgo shared library under
  `accel/go/neurons/fitzhugh_nagumo/`), so the whole services package test build failed. Added
  `services/fitzhugh_nagumo.go` — a real RK4 `SimulateFitzHughNagumoNeuron` in parity with
  `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron` (the cube written
  `v*v*v`, exact arithmetic, fail-closed on non-finite input/state/candidate) — with a
  golden-parity test (one action potential at `I=10` over 100 steps and a five-spike partial
  train at `I=0.5` over 2000 steps, final `v` bit-identical to the NumPy reference) and an
  honest per-step benchmark. Also strengthened the `accel/rust/safety` FitzHugh-Nagumo test
  from a `spike is 0 or 1` smoke check to the same Python-golden spike count, dropped a
  vestigial `#![allow(dead_code)]`, and added the idiomatic `Default` impl. The services
  surface is Go-native, not FFI-dispatched, so no Python runtime path, FFI dispatch,
  cross-language `simulate` benchmark, or committed benchmark artefact changed.
- Replaced the Q4.12 LIF zero-current co-simulation xfail with an explicit
  range-classification regression that checks the Q-format diagnostics and the
  public `precision lif` CLI. The co-simulation and precision docs now state
  that Q4.12 is a normalized-dynamics mode, not a zero-current millivolt-scale
  LIF parity mode. No HDL logic, runtime neuron dynamics, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Studio dependency profile
- Added `httpx2` to the Studio and full install profiles so Starlette
  `TestClient` uses its non-deprecated transport in Studio tests. The install
  profile guide and metadata contract tests now cover the dependency. No Studio
  runtime endpoint, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Pytest warning hygiene
- Disabled the ambient `pytest_nengo` plugin in the repository pytest config so
  unrelated test collection no longer imports Nengo and emits the third-party
  NumPy 2.x `numpy.core` deprecation warning before SC-NeuroCore tests run.
  Added a policy test for the pytest configuration and active plugin registry.
  No runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Co-simulation toolchain gate
- Added a typed Icarus Verilog dependency checker and wired the CI test matrix
  to verify `iverilog -V` and `vvp -V` against the documented 12.x
  co-simulation floor before package tests run. The FPGA toolchain guide now
  records the same minimum and CI command. No HDL logic, runtime package code,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Public prose hygiene
- Removed self-applied public superlatives from the CMOS profile notes, GPU
  backend guide, and archived state report while preserving the underlying
  measured values and benchmark comparison tables. No platform registry
  behavior, GPU backend code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Copyright spelling hygiene
- Standardised copyright spelling across `tools/` and
  `.github/workflows/security-scanners.yml`: legacy `(c)` forms, ASCII
  date ranges, and `Sotek` spellings now use the canonical `©`, en dash year
  ranges, and `Šotek`. Touched tool headers now keep full seven-line GOTM
  descriptions, and generator string outputs were updated so emitted Vivado and
  Vertex config artefacts keep the same spelling. Strict mypy now passes on the
  touched tool scope. No runtime package code, HDL logic, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Header hygiene
- Split joined SPDX/commercial-license headers across manifests, workflow
  YAML, HDL sources, Vivado-import HDL copies, and the Vmin-LIF LUT generator.
  The touched TOML/Cargo manifests now retain full seven-line GOTM headers and
  canonical author spelling. No runtime package code, HDL logic, polyglot
  mirror, benchmark dispatch, or benchmark artefact changed.

### Configuration validator hygiene
- Hardened `scripts/validate_configs.py` into a typed repository configuration
  validator, removed the unused top-level `tomli` import, aligned the required
  user guide path with `docs/guides/USER_MANUAL.md`, and added focused CLI
  validation tests. The generated capability surfaces were refreshed for the
  new test file. No runtime package, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Expression differentiator typing
- Restored strict-mypy compliance for the neuron expression differentiator by
  isolating SymPy's partially typed constructors, differentiation, and printer
  APIs behind typed boundary helpers. The in-grammar derivative contract,
  finite-difference behavior, generated API surface, polyglot mirrors, and
  benchmark-dispatched paths are unchanged.

### Studio sandbox hardening
- Hardened Studio job sandbox path confinement so generated job directories,
  seed/control seed inputs, live artifact reads, artifact downloads, control
  commands, and purges validate canonical paths before filesystem access. Added
  focused regression coverage for malformed job IDs and symlinked reserved seed
  directories. No Studio job API, worker payload schema, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Sensor package exports
- Exported the ADC-to-spike kernel surface from `sc_neurocore.sensors`,
  including `ADCSpikeWindowConfig`, `ADCSpikeWindowResult`, backend selection,
  the bit-true Python floor, and `quantise_adc`. Existing submodule imports
  remain compatible. Sensor API docs, hardware docs, and module-specific tests
  were updated. No ADC arithmetic, Rust/Julia/Go/Mojo backend, dispatch order,
  benchmark output, or benchmark artefact changed.
### Added
- Added a discrete-map integration mode (`method="map"`) to `EquationNeuron` and
  the schema-DSL runner: it iterates `state_{n+1} = f(state_n)` directly instead
  of integrating an ODE. The Verilog compiler gained `IfExp` lowering (piecewise
  `?:` selects) and a discrete-map datapath, so a piecewise map lowers to
  Icarus-Verilog-valid RTL. The `rulkov_map` schema now uses the DOI-verified
  Rulkov (2002) piecewise fast/slow map with `method="map"`, replacing the smooth
  variant that was integrated with explicit Euler and diverged; its reference
  trace is a bounded driven-spiking trace with exact independent map-iteration
  parity, so all 17 deterministic reference traces now hold independent parity.
  The schema-DSL runner has no polyglot mirror; no benchmark dispatch or
  benchmark artefact changed.
- The quantum package namespace now exports the SC→quantum compiler surface,
  including `QuantumGate`, `SCQuantumCircuit`, probability/rotation helpers, and
  `compile_sc_multiply` / `compile_sc_layer`. Existing submodule imports remain
  compatible; quantum API docs, tutorial examples, and module-specific tests
  were updated. No quantum algorithm, polyglot safety mirror, benchmark
  dispatch, or benchmark artefact changed.
- The training package namespace now exports the NumPy equilibrium-propagation
  research surface as `sc_neurocore.training.EPNetwork`, making the existing
  two-phase settle-and-nudge path selectable without importing the submodule.
  Training docs and module-specific tests were updated. No Torch training path,
  polyglot kernel, benchmark dispatch, or benchmark artefact changed.
- The spintronic package namespace now exports the documented magnetic-domain
  mapper surface, including `SpintronicMapper`, device/material models, MuMax3
  helpers, racetrack/skyrmion utilities, aging/radiation/defect models, and the
  Verilog generator. The quick-start docs now use actual exported symbols;
  existing submodule imports remain compatible. Generated capability surfaces
  were refreshed for the new package API test file. No mapper kernel,
  Julia/Rust/Mojo mirror, benchmark dispatch, or benchmark artefact changed.
- The memristor package namespace now exports the documented crossbar mapper
  surface, including `MemristorMapper`, conductance/crossbar models,
  compensation helpers, Monte Carlo reports, and the SystemVerilog emitter.
  Existing submodule imports remain compatible; docs and package-facade tests
  were updated. No mapper kernel, Julia/Rust/Mojo mirror, benchmark dispatch,
  or benchmark artefact changed.
- The JAX dense layer is now exported through the lazy public facades as
  `sc_neurocore.JaxSCDenseLayer` and `sc_neurocore.layers.JaxSCDenseLayer`,
  making the existing optional JAX backend selectable without importing the
  submodule path. The package wiring stays lazy so base imports do not construct
  accelerator state. Public API tests, layer docs, and generated capability
  surfaces were updated; no JAX kernel, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.
- The analysis package namespace now exports the maintained Phi* estimator
  functions `phi_star` and `phi_from_spike_trains`, making the existing
  NumPy/Rust/Julia/Go/Mojo-backed integrated-information surface selectable
  without importing the submodule path. Tests and curated analysis docs were
  updated, and generated API freshness was verified; no backend kernel,
  benchmark dispatch, or benchmark artefact changed.
- Quantum-cognition coverage contracts now exercise GOTM brain local-LLM import
  fallback and spike-index accumulation, lifting the focused `dashboard.py`,
  `gotm_brain.py`, and `radical_pair.py` selector to 100% exact-file coverage.
  No runtime path, polyglot mirror, benchmark dispatch, generated API surface,
  or benchmark artefact changed.
- The generated public API reference now excludes single-underscore classes,
  functions, and methods while retaining public and dunder surfaces. The
  generator contract, committed `docs/API_REFERENCE.md`, and documentation
  workflow were updated together; no runtime path, polyglot mirror, benchmark
  dispatch, or benchmark artefact changed.
- Quantum-cognition CLI SNN stimulus records now emit the canonical numeric
  fleet-memory `timestamp` field while preserving `content`, `project`, `actor`,
  `kind`, and `source_ref`; the focused CLI contract rejects legacy `text` and
  `source` aliases. No runtime model dynamics, polyglot kernel,
  benchmark-dispatched path, or benchmark artefact changed.
- Added a repeatable SNN memory-discipline audit/repair tool for SC-NeuroCore
  stimulus producers and records. The tool validates canonical fleet-memory
  keys, controlled actors, timestamps, entities, kinds, and source provenance,
  and can normalize legacy local records without changing runtime model
  dynamics, polyglot kernels, benchmark dispatch, or benchmark artefacts.
- Added a schema-driven neuron reference-trace validation harness with immutable
  corpus contracts, fail-closed JSON payload parsing, package-data loading,
  `UniversalNeuron` execution, and feature-level validation reports. The seed
  corpus covers analytic closed-form `lif`, `lapicque`, and `quadratic_if`
  traces and is covered by strict mypy checks plus 100% exact-file focused
  coverage; no polyglot kernel or benchmark-dispatched runtime path changed.
- Compiler coverage-contract tests now lock the MLIR bundle fail-closed CIRCT
  paths, pipeline tool-resolution failures, folded Verilog datapath guard paths,
  and split expression-emitter LUT/error branches that keep the compiler core at
  100% exact-file coverage under its focused selector.
- The `sc.kuramoto_step` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_kuramoto_step.v` fixed-point phase core (signed Q8.16, 64-entry sine LUT)
  that computes one Euler step
  `theta_n += dt * (omega_n + sum_m K_nm sin(theta_m - theta_n))` over an explicit
  N×N coupling matrix. The core is co-simulated against a bit-exact fixed-point oracle
  with Icarus Verilog and checked against the ideal float step within the LUT
  resolution, and the MLIR emitter now instantiates the same core instead of a
  passthrough. The Python IR builder gains `constant_f64_vec` and `kuramoto_step` so
  the graph can be constructed, emitted, and simulated end to end.
- The `sc.graph_forward` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_graph_forward.v` fixed-point aggregation core (signed Q8.16) that computes the
  degree-normalised neighbourhood aggregation `agg[i][f] = (Σ_j A[i][j]·X[j][f]) / deg[i]`
  — the graph-structural half of the reference `StochasticGraphLayer` rate-mode forward
  pass — over an explicit adjacency matrix, deferring rounding to a single signed division.
  The core is co-simulated against a bit-exact fixed-point oracle with Icarus Verilog and
  checked against the ideal float aggregation within fixed-point resolution, and the MLIR
  emitter now instantiates the same core instead of a passthrough. The Python IR builder
  gains `graph_forward`.
- The `sc.softmax_attention` IR operation now lowers to synthesisable RTL, replacing the
  previous hard error. The SystemVerilog emitter instantiates a new
  `hdl/sc_softmax_attention.v` fixed-point single-head scaled-dot-product attention core
  (signed Q8.16) that computes `softmax(Q·Kᵀ / sqrt(dim_k)) · V` — the reference
  `StochasticAttention::forward_softmax` — with a numerically stable row-max subtraction,
  a 256-entry exp lookup over the symmetric [-16, 16) grid at 0.125 spacing (mirroring the
  `expr_lut_tables` / `c_fixed_emitter` transcendental machinery), and a single integer
  division per softmax weight. Query/key/value shapes are inferred from the constant
  operand lengths and `dim_k`, and the `1/sqrt(dim_k)` scaling plus exp-LUT geometry are
  baked as instance parameters. The core is co-simulated against a bit-exact fixed-point
  oracle with Icarus Verilog and checked against the ideal float softmax attention within
  the exp-LUT resolution, and the MLIR emitter now instantiates the same core instead of a
  passthrough. The Python IR builder gains `softmax_attention`. With this, all three IR ops
  that previously had no synthesisable RTL (KuramotoStep, GraphForward, SoftmaxAttention)
  now lower to co-simulated fixed-point cores.
- `generate_ip_xact` is now re-exported from the `sc_neurocore.hdl_gen` package namespace
  alongside the other `generate_*` HDL emitters, so the IP-XACT (IEEE 1685) component-XML
  generator can be imported without reaching into the submodule. The generator gained a
  dedicated structural test suite (`tests/test_ip_xact.py`) that parses the emitted
  component tree — the AXI-Lite slave interface, the optional parameter block, and the
  port-vector geometry — and it is no longer excluded from coverage measurement (now 100 %).
- The bipolar stochastic-computing primitives (`core/bipolar.py`) are no longer excluded
  from coverage measurement: the guard clauses (non-finite and out-of-range values,
  non-binary bitstreams, wrong-dimensional MAC operands), the default-RNG path of
  `bipolar_encode`, and the SC-layer optional-bias and `tanh` branches gained tests,
  taking the module from 84 % to 100 %.
- The closed-loop sleep optimiser (`sleep/sleep_optimizer.py`) is no longer excluded from
  coverage measurement: the inactive-session guards (`add_sample` / `add_samples` /
  `check_and_adapt` before a session starts), the None-detection fallback to WAKE, and the
  re-induction branch (two consecutive unwanted awakenings arm a gentle N1 re-induction)
  gained tests, taking the module from 91 % to 100 %.
- The post-session sleep report generator (`sleep/report_generator.py`) is no longer excluded
  from coverage measurement: the grade bands (A/B/D/F), the sleep-onset-latency scoring tiers,
  the mid-session wakeup counter, and the N3/REM/latency/wakeup recommendation branches gained
  tests over a controlled tick history, taking the module from 84 % to 100 %.
- The closed-loop adaptive audio engines (`audio/adaptive_engine.py` and `audio/evs_engine.py`)
  are no longer excluded from coverage measurement. Both were already exercised to 100 %
  statement coverage by the existing behaviour and contract suites (`test_adaptive_audio`,
  `test_audio_evs_contracts`, and the audio package/mapping/profile contracts); the omit
  entries were stale, so removing them simply lets that measured coverage count.
- The NeuroML 2 importer (`adapters/neuroml.py`) is no longer excluded from coverage
  measurement. It is a pure-standard-library (`xml.etree`) importer with no optional
  dependency, and the unit/current parsers' missing-value and dimensionless fall-through
  paths, the `iafTauRefCell` refractory branch, and the `create_neuron` AdEx and
  unknown-cell-type branches gained tests, taking the module from 91 % to 100 %.
- The CCW/VIBRANA bridge (`interfaces/ccw_bridge.py`) gained a dedicated behavioural test
  suite (`tests/test_ccw_bridge.py`, 45 tests) and is no longer excluded from coverage
  measurement. Despite the "CCW system bridge" label it is a self-contained,
  pure-standard-library-and-numpy data transformation (SCPN metrics → binaural-audio
  parameters, L7 glyph vectors → VIBRANA visualisation states, plus metadata and session
  configs) with no live-system dependency, so every mapping, mode-selection branch, metric
  smoothing path, glyph-length guard, and the optional JSON file-export sink are now
  covered at 100 %.
- The PYNQ/hardware-in-the-loop drivers (`drivers/physical_twin.py`,
  `drivers/sc_neurocore_driver.py`, `drivers/verify_hardware_link.py`) are no longer
  excluded from coverage measurement — the whole `*/drivers/*` omit glob is removed. They
  are pure-Python and already tested with mocked PYNQ overlays and sockets (no real FPGA);
  added tests close the remaining branches — the twin's constructor validation guards, its
  TCP connection-failure / empty-reply / non-JSON-reply error paths, the driver's
  missing-bitstream failure, and the hardware-link diagnostic's FPGA success/unexpected-error
  branches plus the present-but-unreachable Evo 2 and Opentrons probe outcomes — taking all
  three modules to 100 %.

### Quantum package exports
- Exported the SC→quantum compiler surface from `sc_neurocore.quantum`,
  including `QuantumGate`, `SCQuantumCircuit`, probability/rotation helpers, and
  `compile_sc_multiply` / `compile_sc_layer`. Existing submodule imports remain
  compatible. Quantum API docs, tutorial examples, and module-specific tests
  were updated. No quantum algorithm, polyglot safety mirror, benchmark
  dispatch, or benchmark artefact changed.
### Fixed
- Capped the `nir` dependency to `<1.0.8`. nir 1.0.8 makes `Flatten.input_type` a required
  constructor argument, which broke the NIR bridge's `Flatten` construction and turned the
  test suite red once the release reached CI (the pinned 1.0.7 in `requirements/hub.txt` was
  overridden when the package was built and installed with extras). The extras now match the
  tested 1.0.7 line; the NIR canary continues to track the latest nir separately.
- The SC IR text format was not idempotent under `parse . print` (found by the
  `roundtrip_ir` fuzz target once the fuzz workflow was repaired). Two float-handling
  asymmetries are corrected: (1) the printer formatted whole-number `f64` constants without a
  decimal point (`5.0` → `"5"`, `[5.0, 6.0]` → `"[5, 6]"`), so they re-parsed to integer
  variants (`U64`/`I64Vec`) — the printer now appends `.0` so such values re-read as floats;
  and (2) the parser accepted non-finite float literals (`NaN`, `inf`), and a `NaN` constant
  can never round-trip because `NaN != NaN` — the parser now rejects non-finite literals at
  every float site (constants, vector elements, and the `scale`/`offset`/`kuramoto_step`
  parameters). Regression tests cover whole-number scalar/vector round-tripping and non-finite
  rejection.
- The NeuroML 2 importer produced an unusable Adaptive-Exponential (AdEx) cell:
  `_import_adex_cell` emitted the NeuroML attribute names (`C`, `g_L`, `E_L`, `V_T`,
  `delta_T`, `V_reset`, `V_thresh`) rather than the `AdExNeuron` constructor names, so
  `create_neuron` raised `TypeError: unexpected keyword argument 'C'` for every imported
  AdEx cell. The importer now maps the attributes onto the model's own parameters in its
  native unit system — leak reversal to `v_rest`, exponential threshold `V_T` to `v_rh`,
  `tau = C / g_L` (pF/nS = ms) with the capacitance kept as `c_m` — and, critically,
  parses the spike-triggered adaptation `b` as a *current* in pA (`0.0805 nA = 80.5 pA`)
  rather than in nA, since `b`, the adaptation state `w`, and the injected current share
  the pA unit. The correction is validated against the Brette & Gerstner (2005) hallmark
  of spike-frequency adaptation (a lengthening inter-spike interval under sustained drive).
- The SystemVerilog emitter wrote malformed signed literals (`16'sd-51`) for negative
  fixed-point constants; the sign is now placed outside the sized base (`-16'sd51`),
  so graphs with negative vector or scalar constants emit valid Verilog.

### Training package exports
- Exported the NumPy equilibrium-propagation research surface as
  `sc_neurocore.training.EPNetwork`, so the existing two-phase
  settle-and-nudge path is selectable from the package facade. Training docs and
  module-specific tests were updated. No Torch training path, polyglot kernel,
  benchmark dispatch, or benchmark artefact changed.

### Spintronic package exports
- Exported the documented magnetic-domain mapper surface from
  `sc_neurocore.spintronic`, including `SpintronicMapper`,
  device/material models, MuMax3 helpers, racetrack/skyrmion utilities,
  aging/radiation/defect models, and the Verilog generator. The quick-start docs
  now use actual exported symbols, and existing submodule imports remain
  compatible. Generated capability surfaces were refreshed for the new package
  API test file. No mapper kernel, Julia/Rust/Mojo mirror, benchmark dispatch,
  or benchmark artefact changed.

### Memristor package exports
- Exported the documented memristor crossbar mapper surface from
  `sc_neurocore.memristor`, including `MemristorMapper`,
  conductance/crossbar models, compensation helpers, Monte Carlo reports, and
  the SystemVerilog emitter. Existing submodule imports remain compatible. No
  mapper kernel, Julia/Rust/Mojo mirror, benchmark dispatch, or benchmark
  artefact changed.

### JAX dense layer exports
- Exported `JaxSCDenseLayer` through the lazy package facades as
  `sc_neurocore.JaxSCDenseLayer` and
  `sc_neurocore.layers.JaxSCDenseLayer`. The optional backend remains
  construction-time gated by the `jax` extra, and package import stays
  lightweight. Public API tests, layer docs, and generated capability surfaces
  were updated. No JAX kernel, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Analysis package exports
- Exported `phi_star` and `phi_from_spike_trains` from
  `sc_neurocore.analysis`, so the maintained Phi* integrated-information
  estimator is selectable from the public analysis namespace. The curated
  analysis guide was updated and generated API freshness was verified. No
  backend kernel, benchmark dispatch, or benchmark artefact changed.

### Quantum cognition coverage
- Added focused GOTM brain contracts for local-LLM import fallback and
  spike-index accumulation. The focused selector for `dashboard.py`,
  `gotm_brain.py`, and `radical_pair.py` now reports 100% exact-file coverage.
  No runtime path, polyglot mirror, benchmark dispatch, generated API surface,
  or benchmark artefact changed.

### Public API reference
- Changed the generated API reference to publish public classes, public module
  functions, public methods, and dunder methods only. Single-underscore helper
  classes, functions, and methods are now omitted from `docs/API_REFERENCE.md`;
  the generator contract and documentation workflow were updated together. No
  runtime path, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Quantum cognition memory schema
- Changed quantum-cognition CLI SNN stimulus records to emit the canonical
  numeric fleet-memory `timestamp` field while keeping `content`, `project`,
  `actor`, `kind`, and `source_ref` stable. The CLI memory-discipline contract
  now rejects legacy `text` and `source` aliases and locks the numeric timestamp
  shape. No runtime model dynamics, polyglot kernel, benchmark-dispatched path,
  or benchmark artefact changed.
- Added `tools/snn_memory_discipline_audit.py` to validate SC-NeuroCore SNN
  stimulus producers and existing records against the fleet memory-write
  schema, including canonical keys, controlled actors, timestamps, entities,
  kinds, and source provenance. Repair mode normalizes legacy local records
  without deleting files or changing runtime model dynamics, polyglot kernels,
  benchmark dispatch, or benchmark artefacts.

### Perfect-integrator co-simulation schema
- Added the DOI-backed `perfect_integrator` UniversalNeuron schema in TOML and
  JSON form, packaged bundled schema assets into the wheel, and enrolled the
  model in deterministic Q8.8 Python-to-Verilog co-simulation. New tests compare
  the schema against the hand-authored `PerfectIntegratorNeuron` and assert
  emitted-RTL spike-count parity. No polyglot mirror implementation or
  benchmark-dispatched path changed.

### Compiler proof-transform wiring
- Added an explicit opt-in `proof_transforms` compiler facade for whitebox state
  taps and operator abstraction. The package root now exposes registry lookup and
  dispatch helpers, docs classify the transforms as proof-only rather than
  production compiler flags, and compatibility coverage keeps `quantize_core`
  anchored to the canonical quantizer surface. No polyglot kernel,
  benchmark-dispatched path, or benchmark artifact changed.

### Public docstring quick wins
- Closed the AB-DOC-1 quick-win set by documenting the public package entry
  points for layers, synapses, utils, datasets, formal verification, and SCPN
  layers; the CLI entrypoint; optional Rust fallback surfaces for DNA,
  quantum-annealing, photonic, and Studio helpers; Studio preset/template
  helpers; and Studio request schemas. The cleaned files are now locked in
  `docs/docstring_policy.toml`, raising the scoped public-docstring gate from
  258 to 273 files. Generated API docs were refreshed. No runtime path,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### NotImplemented guard audit
- Added a tracked source audit for executable Python `NotImplementedError`
  sites. The audit allows only explicit fail-fast guards for unsupported MPI,
  forced-Rust, Torch bridge, NIR node-map, optics GDSII, hardware-DMA, and
  abstract-neuron boundaries, preventing hidden selectable stubs from entering
  tracked Python sources. No runtime path, polyglot mirror, or benchmark
  artefact changed.

### Rust interneuron performance-test gating
- Marked the Rust interneuron wall-clock smoke tests for PV, SST, VIP,
  Chandelier, cerebellar basket, and Martinotti neurons as opt-in ignored tests
  so default `cargo test` no longer fails under host CPU contention. Timing
  evidence remains owned by the Criterion benchmark surfaces; no neuron
  dynamics, Python surface, polyglot mirror, or benchmark artefact changed.

### Block-floating scalar guard
- Added an explicit `BlockFloatingScalarEncodingError` and scalar-only preset
  guard for mixed-precision configs. Block-floating precision remains available
  for metadata-aware dense, adaptive, and manifest paths, while scalar parameter
  encoders can now call `from_preset(..., scalar_only=True)` or
  `encode_scalar_value(...)` to fail closed before detached exponent metadata is
  lost. No polyglot kernel, HDL datapath, or benchmark-dispatched path changed.

### Reference trace validation harness
- Added a schema-driven neuron reference-trace validation harness with immutable
  corpus contracts, fail-closed JSON payload parsing, package-data loading,
  `UniversalNeuron` execution, and feature-level validation reports. The seed
  corpus covers analytic closed-form `lif`, `lapicque`, `quadratic_if`,
  `theta`, and spike-bearing `perfect_integrator` traces and is covered by
  strict mypy checks plus 100% exact-file focused coverage; no polyglot kernel or
  benchmark-dispatched runtime path changed.

### Network Rust backend contract hardening
- Reverified the DEEP_AUDIT network/Rust findings against the current tree:
  spike events use `u64` packing with 32-bit neuron/timestep lanes, population
  dispatch uses `Population.model_name` rather than labels, Rust final voltages
  sync back into populations, and the effective workspace release profile now
  owns `panic = "abort"`. Added regression tests for those contracts and made
  forced Rust fail fast for `StateMonitor`, `RateMonitor`, `spike_gating`, and
  `fim_lambda`, while `backend="auto"` falls back to Python for those
  Python-only semantics. No benchmark-dispatched path, polyglot mirror, or
  benchmark artefact changed.

### Release provenance and publish retry hardening
- Added manual tagged-release backfill support to the release workflow,
  retained the release security packet as a workflow artifact even when the
  sweep fails, and aligned release/security scanner cargo-fuzz installation on
  the validated `cargo-fuzz==0.13.2` pin. Made PyPI and crates.io publish
  retries idempotent for already-published versions while preserving manual
  dry-run validation. No runtime package, polyglot kernel, or benchmark path
  changed.

### Mojo helper contract honesty
- Removed the hidden `NotImplementedError` IPC stub from
  `accel.mojo.runner.MojoKernelRunner.popcount` and `lfsr_encode`, exported
  `MOJO_HELPER_BACKEND="python-fallback"` plus
  `MOJO_HELPER_IPC_AVAILABLE=False`, and added AST regression coverage so scalar
  helpers cannot silently present a fake Mojo path. Refreshed the Mojo docs to
  separate actual build/benchmark subprocess execution from Python helper
  fallbacks. No Mojo kernel, benchmark-dispatched path, Rust path, or polyglot
  mirror implementation changed.

### Backend selector coverage ratchet
- Extended the benchmark-driven backend selector tests to cover CPU probe
  fallback, missing benchmark directories, and malformed benchmark JSON files,
  bringing `accel.backend_selection` to 100% exact-file coverage. Promoted the
  selector into the scoped NumPy docstring policy. No dispatch order, benchmark
  artefact, polyglot runtime, or kernel implementation changed.

### Constants ledger audit hardening
- Added dedicated constants-ledger regression tests for the 44 public constants,
  scalar types, physical ranges, Q8.8 invariants, maintained-source import map,
  module-docstring honesty, and Izhikevich spike-threshold wording. Corrected
  the constants module docstring to state the current 16-module Python adoption
  boundary, promoted it into the scoped NumPy docstring policy, and refreshed
  the constants audit page. No constant values, Rust mirror values, or
  benchmark-dispatched paths changed.

## [3.16.0] - 2026-07-05

### Provenance and citation integrity
- Verified every neuron-descriptor DOI against its registry (Crossref, or DataCite
  for arXiv preprints) and corrected misstated and fabricated citations: a
  non-existent "Kilinc & Bhatt (2023)" (the model is the Nagumo-Sato/Aihara sigmoid
  map), a fabricated "Jahns et al. (2025)", a recurring phantom "Bhatt" co-author
  across several cerebellar models, an arXiv identifier that pointed at an unrelated
  paper, and several digit-transposed or wrong DOIs. Added `tools/provenance/verify_dois.py`
  and a committed DOI ledger checked offline by `tests/test_provenance_doi_integrity.py`,
  so a fabricated or mistyped descriptor DOI now fails CI. Descriptor citeable coverage
  rose from 119 to 131 models.

### Verification safety screen hardening
- Hardened `verification.safety.CodeSafetyVerifier` to reject AST-visible
  filesystem, process, network, relative-import, dynamic-import,
  dynamic-execution, and reflection escape routes including `open(...)`,
  `Path(...).write_text(...)`, `socket.socket()`, `__builtins__.eval(...)`,
  `__builtins__['eval'](...)`, and `getattr(__builtins__, 'eval')`. Promoted
  the verifier into the scoped NumPy docstring policy, strict-typed the focused
  tests, refreshed the verification guide and generated API reference, and kept
  `verification/safety.py` at 100% exact-file coverage. The generated
  capability manifest, snapshot, and README capability block were also refreshed
  after the public-claims selector exposed a stale tracked-test count. No
  polyglot mirror or benchmark-dispatched path changed.
### Added
- `compile_network_to_fpga` now fails closed on IR that would exhaust synthesis resources,
  before any RTL is emitted: `data_width` must lie in `[1, 64]` and `fraction` must satisfy
  `0 <= fraction < data_width` (so the signed Q-format keeps at least one integer or sign
  bit — otherwise it would silently emit broken RTL); the neuron count is
  capped — at 8192 for the direct/AER interconnects (which instantiate one module per
  neuron, so an unbounded count is a synthesis-time denial of service) and higher, 262144,
  for the folded interconnect (which shares one processing element and is bounded by its
  state-RAM depth). A network over the direct/AER cap is pointed at `interconnect="folded"`.
  The total synapse count is capped too (at 1048576), since every interconnect flattens all
  weight matrices into one shared ROM — a blow-up axis independent of the neuron count.
- `generate_host_driver` now validates its inputs before emitting any driver code: the
  `data_width`/`fraction` Q-format (as the FPGA compiler does), a non-negative memory-mapped
  `base_address`, and a positive, bounded bit width (`[1, 4096]`) for every parameter
  register — so a malformed AXI-Lite/Wishbone driver request fails closed instead of
  emitting a driver with degenerate or unbounded runtime masks.
- `cargo-fuzz` targets (`engine/fuzz`): `parse_ir` fuzzes `ir::parser::parse` (arbitrary
  input — exposed to Python as `ir_parse` — must only ever return `Err`, never panic);
  `roundtrip_ir` asserts the parser and `printer::print` are inverse (`parse(print(g)) == g`,
  and a printed graph always re-parses); `bitstream` differentially checks that
  `bitstream::pack` and `pack_fast` agree bit-for-bit on arbitrary bytes and that
  `unpack` round-trips; and `csr_matrix` builds a CSR matrix from `arbitrary` vectors and
  expands it, checking that a matrix `CsrMatrix::new` accepts cannot panic on use (it found
  a real out-of-bounds — see Fixed). The fuzz crate depends on the engine with
  `default-features = false`, so the build skips the bundled-Z3 compile (made optional in the
  same release) and the pyo3 module dead-strips out, keeping the instrumented build fast;
  `fuzz/Cargo.lock` is committed so NumPy/ndarray resolve to the engine's versions. A
  scheduled weekly `SC-NeuroCore Fuzz` workflow runs all four (not a per-PR gate).
- The folded FPGA interconnect now folds populations with **heterogeneous per-neuron
  parameters**. Each parameter that varies across a population's neurons is exposed on a
  processing-element input port (`compile_to_datapath(param_ports=...)`) and streamed from
  a per-neuron ``case(nidx)`` ROM — the parameter-space analogue of the state BRAM — so
  every neuron receives its own parameters, bit-for-bit the direct path's per-neuron
  ``#(.P_X(...))`` overrides (golden co-simulation parity for heterogeneous firing
  thresholds and membrane time constants). `FoldedResourceMetrics` gains `param_rom_bits`
  and the folded area estimator charges the parameter ROM. Populations with uniform
  parameters are unchanged (the PE bakes them). This removes the former restriction that a
  folded population be parameter-uniform.
- Formal equivalence toolkit generalised to a two-state neuron shape — the
  Izhikevich model, the first with two coupled state registers (membrane ``v`` and
  recovery ``u``). The quadratic ``(v-VR)*(v-VT)`` product drives ``v`` and a spike
  resets both (``v <- C``, ``u <- u + D``). Proving it equivalent unbounded needs
  the coordinated two-register state-matching invariant: a `StateTap` for ``v`` and
  ``u``. The point is empirically pinned by a companion proof that taps only ``v``
  and returns `UNKNOWN` (inconclusive — the modules are equivalent, the invariant is
  merely too weak) rather than `PASS`, establishing that every coupled state register
  must be tapped, not just the observable output. With the one product abstracted and
  both states tapped, a structurally distinct DUT and golden reference prove
  equivalent unbounded by k-induction (≈8 s at 16-bit under `z3`). No toolkit code
  change — the existing multi-tap `expose_state_taps` and `abstract_to_free_inputs`
  already carry the two-state case, confirming the flow spans linear (LIF), quadratic
  single-state (QIF) and quadratic two-state (Izhikevich) neurons.
- Formal equivalence toolkit generalised to a second neuron shape — the quadratic
  integrate-and-fire (QIF). Its state update carries a ``v*v`` *self*-multiply
  (the LIF only multiplied state by a free input) declared inline as
  ``wire v_sq = v * v;``. `operator_abstraction.abstract_to_free_inputs` now
  handles both that inline-initialiser form and the declaration-plus-``assign``
  form, and anchors the lifted name as the declared identifier so a use in another
  statement's right-hand side is never matched. With the self-multiply abstracted
  to a shared free input and the single membrane state tapped, a structurally
  distinct QIF DUT and golden reference prove equivalent unbounded by k-induction —
  demonstrating the whitebox-tap + multiplier-abstraction flow is not LIF-specific.
- Multiplier abstraction for full-width unbounded equivalence proofs
  (`sc_neurocore.compiler.operator_abstraction`). Whitebox taps make k-induction
  converge, but bit-blasting the fixed-point multiplier keeps it tractable only at
  a narrow (4-bit) datapath. `abstract_to_free_inputs` drops the signal carrying a
  product and exposes it as a free input port; when the device-under-test and the
  reference lift the same product to the same input name, the miter drives both
  instances from one shared free wire, so the products are equal by construction
  (congruence) and the solver never reasons about multiplication. The abstraction
  is a sound over-approximation for a `PASS` (equivalence for every product value
  implies it for the real product); a blackbox uninterpreted function would be the
  textbook route but yosys 0.33's `smtbmc` crashes on a blackbox submodule. With
  the multipliers abstracted and the state tapped, the LIF proves equivalent to
  its reference **unbounded at full width** — about 1 s at 16-bit, 4 s at 32-bit,
  and 22 s at 64-bit under `z3` k-induction, where before it was intractable past
  4-bit; the residual growth is the datapath adders and comparators, not the
  (now-abstracted) multiplier.
- Unbounded equivalence proofs via whitebox state taps
  (`sc_neurocore.compiler.whitebox_taps`). Naive k-induction on the equivalence
  miter is intractable — the induction step starts from unreachable states where
  the outputs agree but the hidden state differs, and the fixed-point multiplier
  compounds it. `expose_state_taps` instruments a module to surface its internal
  registers as observation output ports (an `output wire` plus one continuous
  `assign`, adding no register and rewriting no logic, so behaviour on the original
  ports is unchanged); a tap whose source is a constant lets two structurally
  different modules present the same tap interface. The miter then compares the
  taps like any other output, and that tap equality is the reachable-state
  invariant that makes `prove_equivalence(mode="prove")` converge — the LIF
  primitive is proven equivalent to its reference *unbounded*, not just to a
  bounded depth. (yosys 0.33 resolves neither hierarchical references nor
  SystemVerilog `bind`, so exposing the state is the route to reference it.) The
  remaining bound is SMT tractability, not soundness: the proof closes for a narrow
  4-bit datapath and slows as the multiplier widens.
- Machine-checked Python↔RTL equivalence flow for generated models
  (`sc_neurocore.compiler.equivalence_miter` + `equivalence_check`). Given the
  compiler's generated Verilog and an independent reference module,
  `prove_equivalence` builds a sequential miter — both instances driven by
  identical free inputs and a shared counter-derived reset, asserting output
  agreement on every post-reset cycle — runs it through SymbiYosys bounded model
  checking, and returns a real verdict (`proven` with the checked depth, or a
  counterexample with its failing assertion and trace path). This replaces the
  prior text-only equivalence sketch and the standalone `.sby` script generator,
  which are now cross-referenced to the runnable flow. Bounded model checking
  proves equivalence to a configurable depth; unbounded k-induction is available
  but not the default (wide-multiplier datapaths need an invariant to converge).
  The proof functions raise when `sby`/`yosys` are absent, and the tests skip in
  that case.
- Machine-checked RTL property proofs for adaptive-precision evidence bundles
  (`sc_neurocore.compiler.formal_property_check` + a shared
  `sc_neurocore.compiler._sby_runner`). `write_precision_formal_evidence_bundle`
  now renders a real synthesisable bounded-error monitor (the RTL `.v`, which the
  previous bundle named but never wrote) and a bound assertion checker whose
  immediate `assert`/`assume` statements encode the plan's safety obligations —
  accumulated error within the claimed total bound, sequencer within the declared
  bitstream length, and an accumulator wide enough that it never wraps — in place
  of the earlier placeholder comments. `prove_property` proves the RTL satisfies
  those obligations via SymbiYosys bounded model checking and returns a real
  verdict; because the accumulator stops updating after the bitstream length, the
  bounded proof to `length + 2` cycles is complete, not merely bounded. The
  bundle gains an `execute` flag: `execute=False` (default) writes the artefacts
  deterministically with no external tools, while `execute=True` runs the proof
  when the toolchain (`sby`/`yosys`/`z3`) is present and records the machine-checked
  verdict — or a skip reason when it is absent, never a fabricated pass. The
  equivalence and property runners now share one audited `sby` invocation and
  verdict-parse boundary. (yosys 0.33 silently ignores SystemVerilog `bind`, so
  the checker is instantiated explicitly under `` `ifdef FORMAL `` rather than
  bound in.)
- Unbounded k-induction for the adaptive-precision property proof. The bounded
  monitor's assertion checker now carries a strengthening lemma
  (`err_acc <= step_count * per_step`) that is 1-inductive, so
  `write_precision_formal_evidence_bundle(..., unbounded=True)` (and
  `prove_property(mode="prove")`) prove the obligations by k-induction — an
  unbounded proof whose depth is a small constant independent of the bitstream
  length, where bounded model checking's completeness depth scales with it. The
  lemma is trivially true under BMC, so both modes share one checker. k-induction
  adds a third outcome to the runners: an *inconclusive* result (the base case
  holds but the induction step does not converge) is recorded as `proven=False`
  with an `UNKNOWN` verdict and no counterexample — never a fabricated pass — and
  is distinguished from a tool failure, which still raises.
- GPU Izhikevich neuron batch runner (`sc_neurocore_engine.GpuIzhikevichBatch`, behind
  the Rust `gpu` feature), extending the GPU neuron-dynamics path. A wgpu/WGSL compute
  shader runs one thread per neuron, each looping all steps internally with a constant
  current and applying the two half-steps of the Euler update then the `v ≥ v_peak`
  threshold with reset `v ← c`, `u ← u + d`; it returns row-major `[n_neurons × n_steps]`
  spikes (`int32`) and voltages (`float32`). It mirrors the CPU `neuron::Izhikevich`
  model; because WGSL has no `f64` the arithmetic is `f32`, so GPU tests check agreement
  with the `f64` CPU oracle by tolerance — a tight sub-threshold voltage trace, spike
  count within a small margin, and firing-rate monotonicity in the drive current. Like
  the LIF kernel this is O(N) per-neuron work with no inter-neuron coupling, so the rayon
  CPU stays ahead across the benchmarked range; a Criterion benchmark records the measured
  CPU/GPU order. Tests and benches self-skip when no GPU adapter is present.
- GPU Kuramoto oscillator integrator (`sc_neurocore_engine.GpuKuramoto`, behind the
  Rust `gpu` feature), extending the GPU neuron-dynamics path beyond the LIF batch
  kernel. A wgpu/WGSL compute shader runs one thread per oscillator, each summing its
  coupling row `K_nm·sin(θ_m − θ_n)` over all others, so the O(N²) all-to-all coupling
  is fully parallel; the host ping-pongs two phase buffers across the Euler steps in a
  single command encoder. It mirrors the noise-free baseline of the CPU
  `KuramotoSolver`; because WGSL has no `f64` the arithmetic is `f32`, so a GPU test
  checks agreement with the `f64` CPU solver within tolerance (order parameter and
  per-oscillator circular distance) plus a phase-locking sanity check, and a Criterion
  benchmark sweeps oscillator counts for the CPU/GPU crossover. Tests and benches
  self-skip when no GPU adapter is present.
- Learned quantisers for quantisation-aware training in `sc_neurocore.qat`,
  extending the previous straight-through/ternary support. `LSQLinear` /
  `LSQQuantizer` implement Learned Step Size Quantization (Esser et al. 2020):
  the quantiser step size is a trainable parameter — per-tensor or
  per-output-channel — learned jointly with the weights, with the paper's
  step-size gradient and `2*mean(|w|)/sqrt(qmax)` initialisation.
  `PACTActivation` implements PACT (Choi et al. 2018): a learnable clipping
  bound bounds the activation range before uniform quantisation.
  `MinMaxObserver` and `PerChannelMinMaxObserver` derive per-tensor or
  per-channel `(scale, zero_point)` from calibration statistics (with a
  `fake_quantize` helper). `LSQPACTLIFNet` wires LSQ per-channel weights and a
  PACT-quantised analogue input into a feedforward LIF SNN end to end.
- QCFS conversion route in `sc_neurocore.conversion.convert`. When the source
  model carries `QCFSActivation` layers, conversion now uses their learned
  per-layer thresholds directly (no calibration pass), adopts the layers'
  trained timestep budget when `T` is unset, and pre-loads each IF neuron to a
  membrane potential of `theta / 2` — the optimal shift (Bu et al. 2022) that
  cancels the quantisation flooring bias, giving near-lossless conversion of a
  QCFS-trained ANN. `ConvertedSNN` gained an `initial_membrane_fraction` field
  carrying this shift (`0.0` reproduces the threshold-balancing route). A new
  `replace_relu_with_qcfs` helper substitutes every `ReLU`/`ReLU6` in a model
  for a `QCFSActivation`, preparing it for conversion-aware fine-tuning.
- Pre-synthesis area, latency, and power estimate for the folded interconnect
  (`sc_neurocore.energy.estimate_folded_area` → `FoldedAreaEstimate`). It maps a folded
  compile's `FoldedResourceMetrics` onto the existing Yosys-calibrated per-block costs in
  `energy/fpga_models.py` — one combinational PE per neuron type, one DSP (or a LUT-based
  multiply on a DSP-less target) per shared multiplier, a `data_width`-wide weight/threshold
  /bias ROM mux per multiplier column, the per-neuron spike-bus double-buffer flip-flops, the
  sequencer counters, and `state_ram_bits` of state BRAM — and turns `cycles_per_tick` into
  latency, time, and energy per tick. `sc-neurocore compile-nir --interconnect folded` prints
  the estimate and persists it under an `area_estimate` block in `folded_metrics.json` (skipped
  for the non-FPGA `web` target). The estimate inherits the underlying primitives' ~20%
  accuracy and introduces no new uncalibrated coefficients.
- Folded (time-multiplexed) FPGA interconnect for `compile_network_to_fpga`,
  opt-in via `interconnect="folded"` and the `sc-neurocore compile-nir
  --interconnect folded` flag. Multi-layer networks share one combinational
  processing element per neuron type (a per-type PE pool) across every neuron of
  that type, hold per-neuron state in a per-population BRAM, and a single sequencer
  walks each population in turn, advancing the whole network in `neurons + 1`
  cycles per timestep. A single global spike bus committed at the end of each tick
  gives recurrent and inter-population spiking fan-in a stable prior-tick
  double-buffer. Per-source synaptic delays fold via a depth-`d` `spike_bus` history
  shift-register (a delay of `d` ticks reads the bus committed `d` ticks ago). NIR
  `Threshold` transforms fold too: a source threshold gates each sign-extended weight on
  the source value (spike magnitude or external input); a destination threshold replaces a
  connection's per-neuron weighted sum with one spike-magnitude when it exceeds the
  threshold (selected from the per-neuron weight ROM). A per-destination-neuron connection
  bias folds as a constant ACC_WIDTH term added to that connection's fan-in (held in the
  same per-neuron ROM), so a destination threshold wraps the bias along with the weights.
  An analogue source population (`li`/`cuba_li`/`integrator`, whose output is the membrane
  voltage) folds via a global voltage bus — one `DATA_WIDTH` word per analogue source
  neuron, committed once per tick like the spike bus — that the destination multiplies by
  the weight (or threshold-gates), mirroring the direct path's registered `v_out`; a
  delayed analogue source reads a depth-`d` voltage-bus history register, the analogue of
  the spike-bus history. Bit-exact with the direct path (golden co-simulation parity for
  connection-less, external-weighted, recurrent-spiking, two-population
  feedforward/recurrent, delayed recurrent / mixed-per-column-delay two-population,
  external/spiking source-threshold, mixed destination-threshold, inter-population
  source-threshold, external-bias, biased destination-threshold, analogue-source, analogue
  source-threshold, and delayed-analogue fan-in). Reports a `FoldedResourceMetrics` summary
  (populations, processing elements, shared multipliers, state-RAM bits, cycles per tick,
  collapsed direct instances) on the result, in the CLI output, and as a
  `folded_metrics.json` artefact. The folded subset now covers every direct fan-in shape
  except a delayed *external* (non-population) source connection, which falls back to
  direct. Never auto-selected; the direct/AER paths and the SC-NIR source-handoff manifest
  are unchanged.

### Go services namespace docstring ratchet
- Promoted `accel.go.services` into the scoped NumPy docstring policy, added
  an AST-visible package docstring, declared the checked-in Go service file and
  package-boundary globs, and extended the acceleration mirror-authority tests
  to cover the services namespace at 100% exact-file coverage. No Go runtime,
  polyglot mirror, or benchmark-dispatched path changed.
### Changed
- The engine's Z3-backed safety-verification supervisor is now behind an optional
  `z3` Cargo feature, on by default so the published wheel and CI test matrix are
  unchanged. Building the engine with `--no-default-features` skips the bundled-Z3
  C++ compile (and omits the `PySpikingControllerPool` class), cutting a from-scratch
  engine check from minutes to ~30 s for fast or instrumented builds (e.g. `cargo-fuzz`
  targets) that do not need the verifier. `z3::` was already contained to
  `engine/src/supervisor.rs`; a CI step guards that the no-Z3 build stays compilable.
- Reconciled the lightweight dict-form NIR importer
  (`sc_neurocore.compiler.intelligence.import_nir_graph`) with the authoritative
  `nir_bridge`. It previously mapped only two node types to hand-written,
  divergent ODE strings; it now derives every equation from the shared canonical
  templates (extracted to `sc_neurocore.nir_bridge.neuron_templates`, the single
  source of truth also used by the FPGA back-end) and covers the six NIR
  point-neuron types (LIF, IF, LI, CuBa-LIF, CuBa-LI, integrator) plus an
  explicit Izhikevich extension, resolving framework/case aliases and falling
  back to a leaky integrator for unknown types. `NIRGraph` now also carries the
  full multi-compartment `state_equations`, `thresholds`, `resets`, resolved
  `parameters` and `node_types`. The docstring no longer overstates the module's
  role: it is a dependency-free convenience front-end, and
  `sc_neurocore.nir_bridge.from_nir` remains the authoritative typed importer for
  real `nir.*` graphs, affine/convolutional layers, subgraphs and hardware
  lowering.
- Rebuilt the bit-true simulation kernel
  (`sc_neurocore.compiler.intelligence.generate_bittrue_kernel`) onto a real
  integer fixed-point lowering. The previous C/Rust ``step`` was a no-op that
  only carried the equation in a comment while claiming identity with the RTL;
  it now advances each state variable with a genuine wrap-truncate multiply and
  saturating explicit-Euler accumulate via the new
  `sc_neurocore.compiler.c_fixed_emitter`, and its documentation no longer
  overstates the guarantee. A new
  `generate_bittrue_kernel_from_neuron` emits a whole-neuron C or Rust kernel —
  dt scaling, parameter encoding, threshold, reset rules and spike sequencing —
  that reproduces `compile_to_verilog` bit-for-bit; the identity is proven by an
  iverilog co-simulation that drives the compiled RTL and the kernel with the
  same stimulus and requires equal per-cycle state traces (LIF, Izhikevich,
  FitzHugh-Nagumo, a tanh cell and an integrate-and-fire, at Q8.8 and Q16.16),
  with the Rust kernel checked against the C kernel for the same guarantee.
- Rebuilt the HLS C++ export (`sc_neurocore.compiler.intelligence.generate_hls_cpp`)
  onto a real AST-based lowering. A new `sc_neurocore.compiler.c_expr_emitter`
  (`CExprEmitter` / `emit_c_expr`) parses each ODE expression and emits valid
  `ap_fixed` C++ — translating Python operators, integer powers, roots, and the
  supported transcendentals (via `hls_math` and inline sigmoid/exprel helpers)
  instead of embedding the raw expression string. Equations are now Euler-integrated
  (`<var>_next = <var> + dt * d<var>`, matching the Verilog backend), the membrane
  variable resets by subtracting a configurable threshold on a spike, and free
  identifiers become function inputs so the generated unit is self-contained. The
  output is verified to compile with a host C++ compiler.
- Extracted the target-independent expression-lowering numerics — the
  compile-time constant folder, the 256-point symmetric sample grid, and the
  quantised transcendental LUT generators (exp, log, sqrt, tanh, cosh, exprel,
  sigmoid, sin, cos, cbrt) plus the supported-function vocabulary — into a new
  `sc_neurocore.compiler.expr_lut_tables` module. The Verilog expression emitter
  now delegates to it, so the generated RTL is byte-for-byte unchanged; the
  shared tables give future C/C++/Rust lowering backends a single bit-exact
  source of truth.

### Go acceleration namespace docstring ratchet
- Promoted `accel.go` into the scoped NumPy docstring policy, added an
  AST-visible package docstring, declared the maintained Python ctypes loader
  entry points and broad Go service namespace globs, and extended the
  acceleration mirror-authority tests to cover the Go namespace at 100%
  exact-file coverage. No Go runtime, polyglot mirror, or benchmark-dispatched
  path changed.
### Fixed
- `graph::CsrMatrix::new` validated only the lengths of the CSR arrays, not that
  `row_offsets` is non-decreasing and starts at 0 or that every column index is `< n_cols`.
  A malformed CSR (reachable from Python via the sparse graph constructor) was therefore
  accepted and later panicked with an index-out-of-bounds when expanded by `to_dense`.
  `new` now rejects a non-monotonic offset array, a non-zero first offset and any
  out-of-range column index, so a constructed matrix is always safe to use. Found by the
  new `csr_matrix` fuzz target.
- The opt-in folded FPGA interconnect (`interconnect="folded"`) built its shared
  processing element from the *quantised* population, so `Q88.encode` ran a second time
  over already-quantised parameters and baked a corrupt value into the PE for every graph
  carrying explicit parameters (a 16-bit `tau = 5120` re-encoded to
  `5120 × 256 mod 2**16 = 0`). Real NIR networks always carry explicit parameters, so the
  folded PE was silently generated with `tau = 0`. Folded PE parameters are now built from
  the real-valued parameters (`_dequantised_pop`, a lossless rescale for fixed-point
  values), matching the per-instance module the direct interconnect emits. The
  double-encoding had gone undetected because the folded co-simulation suites used
  template-default (already real-valued) parameters; a co-simulation of a real
  explicit-parameter network now guards it.
- Hardened the public swarm agent/evolver contracts. `AgentConfig`,
  `SwarmAgent.weights`, `think`, `act`, `reset`, `EvolverConfig`,
  `evaluate_individual`, mutation, and generation execution now reject
  malformed or non-finite inputs before mutating state; the touched swarm
  modules are strict-mypy clean, enforced by the scoped public-docstring
  policy, and covered at 100% exact-file coverage by the focused swarm suite.
- `network.topology.scale_free` now rejects invalid Barabasi-Albert dimensions
  and non-finite weights before allocating the graph or sampling attachment
  probabilities, closing the `m=0` divide-by-zero path and the `m >= n`
  empty-graph path. The existing topology generator tests are strict-typed and
  cover the fail-closed parameter boundary at 100% exact-file coverage.
- Fixed the direct FPGA interconnect emitting an assignment to an undeclared
  delay register when a connection mixed zero and non-zero per-source-column
  synaptic delays: a delayed source's register chain was registered (and its
  `*_spike_d1 <= *_spike` shift emitted) even for columns with delay zero, which
  have no declared chain, so the generated RTL failed to elaborate. Undelayed
  columns are no longer given a register chain.
- Removed duplicate hardware-profile registrations in the built-in platform
  modules: `cortical_labs_dishbrain` and `finalspark_neuroplatform` were each
  registered four times, `biomemory_dna` three times, and `belousov_zhabotinsky`,
  `fujitsu_digital_annealer`, `rl_toffoli_asic`, `ibm_microfluidic`,
  `mems_resonator`, `brainscales2` and `spinnaker2` twice — with conflicting
  fields silently resolved by last-registration-wins. Each profile name now
  registers exactly once with its previously-effective values (behaviour
  unchanged); `_reg` rejects a duplicate name unless `allow_override=True`, and
  the `hardware_profiles`/`research_platforms` guides now match the registered
  values.
- Corrected the `online_learning` package docstring, which advertised "e-prop,
  RTRL, and forward-gradient methods" though only e-prop (`EpropTrainer`) and an
  eligibility-based online trainer (`OnlineTrainer`/`OnlineLIFLayer`) are
  implemented. The docstring now lists only the shipped methods.

### Precision solver width and budget hardening
- Hardened `compiler.precision_solver` with sign-inclusive integer-width
  calculation, finite/range validation, unsigned-negative range rejection,
  invalid budget/alignment rejection, and aligned budget reductions that
  recompute datapath width from the integer floor and reduced fraction. Added a
  dedicated strict-typed precision-solver test module with 100% exact-file
  coverage, promoted the solver into the scoped NumPy docstring policy,
  refreshed generated API documentation, and regenerated capability snapshots
  for the new test-file count. No polyglot mirror or benchmark-dispatched path
  changed.
### Physics and mathematics hardening
- Rebuilt `MarderSTGNeuron` (Liu, Golowasch, Marder & Abbott 1998 stomatogastric
  ganglion neuron) as a faithful thirteen-state model integrated with
  fourth-order Runge-Kutta, replacing the prior single-step forward-Euler
  approximation. Restored the CaS inactivation gate and the K-C activation gate
  (`h_cas`, `m_kca`), replaced the constant gate time constants with the
  published voltage-dependent `tau(V)` functions, made the K-C activation
  voltage- and calcium-dependent, switched the calcium reversal to the Nernst
  equation, and corrected the calcium dynamics to the published 20 ms relaxation
  form (transcribed from ModelDB 93321). Propagated the same RK4 model across the
  Rust engine, Rust safety mirror, Julia, and Go surfaces with bit-exact
  Python-Rust spike-count parity and a reference Mojo kernel; refreshed the model
  descriptor, the model documentation, the Rust benchmark figure, and the test
  suite to 49 multi-angle tests. The neuron is an endogenous burster (fires at
  zero injected current).
- Rebuilt `PinskyRinzelNeuron` (Pinsky & Rinzel 1994, 2-compartment CA3
  pyramidal cell) as a faithful eight-state model integrated with fourth-order
  Runge-Kutta, replacing the prior single-step forward-Euler approximation.
  Restored the separate voltage/Ca-dependent K-C activation gate `c` and the
  dendritic calcium state `ca` (previously conflated), added the membrane
  capacitance `cm = 3`, corrected `chi(Ca) = min(Ca/250, 1)` (previously a
  voltage expression), and aligned the Na/K-DR/Ca/K-C rate functions and
  reversal convention with the published kinetics (ModelDB 35358). Propagated
  the same RK4 model across the Rust engine, Rust safety mirror, Julia, and Go
  surfaces with Python-Rust spike-count parity and a reference Mojo kernel;
  refreshed the model descriptor (RK4, `ca`, `cm`), the model documentation, the
  Rust benchmark figure, and the test suite to 54 multi-angle tests. The model
  fires repetitively at low somatic drive and enters depolarisation block at
  high drive (non-monotonic f-I).

### Precision config docstring and coverage ratchet
- Promoted `compiler.precision_config` into the scoped NumPy docstring policy,
  added compliant public docstrings for fixed-point and block-floating
  precision value-object properties and manifest helpers, and expanded the
  focused compiler precision tests to cover validation, range maths, manifest
  payloads, exponent-layout delegation, and saturating fixed-point encoding at
  100% exact-file coverage. The generated API reference was refreshed. No
  runtime contract, polyglot mirror, or benchmark-dispatched path changed.

### MLIR emitter docstring and coverage ratchet
- Promoted `compiler.mlir_emitter` into the scoped NumPy docstring policy,
  added compliant docstrings for the MLIR node, bundle, emitter, wire, and
  operation helpers, strict-typed the focused MLIR tests, and covered
  `MLIRBundle.to_dict()` so the emitter reaches 100% exact-file coverage under
  the focused selector. The generated API reference was refreshed. No runtime
  contract, polyglot mirror, or benchmark-dispatched path changed.

### IR type checker docstring ratchet
- Promoted `compiler.ir_type_checker` into the scoped NumPy docstring policy,
  added compliant public class docstrings for the signal-domain enum and edge
  record, strict-typed the focused IR type-checker tests, and refreshed the
  generated API reference. The existing focused selector keeps the module at
  100% exact-file coverage. No runtime contract, polyglot mirror, or
  benchmark-dispatched path changed.

### Cortical column coverage hardening
- Added strict-typed cortical-column coverage contract tests for static scale
  validation, auto-backend single Rust SpMV fallback, and delayed per-bin
  injection. `network.cortical_column` now reaches 100% exact-file coverage
  under the focused fast selector, and the generated capability inventory was
  refreshed. No runtime contract, polyglot mirror, or benchmark-dispatched path
  changed.

### Native learning bridge docstring hardening
- Promoted `_native.learning_bridge` into the scoped NumPy docstring policy,
  documented the public Rust, Rust-WGPU, and Torch bridge methods, covered the
  single-step FFI, layer reset, Torch reset-scope, bit-spec length, and
  PyTorch-unavailable fallback paths, and refreshed the autonomous-learning docs
  plus generated API reference. No FFI contract, polyglot mirror, or benchmark
  path changed.

### DARTS NAS docstring hardening
- Promoted `nas.darts_sc_nas` into the scoped NumPy docstring policy, added
  public docstrings for the differentiable bitstream-selection methods,
  strict-typed the focused DARTS NAS tests, refreshed the NAS API guide and
  generated API reference, and preserved the torch-gated runtime behavior. No
  polyglot mirror or benchmark-dispatched path changed.

### NAS search surface docstring hardening
- Promoted `nas.search_space`, `nas.search`, and `nas.equiv` into the scoped
  NumPy docstring policy, added missing public summary/property docstrings,
  strict-typed the focused NAS tests, refreshed the NAS API guide and generated
  API reference, and removed local NAS type-ignore escapes. No runtime contract,
  polyglot mirror, or benchmark-dispatched path changed.

### SC-NAS engine docstring and coverage hardening
- Promoted `nas.sc_nas_engine` into the scoped NumPy docstring policy, added
  compliant public docstrings for the hardware-aware search/report surface,
  strict-typed the focused NAS engine tests, covered the optional Rust
  tournament import branch through the module import boundary, and refreshed
  the public NAS docs plus generated API reference. No runtime contract,
  polyglot mirror, or benchmark-dispatched path changed.

### SCConv layer docstring and test hardening
- Promoted `layers.sc_conv_layer.SCConv2DLayer` into the scoped NumPy
  docstring policy, added compliant module, initialization, and forward
  docstrings, strict-typed the focused SCConv tests, refreshed the public layer
  docs and generated API reference, and verified the production file at 100%
  exact-file coverage. No runtime contract, polyglot mirror, or
  benchmark-dispatched path changed.

### Compiler MLIR export hardening
- Hardened `export.compiler_export.CompilerExporter` with explicit `mlir`
  target validation, empty-graph rejection, duplicate node/output detection,
  wrong-arity and unsupported-node failures, missing external shape checks,
  graph-input/output collision rejection, and positive tensor-dimension
  validation before SSA MLIR emission. Removed the embedded demo path, covered
  the module at 100% exact-file coverage, strict-typed the focused tests, and
  promoted the compiler exporter into the scoped NumPy docstring policy.

### DVS input contract hardening
- Hardened `interfaces.dvs_input.DVSInputLayer` with fail-closed dimension,
  decay, AER address, timestamp, polarity, and bitstream-length validation
  before event-surface mutation. Empty event batches now return probability
  frames instead of exposing the mutable internal surface, and invalid
  cross-batch timestamp rewinds leave `surface` and `last_update_time`
  unchanged. The Rust safety mirror, Julia validation mirror, and Mojo FFI
  validation shim now enforce the same DVS boundaries, and the advanced-module
  DVS benchmark now uses monotonic precomputed event batches. The public DVS
  input surface is covered at 100% exact-file coverage under strict mypy plus
  the scoped NumPy docstring policy.

### Fisher-Posner LIF contract hardening
- Hardened `quantum_cognition.fisher_posner` with fail-closed neuron-id,
  timestep, voltage, membrane-time-constant, ATP-domain, and step-current
  validation before state mutation. Invalid input current now fails without
  advancing counters, membrane voltage, ATP, or spin-pool measurement state.
  The public Fisher-Posner LIF surface is covered at 100% exact-file coverage
  under strict mypy plus the scoped NumPy docstring policy.

### RNG utility contract hardening
- Hardened `utils.rng.RNG` with fail-closed seed, normal, uniform, and
  Bernoulli parameter validation before generator state advances. Scalar draws
  now expose Python scalar types, shaped draws return dtype-stable NumPy
  arrays, and the public utility is covered at 100% exact-file coverage under
  strict mypy plus the scoped NumPy docstring policy.

### L13 holonomic source-field adapter hardening
- Hardened `adapters.holonomic.l13_source` with fail-closed parameter,
  timestep, feedback-rank, feedback-emptiness, finite-value, decode, and
  no-mutation validation before vacuum/Fisher state updates. Scalar feedback
  now broadcasts across the vacuum lattice, and mismatched vector or batch
  feedback projects deterministically by mean drive. The public adapter is
  covered at 100% exact-file coverage under strict mypy plus the scoped NumPy
  docstring policy. The Rust safety mirror now validates real state, the Julia
  mirror is callable, and the Mojo contract shim builds as a shared library
  with callable validation helpers.

### L6 holonomic planetary adapter hardening
- Hardened `adapters.holonomic.l6_plan` with fail-closed parameter, timestep,
  input-rank, input-width, non-empty-row, and finite-value validation before
  Gaia-field state mutation. Mismatched upstream region counts now broadcast a
  deterministic mean regional drive across configured planetary regions, and
  the public adapter is covered at 100% exact-file coverage under strict mypy
  plus the scoped NumPy docstring policy. The Rust safety mirror now enforces
  the same no-mutation input contract, the Julia mirror is callable, and the
  Mojo contract shim builds as a shared library with callable validation
  helpers.

### L9 holonomic memory adapter hardening
- Hardened `adapters.holonomic.l9_mem` with fail-closed parameter, timestep,
  input-rank, input-width, non-empty-row, and finite-value validation before
  TSVF state mutation. Mismatched upstream slot counts now tile
  deterministically across configured memory slots, and the public adapter is
  covered at 100% exact-file coverage under strict mypy plus the scoped NumPy
  docstring policy. The Rust safety mirror now validates real L9 state, the
  Julia mirror is callable, and the Mojo contract shim builds as a shared
  library with callable validation helpers.

### GPU fallback reduction hardening
- Hardened the CuPy/NumPy GPU fallback and shared packed-bitstream vector
  reductions against coverage-time NumPy reloads by routing reductions through
  the active NumPy module with explicit zero initials. Covered the CuPy-absent
  import branch, GPU CPU fallback, and vector pack/unpack paths at 100%
  exact-file coverage under strict mypy. The benchmark suite now synchronizes
  CUDA only while the GPU runtime is still live, so the GPU section completes
  under the NumPy fallback when local CUDA discovery fails.

### Mojo runner contract hardening
- Covered `accel.mojo.runner` at 100% exact-file coverage, including the
  fail-closed missing-`kernels.mojo` constructor path. Promoted the runner to
  the scoped NumPy docstring policy and corrected the Mojo acceleration docs so
  `build`, `run_benchmark`, `popcount`, and `lfsr_encode` describe the current
  fallback and failure contracts exactly.

### Quantum Studio telemetry hardening
- Covered `QuantumStudioHook` snapshot, compact JSON event, and debug
  representation contracts at 100% exact-file coverage under strict mypy.
  Added the telemetry hook to the scoped NumPy docstring policy and refreshed
  the quantum cognition API documentation for the Studio streaming surface.

### Host driver generator hardening
- Hardened generated Python and C host drivers so module names, parameter
  registers, setters, include guards, and C function prefixes are sanitized into
  valid identifiers before source emission. Empty module identifiers and
  sanitized parameter collisions now fail closed, C drivers expose parameter
  setters matching the Python surface, and `compiler.host_driver_gen` is covered
  at 100% exact-file coverage under strict mypy plus the scoped NumPy docstring
  policy.

### Compiler pipeline hardening
- Hardened `CompilerPipeline` path and tool boundaries: artifact paths now use
  `commonpath` work-directory validation, EDA executables are resolved to
  absolute paths before subprocess launch, and the public pipeline surface is
  covered by strict-typed contract tests plus the scoped NumPy docstring policy.

### Universal DSL contract hardening
- Covered the schema loader's explicit missing-path and Python pre-3.11 TOML
  fallback branches, TOML bool/list serialization, and default Verilog module
  name sanitization at 100% exact-file coverage. Added the Universal DSL public
  surface to the scoped NumPy docstring policy.

### SC-NIR compatibility hardening
- Strict-typed the SC-NIR compatibility matrix contract tests, kept the matrix
  validator at 100% exact-file coverage, and promoted the public compatibility
  audit surface into the scoped NumPy docstring policy.

### ONNX graph export hardening
- Fixed dependency-free `ONNXExporter` final-output metadata so a terminal
  `SC_POPCOUNT` emits an `int32` tensor instead of a bitstream bool tensor, and
  mapped operators without shape inference rules now fail closed. Covered the
  exporter at 100% exact-file coverage, strict-typed the focused tests, and
  documented the file exporter versus graph exporter split.

### Pipeline ingestion hardening
- Hardened `DataIngestor` so the reserved `labels` key is preserved as labels
  instead of normalized as a modality, modality arrays must be finite and share
  a sample axis, and scalar/empty inputs fail closed. Covered
  `pipeline.ingestion` at 100% exact-file coverage, strict-typed the focused
  tests, corrected public pipeline docs, and added the surface to the scoped
  NumPy docstring policy.

### Neuron package facade hardening
- Covered the `sc_neurocore.neurons` lazy facade at 100%, including optional
  Rust-dispatch opt-out, cache reuse, pure-Python model fallback, cached
  package-level model exports, and unknown-symbol `AttributeError` handling.
  Added the package facade to the scoped NumPy docstring policy.

### Descriptor schema contract hardening
- Covered the v2 model descriptor parser's defensive contract branches at 100%:
  missing and non-table sections, empty structural descriptors, scalar legacy
  state/parameter forms, string backend statuses, dynamics expression tables,
  invalid tag/range/year/numeric shapes, and single-author provenance fallback.
  Added `neurons.model_descriptor` to the scoped NumPy docstring policy.

### Descriptor generator hardening
- Added a fail-closed class-name guard to the model descriptor generator and
  narrowed legacy v1 schema fallback so missing schemas remain allowed while
  malformed curated schemas abort corpus refresh. Covered plain-constructor
  filtering, source-inspection fallbacks, merge curation preservation, and
  brought `neurons.descriptor_generator` into the scoped NumPy docstring policy.

### Descriptor catalogue hardening
- Added a fail-closed class-name guard for model descriptor lookup so the
  public descriptor helpers accept only public Python identifiers before
  filesystem access. Covered valid-absent descriptor branches, the aggregate
  catalogue coverage summary, and added `neurons.model_catalogue` to the scoped
  NumPy docstring policy.
### Fixed
- Expanded GOTM brain coverage to 100% exact-file evidence, covering the local
  LLM import-success path, valid directive parsing, invalid directive fallback,
  local endpoint exception fallback, and repr/docstring policy.
- Expanded L7 holonomic symbolic adapter coverage to 100% exact-file evidence,
  covering single-node routing, nonstandard ring coordinates, underflowed
  Metatron topology rejection without warnings, boolean integer parameter
  guards, invalid golden-ratio scale rejection, mismatched input broadcast, and
  touched docstring-policy surfaces while keeping the Rust safety mirror green.
- Expanded radical-pair RPM coverage to 100% exact-file evidence, covering
  invalid quadrature, explicit hyperfine tensor construction/state telemetry,
  tensor-shape rejection, zero-nucleus density helper behavior, oversized dense
  nuclear-bath rejection, non-positive kinetic parameter guards, and
  repr/docstring policy while keeping the Rust safety kernel green.
- Expanded quantum terminal dashboard coverage to 100% exact-file evidence,
  covering terminal-size fallback, narrow-terminal hidden-neuron telemetry,
  no-history rendering, spike-raster intensity bands, directive colour fallback,
  empty-scale ATP bars, and repr/docstring policy.
- Expanded spin-pool MPS coverage to 100% exact-file evidence, covering valid
  dense exact evolution, ATP observable site rejection, status/repr telemetry,
  and stable TEBD adjacency validation while keeping the Rust telemetry-kernel
  tests green.
- Expanded Kane silicon mapper coverage to 100% exact-file evidence, covering
  triangular and hexagonal donor placement, zero-site rejection, zero-distance
  exchange coupling, and the touched docstring policy surface while keeping the
  Rust safety-kernel parity tests green.
- Hardened Wong-Wang Julia, Go, and Mojo acceleration facades so `stim1`,
  `stim2`, and `xi` must be one-dimensional time-series before backend
  dispatch, and added module-specific dispatcher contracts for Go/Mojo loader
  failures, unavailable libraries, C return codes, length validation, and
  success return buffers.
- Hardened Wilson-Cowan Julia, Go, and Mojo acceleration facades so
  `ext_input` must be a one-dimensional time-series before backend dispatch,
  and added module-specific dispatcher contracts for Go/Mojo unavailable
  libraries, C return codes, and success paths.
- Preserved local result dtype for non-root MPI gather outputs and expanded the
  MPI driver contract tests to cover no-MPI constructor fallback, root gather
  collection, non-root empty-gather dtype, missing communicator fallback, and
  barrier delegation.
- Kept `sc_neurocore.accel.mojo.MojoKernelRunner` fail-closed when the optional
  Mojo runner import fails, replacing stale reload bindings with a placeholder
  that raises `RuntimeError` with the captured import reason.

### Physics and mathematics hardening
- Promoted `DendriticNMDANeuron` (two-compartment Jahr-Stevens NMDA Mg2+ block
  neuron) from raw dendrite-first Euler to candidate-first RK4 over
  `(v_soma, v_dend)`, with finite parameter/state/input validation,
  reset-on-commit soma spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced Go, Rust safety, Julia,
  and Mojo placeholders or broken paths with real dual-input RK4 mirrors,
  harmonised the Rust engine, added focused Python/Go/Rust tests, a Rust
  benchmark example, a five-backend local non-isolated benchmark artefact, and
  refreshed the model documentation with measured benchmark results. Python,
  Rust, Go, Julia, and Mojo agree on the 253-spike anchor at 20k steps /
  `i_soma=50.0`, `glutamate=0.5`.
- Promoted `NeuroGridNeuron` (reduced Neurogrid two-compartment analog EIF
  neuron) from raw Euler to candidate-first RK4 over `(v_s, v_d)`, with an
  event-limited soma stage cap at `v_peak`, finite input/state/candidate
  validation, reset-on-commit spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced placeholder Go and
  Rust safety mirrors, repaired Julia, added Mojo, harmonised the Rust engine,
  added focused Python/Go/Rust tests, a Rust benchmark example, a five-backend
  local non-isolated benchmark artefact, and refreshed the model documentation.
  Python, Rust, Go, Julia, and Mojo agree on the 94-spike anchor at 20k steps /
  current 100.0.
- Promoted `HayL5PyramidalNeuron` (reduced three-compartment Layer 5 thick-tufted
  pyramidal cell) from four raw Euler sub-steps to candidate-first RK4 over the
  nine-state `(v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a)` system, with
  finite input/state/candidate validation, non-negative tuft-calcium candidates,
  dual soma/tuft input parity, and an explicit `integrator="baseline_euler"`
  regression path. Replaced broken or placeholder Go, Rust safety, Julia, and
  Mojo surfaces with real RK4 mirrors, harmonised the Rust engine, added focused
  Python/Go/Rust tests, a Rust benchmark example, a five-backend local
  non-isolated benchmark artefact, and refreshed the model documentation. Python,
  Rust, Go, Julia, and Mojo agree on the 1-spike anchor at 20k steps /
  `current_soma=10.0`, `current_tuft=0.0`; the dual-input anchor is 4 spikes at
  20k steps / `current_soma=5.0`, `current_tuft=5.0`.
- Promoted `DeSchutterPurkinjeNeuron` (compact De Schutter & Bower Purkinje cell)
  from five raw Euler sub-steps to candidate-first RK4 over the seven-state
  `(v, h_na, n_k, m_cap, h_cap, q_kca, ca)` system, with finite input/state/
  candidate validation, non-negative calcium candidates, and an explicit
  `integrator="baseline_euler"` regression path. Replaced the decorative Go,
  Rust safety, and Mojo placeholders with real RK4 mirrors, harmonised the Rust
  engine and Julia surface, added focused Python/Go/Rust tests, a Rust benchmark
  example, a five-backend local non-isolated benchmark artefact, and refreshed
  the model documentation. Python, Rust, Go, Julia, and Mojo agree on the 1-spike
  anchor at 20k steps / current 500.0.
- Promoted `MulticompartmentMCNNeuron` (Spiking-WM dual-dendrite working-memory
  cell) from raw forward Euler to candidate-first RK4 over the coupled
  `(u, v_basal, v_apical)` system, with finite input/state/candidate validation
  and an explicit `integrator="baseline_euler"` regression path. The Rust engine,
  Rust safety mirror, Go service, Julia mirror, and new Mojo kernel now share the
  same derivative order and threshold-reset rule. Added focused Python/Go/Rust
  tests, a Rust benchmark example, a five-backend local non-isolated benchmark
  artefact, and refreshed the model documentation. All five backends agree on
  the `49,999` spike anchor at 200k steps / basal current 3.2.

## [3.15.35] - 2026-06-26

### Physics and mathematics hardening
- Promoted `HillTononiNeuron` (Hill & Tononi 2005 thalamocortical sleep/wake
  cell) from a hard-coded forward-Euler step to candidate-first RK4 over the
  six-state `(V, h_na, n_k, m_h, h_t, na_i)` system — fast Na⁺, delayed-rectifier
  K⁺, `Ih`, T-type Ca²⁺, a sodium-dependent K⁺ current, and a saturating Na/K
  pump — with input validation and the opt-in `integrator="baseline_euler"`
  regression path. Replaced the decorative `accel/go/services` and Mojo
  placeholders (and corrected a wrong Go spike threshold) with real RK4 backends
  and harmonised the cross-language arithmetic so Python, Rust, Julia, Go, and
  Mojo reproduce the trajectory bit-for-bit: explicit `m·m·m`/`n·n·n·n`
  conductance powers, and the `I_KNa` Hill exponent `3.5` evaluated as
  `b·b·b·sqrt(b)` (an IEEE-754 exact decomposition) instead of a per-platform
  `pow`. Added a `_safe_exp` guard so the saturating gates stay finite under an
  out-of-range stimulus (Python `math.exp` would otherwise raise where the other
  backends return `+inf`), native Go RK4 parity/behaviour tests, a Go benchmark
  hook, a Rust benchmark example, Python RK4/fail-closed coverage, and a
  five-backend local non-isolated benchmark that fails closed unless every
  backend reports an identical spike count (694 at 200k steps / 10 nA).
- Promoted `DurstewitzDopamineNeuron` (Durstewitz, Seamans & Sejnowski 2000
  D1-modulated PFC cell) from a hard-coded forward-Euler step — which advanced
  the gates from the old voltage and then the voltage from the freshly updated
  gates, mixing two inconsistent states — to candidate-first RK4 over the
  three-state `(V, h_na, n_k)` system, with input validation and the opt-in
  `integrator="baseline_euler"` regression path. Replaced the decorative
  `accel/go/services` and Mojo placeholders with real RK4 backends and harmonised
  the cross-language arithmetic (explicit `m·m·m`/`n·n·n·n` conductance powers,
  the `mg / 3.57 · exp` Mg²⁺-block operand order, `math.exp`) so Python, Rust,
  Julia, Go, and Mojo reproduce the trajectory bit-for-bit. Added native Go RK4
  parity/behaviour tests, a Go benchmark hook, a Rust benchmark example, the
  Python RK4/fail-closed test coverage, and a five-backend local non-isolated
  benchmark that fails closed unless every backend reports an identical spike
  count (925 at 200k steps / 10 nA).
- Completed the `UpperMotorNeuron` (Pospischil 2008 corticospinal L5 pyramidal)
  polyglot backend coverage by adding the Mojo exponential-Euler kernel, raising
  it to a Python / Rust / Julia / Go / Mojo set. The membrane keeps its analytic
  frozen-conductance exponential-Euler step and the gates keep their closed-form
  steady/tau update — both unconditionally stable for the stiff sodium gate, so
  RK4 here would be a regression rather than a hardening. Added a Go benchmark
  hook, a Rust exponential-Euler benchmark example, and a five-backend local
  non-isolated benchmark that fails closed unless every backend reports an
  identical spike count.
- Promoted `EnergyLIFNeuron` from raw Euler membrane and metabolic-reserve
  updates to the exact constant-current flow for the coupled `(v, epsilon)`
  state across Python, Go, Julia, Mojo, and Rust safety surfaces. Added
  module-specific Python/Go/Rust exact-flow and invalid-state coverage,
  refreshed the public model documentation with measured five-backend timing
  rows, and added a local non-isolated benchmark gate for exact spike-count
  parity.
- Promoted `MATNeuron` from a split forward-Euler membrane update plus separate
  threshold decay to candidate-first RK4 over `(v, theta1, theta2)` across the
  Python reference, Go service, Julia mirror, Mojo helper, and Rust safety
  surface. Replaced the Go/Rust/Mojo placeholders with numeric parity surfaces,
  added Go/Rust tests and Python RK4/fail-closed coverage, refreshed the model
  documentation with measured five-backend timings, and added a local
  non-isolated benchmark gate for exact spike-count parity.
- Promoted `SFANeuron` from forward-Euler voltage plus separate adaptation
  decay to candidate-first RK4 over the coupled `(v, g_sfa)` adaptation ODE
  across Python, Go, Julia, Mojo, and Rust safety surfaces. Added native Go/Rust
  RK4 tests, refreshed Python module tests, a five-backend local non-isolated
  benchmark artifact, a regression-gate row, and updated model documentation.
- GPFA (Gaussian Process Factor Analysis) now uses a deterministic PCA
  initialisation (top singular vectors of the centred data with a fixed sign
  convention) instead of a random one, so results are reproducible and
  seed-independent. The EM loop is factored into reusable `gpfa_pca_init` and
  `gpfa_em` entry points with a backend dispatch contract that lets acceleration
  backends share an identical starting point, and the marginal log-likelihood is
  the exact Gaussian form. Added a full reference test suite (deterministic init,
  EM convergence, exact log-likelihood and its non-PSD guard, projection, dispatch).
- GPFA Rust backend (`py_gpfa_em`) now shares the deterministic initialisation and
  computes the exact marginal log-likelihood (previously an approximate
  residual-sum form), so it agrees with the NumPy reference to within float64
  round-off (same iteration count, trajectories within ~1e-10); selectable via
  `backend="rust"`, with a gated parity test.
- GPFA Julia backend (`accel/julia/analysis/gpfa.jl`, previously a stub) implements
  the same EM-from-initialisation contract and agrees with the NumPy reference to
  within ~1e-12; selectable via `backend="julia"`, gated parity test included.
- GPFA Go backend (`accel/go/gpfa/gpfa.go`, c-shared via cgo) implements the same
  EM-from-initialisation contract and agrees with the NumPy reference to within
  ~1e-12; selectable via `backend="go"`, with parity and loader-branch tests.
- GPFA Mojo backend (`accel/mojo/kernels/gpfa.mojo`, previously a stub) implements
  the same EM-from-initialisation contract over the `@export` raw-address FFI and
  agrees with the NumPy reference to within ~1e-10; selectable via `backend="mojo"`,
  with parity and loader-branch tests.
- GPFA now uses a structured Cholesky estimator across all five backends. The
  E-step, M-step and marginal log-likelihood operate on the `n_state × n_state`
  posterior precision (`n_state = n_latents · n_bins`) via Cholesky factorisations
  rather than a general elimination, and the log-likelihood uses the Woodbury
  identity and the matrix-determinant lemma so it never forms the dense
  `(n_neurons · n_bins)²` marginal covariance. This is the exact structured
  estimator of Yu et al. (2009): more numerically stable for the symmetric
  positive-definite systems and far cheaper at scale (the dense form would build,
  e.g., an 8000×8000 covariance where the structured form factors a 1600×1600
  precision). The Rust backend factors with `nalgebra`, Julia with native LAPACK,
  Go and Mojo with an in-place Cholesky, and the Python reference with SciPy; the
  structured likelihood is checked against a dense brute-force covariance (~1e-11).
- GPFA `backend="auto"` resolves to the structured Rust path when the engine is
  present and falls back to the NumPy reference otherwise. With the dense
  `n_obs`-sized solve removed, `benchmarks/bench_gpfa.py` measures the Rust backend
  as the fastest path (1.43x over NumPy on the reference workload; Mojo 0.87x, Go
  0.51x); the compiled backends remain available by name for cross-language parity
  and portability. Added `docs/api/gpfa.md` describing the deterministic init, the
  structured EM contract, the five backends and the benchmark.
- Phi* (integrated information, `analysis/phi_estimation.py`) now computes the
  Gaussian mutual information from covariance log-determinants taken via Cholesky
  (`MI = 0.5(log|Cov_X| + log|Cov_Y| - log|Cov_XY|)`) instead of a product/ratio of
  raw determinants with a `1e-300` clamp, which underflowed for larger channel
  counts; the single-channel covariance now uses the unbiased (`ddof=1`) estimator
  consistently. `phi_star(..., backend=...)` gains a polyglot dispatch over five
  parity-verified backends — NumPy, Rust (`py_phi_star`, re-exported from the
  bridge), Julia and Mojo (both previously non-functional stubs, now real), and a
  new Go c-shared backend — agreeing with the reference to ~1e-15 (Mojo ~1e-10).
  `backend="auto"` prefers Rust; `benchmarks/bench_phi.py` measures every compiled
  backend faster than NumPy (Rust ~7x, Julia ~4.4x, Mojo ~3.5x, Go ~3.0x). Added
  parity and loader-branch tests (phi_estimation.py at 100% statement coverage) and
  refreshed the `docs/api/analysis.md` Phi* section.
- Sorting-quality Mahalanobis metrics (`analysis/spike_stats/sorting_quality.py`)
  now evaluate the squared Mahalanobis distance `(x-μ)ᵀ Σ⁻¹ (x-μ)` through the
  Cholesky factor of the regularised cluster covariance (a triangular solve, then
  `Σ z²`) instead of forming the covariance inverse explicitly — more accurate for
  ill-conditioned cluster covariances and cheaper. `isolation_distance` and
  `l_ratio` share the kernel and gain a polyglot dispatch `backend=` over five
  parity-verified backends — NumPy, Rust (`py_isolation_distance` / `py_l_ratio`,
  re-exported from the bridge; the Rust path previously used a Gauss-Jordan
  inverse mislabelled as Cholesky), Julia and Mojo (both previously non-functional
  stubs, now real), and a new Go c-shared backend — agreeing with the reference to
  ~1e-13. `backend="auto"` prefers Rust; `benchmarks/bench_sorting_quality.py`
  measures Mojo ~4.6x, Go ~3.8x and Rust ~1.9x over NumPy on the reference workload
  (Julia ~0.5x, interop-bound on this size). Added Cholesky-kernel, parity and
  loader-branch tests (sorting_quality.py at 100% statement coverage), removed two
  unreachable defensive branches, and expanded the `docs/api/analysis.md`
  sorting-quality section.
- Fixed a latent type-inference error in the Rust Phi* test (`det.ln()` on an
  ambiguous numeric literal) that prevented the engine test target from compiling.
- Dimensionality reduction (`analysis/spike_stats/dimensionality.py`) now uses a
  deterministic, reproducible covariance eigendecomposition — eigenvalues in
  descending order with sign-canonicalised eigenvectors — across PCA, demixed PCA
  and factor analysis. The Rust backend's hand-rolled Jacobi eigensolver and
  Gauss-Jordan inverse are replaced by `nalgebra`'s symmetric eigensolver and
  Cholesky solves; factor analysis starts from a deterministic PCA initialisation
  (replacing a random one, so it is seed-independent) and solves its symmetric
  positive-definite systems by Cholesky. Each estimator gains a `backend=`
  dispatch over five parity-verified backends — NumPy, Rust (`py_pca_components`
  / `py_demixed_components` / `py_factor_loadings`, re-exported from the bridge),
  Julia and Mojo (both previously non-functional stubs, now real), and a new Go
  c-shared backend (cyclic Jacobi where no LAPACK is linked) — agreeing to ~1e-13.
  Dense symmetric eigendecomposition is LAPACK's strength, so on the reference
  workload (`benchmarks/bench_dimensionality.py`) the NumPy/LAPACK path is the
  fastest and `backend="auto"` resolves to it; the compiled backends are kept for
  cross-language parity and portability. Added a dedicated test module
  (dimensionality.py at 100% statement coverage), the benchmark and the
  `docs/api/analysis.md` dimensionality section.
- The Rust `cell_assembly_detection` backend (`analysis/network.rs`) replaces its
  hand-rolled Jacobi eigensolver with `nalgebra`'s symmetric solver, returning
  descending, sign-canonicalised eigenvectors — matching the NumPy/LAPACK
  reference the Python path already uses. No public API change.
- The Rust causality backend (`analysis/causality.rs`) now uses structured
  factorisations for its dense linear algebra. The per-frequency MVAR transfer
  function `H(f) = A(f)⁻¹` in spectral Granger causality and the directed
  transfer function is obtained from a single `nalgebra` LU factorisation of the
  non-Hermitian spectral matrix — one factorisation yields both the
  near-singularity test (its determinant) and the inverse — replacing a separate
  hand-rolled complex Gauss-Jordan inverse and Gaussian-elimination determinant.
  The VAR fit and the Granger sum-of-squared-errors solves now solve their
  ridge-regularised normal equations `XᵀX + εI` (symmetric positive-definite) with
  a Cholesky factorisation rather than generic Gaussian elimination, factoring
  once for all right-hand sides. The spectral-matrix assembly is shared across the
  spectral Granger, partial-directed-coherence and directed-transfer-function
  paths. Results match the previous implementation to float64 round-off; no public
  API change.
- The Fisher linear-discriminant decoder now solves its within-class scatter
  system with a Cholesky factorisation in both backends. The scatter matrix
  `S_w + εI` is symmetric positive-definite, so the per-class Fisher weights
  `w_c = S_w⁻¹ (mean_c − overall_mean)` come from a single factorisation reused
  across classes rather than an explicit matrix inverse (Python reference,
  `scipy.linalg.cho_factor`/`cho_solve`) or a per-class Gaussian elimination
  (Rust backend, `nalgebra`) — the numerically optimal route for an SPD system.
  Decoded classes are unchanged; the naive-Bayes decoder (diagonal covariance) is
  unaffected.
- The Rust world-model Kalman filter (`lgssm.rs`) factors the innovation
  covariance `S = C P_pred Cᵀ + R` with a single `nalgebra` Cholesky decomposition
  per timestep, replacing a hand-rolled Cholesky and an explicit matrix inverse.
  The one factor serves the log-determinant (`2 Σ log Lᵢᵢ`), the innovation
  quadratic form (`S⁻¹ innov` via triangular solves), and the Kalman gain
  `K = P_pred Cᵀ S⁻¹` — computed as `Kᵀ = S⁻¹ (C P_pred)` so `S⁻¹` is never formed
  explicitly. Matches the NumPy/LAPACK reference
  (`world_model/predictive_model.py`) to float64 round-off; the Rust-vs-Python
  parity suite (means, covariances, log-likelihood, atol 1e-9) stays green. No
  public API change.

### Studio platform
- Documented the optional `sc_neurocore.federation` Hub-facing Studio federation
  surface with a dedicated API page and navigation entry, covering schema-A
  manifest emission, evidence bundles, and verifiable-honesty envelopes.
- The FPGA synthesis panel can export a synthesis-scoped evidence bundle from
  the latest single-target or all-target worker job, list the bundle artefacts,
  and download them through the authenticated Studio job-artifact route.

### Public bitstream-inference API
- Restored a stable public stochastic-inference surface over caller-owned packed
  weight bitstreams. `sc_neurocore.accel.sc_forward(weights_packed, input_probs,
  *, length, backend, seed)` returns the AND-then-popcount estimate of
  `weights @ input_probs` for unipolar stochastic computing; the input encoder is
  the deterministic 16-bit LFSR comparator, so the Rust accelerated path and the
  NumPy fallback are bit-identical for a fixed seed. Added
  `sc_neurocore.accel.get_backend()` / `available_backends()` as a fastest-first
  backend selector (Rust accelerated path, NumPy fallback), the Rust engine
  `sc_forward_packed` + PyO3 `py_sc_forward_packed`, a parity test proving the
  estimate matches the dense reference within stochastic tolerance and that the
  Rust and NumPy paths agree exactly, a Rust-vs-NumPy benchmark with a committed
  artefact, and documentation.
- `BitstreamEncoder` again accepts `BitstreamEncoder(length=..., seed=...)`:
  `x_min`/`x_max` default to the unipolar probability domain `[0, 1]`, with
  explicit ranges still supported.
- `import sc_neurocore` is now lightweight and torch-free: public symbols and
  submodules load lazily (PEP 562) instead of eagerly importing the torch-pulling
  graph (plasticity, datasets, layers). Downstream consumers can run their
  coverage tracers over code that imports sc-neurocore without the torch
  C-extension crash; torch loads only when a torch-dependent feature is used.

### Timing-aware formal verification
- Added a two-flop clock-domain-crossing synchroniser property template to the
  timing formal framework: `sc_cdc_two_flop_monitor` (in
  `hdl/formal/timing/timing_wrapper_lib.sv`) and the `SC_ASSERT_CDC_TWO_FLOP`
  macro (in `timing_assertions.svh`). A consumer binds it over its own
  destination-domain synchroniser flops; the monitor proves the output is the
  source delayed by exactly the synchroniser depth with no combinational path past
  the last flop. The property is expressed in the open-source Yosys/SymbiYosys
  procedural-immediate-assertion subset (no concurrent SVA). Added a worked
  `example_cdc_two_flop_synchroniser.{sv,sby}` proof, a SymbiYosys-gated proof
  test, a mutation test proving a one-flop synchroniser is rejected, and a
  consumable-surface section (bounded-latency macros, CDC template, SymbiYosys task
  convention) in `docs/formal/timing_aware_properties.md`.

### Studio platform
- Added Projects panel artifact download controls for exported project evidence
  bundles, reusing the authenticated Studio job-artifact endpoint and showing
  path-confined artifact names, sizes, and SHA-256 labels in the workbench.
- Added Projects panel evidence-bundle export for saved project workspaces,
  using the bounded Studio evidence worker from the normal workbench UI.
- Added Admin panel audit-archive restore controls for path-free payload
  validation and confined restore-artifact materialization, wired to the
  existing admin-gated validate and restore endpoints.
- Added Admin panel audit-archive controls for quarantine archive creation,
  retention review, and prune-candidate purge execution.
- Added an admin-gated Studio audit quarantine archive retention purge endpoint
  that removes only archive jobs marked as prune candidates.
- Added an admin-gated Studio audit quarantine archive restore job that
  materializes validated archive rows as confined evidence artifacts.
- Added an admin-gated Studio audit quarantine archive retention endpoint that
  inventories valid archive jobs and marks non-destructive prune candidates.
- Added an admin-gated Studio audit quarantine archive validation endpoint for
  path-free import and restore preflight checks.
- Added an admin-gated Studio audit quarantine archive job that persists
  quarantined retained rows as path-confined evidence artifacts.
- Added an admin-gated, path-free Studio audit quarantine export for legacy,
  corrupt, or chain-broken retained audit rows.
- Added path-free quarantine metadata to Studio audit status and export payloads
  so legacy, corrupt, or chain-broken retained audit rows are counted separately
  from verified retained evidence.
- Fail closed on non-positive Studio audit retained-file settings so configured
  rotation always keeps at least one archived JSONL segment for incident review
  and retained-chain verification.
- Added retained-chain integrity verification to persistent Studio JSONL audit
  logs. Audit status, audit export, and operator status now report
  path-free `integrity_verified`, `integrity_error`, and latest retained event
  hash fields.
- Authenticated `/ws/progress` when Studio route-policy enforcement is enabled.
  Non-browser clients can use bearer headers, while the React frontend carries
  browser-session tokens through the `studio-auth` WebSocket subprotocol.
- Sent the active Studio bearer token on frontend project-delete requests so
  production route-policy enforcement does not reject the saved-project delete
  workflow.

### Physics and mathematics hardening
- Added the polyglot `adc_to_spike_windows(...)` chain for the ADC-to-spike
  decimating rate-code encoder across python / rust / julia / go / mojo. Each
  decimation window is centred and quantised to a Q-format code, sign-aware
  averaged (round half away-from-zero then truncate toward zero) and converted
  into a deterministic spike count (`|window| // threshold`) with the window sign
  as polarity. The arithmetic is exact integer, so every backend reproduces the
  Python floor bit-for-bit, and a dedicated test pins the Python floor against the
  cycle-stepped golden model in `tools/adc_to_spike_reference.py`. Added the Rust
  engine `adc_to_spike_windows` + PyO3 `py_adc_to_spike_windows`, the wired
  `sc_neurocore.sensors.adc_to_spike_kernel` primary with fastest-first dispatch,
  the Julia/Go/Mojo backends, unit, cross-backend and golden-reference parity
  tests to full coverage of the primary module, a five-backend benchmark with a
  committed artefact, and documentation.
- Added the polyglot batched `mixed_dense_forward_batch(...)` chain for the
  mixed-precision Q8.8 × Q16.16 dense MAC across python / rust / julia / go / mojo.
  The integer branch (per-tensor scale folded so the accumulator divisor equals
  the Q8.8 weight scale) contracts in a signed 64-bit accumulator, divides by an
  arithmetic right shift (floor division) and saturates to the Q16.16 code range
  with explicit overflow/underflow flags, so every backend reproduces the Python
  floor bit-for-bit. Added the Rust engine `mixed_dense_forward_batch_q88_q1616` +
  PyO3 `py_mixed_dense_forward_batch_q88_q1616`, the wired
  `sc_neurocore.compiler.mixed_dense_kernel` primary with fastest-first dispatch,
  the Julia/Go/Mojo backends, unit and cross-backend parity tests to full coverage
  of the primary module, a five-backend benchmark with a committed artefact (the
  NumPy and Julia matmul paths measured competitive with or faster than the scalar
  FFI backends), and documentation.
- Added the polyglot batched `dcls_max_forward_batch(...)` chain for the DCLS-max
  Q8.8 triangular (tent) weighting kernel (Khalfaoui-Hassani, Pellegrini &
  Masquelier 2023) across python / rust / julia / go / mojo. The kernel is exact
  integer Q8.8 arithmetic with a Q16.16 saturating accumulator, so every backend —
  including Mojo — reproduces the Python floor bit-for-bit with a parity tolerance
  of exactly zero. Added the Rust engine `dcls_max_forward_batch_q88` + PyO3
  `py_dcls_max_forward_batch_q88`, the wired `sc_neurocore.scpn.dcls_tent_kernel`
  primary with per-channel learnable centre/sigma and fastest-first dispatch, the
  Julia/Go/Mojo backends, unit and cross-backend parity tests to full coverage of
  the primary module, a five-backend benchmark with a committed artefact, and a
  documentation upgrade.
- Replaced `ExpIFNeuron` raw Euler mutation with candidate-first RK4 across the
  maintained Python reference, Rust engine, Go service, Julia mirror, and Mojo
  mirror. The Fourcaud-Trocmé EIF ODE and hard reset are unchanged; all surfaces
  now reject non-finite RK4 derivatives/candidates before mutation. Added focused
  Python/Rust/Go RK4 tests, a Go benchmark hook, a local non-isolated Python RK4
  regression artifact, and refreshed the public model documentation.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `McKeanNeuron` (McKean 1970 piecewise-linear FitzHugh-Nagumo caricature) across
  python / rust / julia / go / mojo. The piecewise-linear RK4 right-hand side is
  exact arithmetic, so Rust, Julia and Go reproduce the NumPy reference
  bit-for-bit; the Mojo backend is ULP-bounded and non-amplifying (a
  two-dimensional autonomous flow cannot be chaotic). Added the Rust engine
  `simulate` plus PyO3 `py_mckean_simulate`, the Julia/Go/Mojo backends,
  cross-backend parity tests, a multi-language benchmark with a committed results
  artefact, and a model-documentation upgrade; replaced the decorative
  `accel/go/services` stub with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `WilsonHRNeuron` (Wilson 1999 polynomial cortical model) across
  python / rust / julia / go / mojo. The polynomial RK4 right-hand side with a
  hard voltage reset is exact arithmetic, so Rust, Julia and Go reproduce the
  NumPy reference bit-for-bit; the Mojo backend is ULP-bounded and non-amplifying
  (the per-spike reset re-anchors the 2D autonomous flow). Added the Rust engine
  `simulate` plus PyO3 `py_wilson_hr_simulate`, the Julia/Go/Mojo backends,
  cross-backend parity tests, a multi-language benchmark with a committed results
  artefact, and a model-documentation upgrade; replaced the decorative
  `accel/go/services` stub with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `PernarowskiNeuron` (Pernarowski 1994 pancreatic beta-cell burster) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` (from
  `v**3`) so it is bit-identical to the engine's `v.powi(3)` and removed the now
  unreachable `OverflowError` branch; Rust, Julia and Go then reproduce the NumPy
  RK4 reference bit-for-bit, and the Mojo backend is ULP-bounded and
  non-amplifying. Added the Rust engine `simulate` plus PyO3
  `py_pernarowski_simulate`, the Julia/Go/Mojo backends, cross-backend parity
  tests, a multi-language benchmark with a committed results artefact, and a
  model-documentation upgrade; replaced the decorative `accel/go/services` stub
  with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `TermanWangOscillator` (Terman-Wang 1995 LEGION relaxation oscillator) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` (from
  `v**3`) so it matches the engine's `v.powi(3)` and removed the now-unreachable
  `OverflowError` branch. The right-hand side mixes the exact cubic with a `tanh`
  gating term: the Rust engine resolves `tanh` to the same glibc symbol as Python
  and is bit-identical, while Julia/Go/Mojo use their own libm `tanh` and are
  ULP-bounded (the 2D relaxation oscillator is non-chaotic, so it does not
  amplify). Added the Rust engine `simulate` plus PyO3 `py_terman_wang_simulate`,
  the Julia/Go/Mojo backends, cross-backend parity tests, a multi-language
  benchmark with a committed results artefact, and a model-documentation upgrade;
  replaced the decorative `accel/go/services` and `accel/mojo/kernels` stubs with
  real backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `MihalasNieburNeuron` (Mihalas-Niebur 2009 generalised integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state `(v, theta, i1, i2)`
  right-hand side is purely linear — no transcendental functions — advanced by
  candidate-first RK4 with a discontinuous spike reset, so the Rust engine, Julia
  and Go backends reproduce the NumPy reference bit-for-bit (trace, spike count
  and final state); the Mojo backend fuses multiply-add and is validated as
  non-amplifying within a ULP band with identical spike counts. Added the Rust
  engine `simulate` plus PyO3 `py_mihalas_niebur_simulate`, the Julia/Go/Mojo
  backends, cross-backend parity tests, a multi-language benchmark with a
  committed results artefact, and a model-documentation upgrade; replaced the
  decorative `accel/go/services` and `accel/mojo/kernels` stubs with real
  backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `GLIFNeuron` (Allen Institute GLIF5 generalised leaky integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state
  `(v, theta, i_asc1, i_asc2)` right-hand side is purely linear — no
  transcendental functions — advanced by candidate-first RK4 with an additive
  threshold spike reset, so the Rust engine, Julia and Go backends reproduce the
  NumPy reference bit-for-bit (trace, spike count and final state); the Mojo
  backend fuses multiply-add and is validated as non-amplifying within a ULP band
  with identical spike counts. Added the Rust engine `simulate` plus PyO3
  `py_glif_simulate`, the Julia/Go/Mojo backends, cross-backend parity tests, a
  multi-language benchmark with a committed results artefact, and a
  model-documentation upgrade; replaced the decorative `accel/go/services` and
  `accel/mojo/kernels` stubs with real backends.

### Studio
- Added the admin `POST /api/studio/training/weight-restore/attach/live` endpoint
  and the confined control channel that backs it. The endpoint delivers the
  verified weights of a completed source job to a running target training job;
  the worker polls a reserved control directory at each epoch boundary and applies
  the attach with a strict `load_state_dict` that records a
  `studio.training.weight-restore-attach.v1` (`mode: live`) evidence artifact. An
  incompatible or malformed attach is rejected with an `attach_rejected` metric
  event and never interrupts the running job. Added the control channel
  (`StudioJobManager.send_control_command` with atomic command publication +
  `StudioJobContext.poll_control_command`/`read_control_seed`, reserved
  `.studio_control` and `.studio_control_seed` directories), the epoch-boundary
  poll in the training loop, an architecture-fingerprint pre-check, the route
  policy, the `studio.training.weight_restore.attach_live` audit action, the
  preflight required-route entry, a Training Monitor live-attach action with a
  path-free request strip, frontend client types, and full backend and frontend
  tests plus documentation.
- Added the admin `POST /api/studio/training/weight-restore/attach` endpoint and
  the confined seed-input channel that backs it. The endpoint rebuilds the
  canonical restore plan from a completed training job's checkpoint, delivers the
  integrity-checked weight artifacts to a bounded `studio-training-restore` worker
  as confined seed inputs, and warm-starts a training job that loads the verified
  weights at the epoch-zero checkpoint boundary before training forward. A strict
  `load_state_dict` fails closed on an architecture mismatch before training
  begins. Added an architecture fingerprint that gates compatibility on the
  shape-determining config fields only, the
  `studio.training.weight-restore-attach.v1` evidence contract, the route policy,
  the `studio.training.weight_restore.attach` audit action, the preflight
  required-route entry, a `weight_restore_attach_results` evidence-bundle field
  stored under `evidence/training-weight-restore-attaches/`, a Training Monitor
  warm-start action with a path-free evidence strip, frontend client types, and
  full backend and frontend tests plus documentation.
- Added the admin `POST /api/studio/training/weight-restore` endpoint. It
  rebuilds the canonical restore plan from a completed training job's stored
  checkpoint metadata, fetches the integrity-checked weight and metadata
  artifacts, and materializes the weights inside a bounded
  `studio-training-restore` worker job using a `weights_only=True` trusted
  state-dictionary loader. The worker writes a path-free
  `studio.training.weight-restore.v1` evidence artifact holding only verified
  digests, parameter count, and loaded-key total; the deserialized tensors never
  reach the API response. Added the route policy, the
  `studio.training.weight_restore.materialize` audit action, the preflight
  required-route entry, a `weight_restore_results` evidence-bundle field stored
  under `evidence/training-weight-restores/`, a Training Monitor materialize
  action with a path-free evidence strip, frontend client types, and full
  backend and frontend tests plus documentation.

## [3.15.34] - 2026-06-15

### Physics and mathematics hardening
- Corrected `CourageNekorkinMapNeuron` to the canonical Courbage-Nekorkin-Vdovin
  2007 map (`Chaos` 17:043109): `x + F(x) - y - beta*H(x - d)` with the
  piecewise-linear field, Heaviside discontinuity at `x = d`, and `B^+`
  invariant-region default parameters, replacing the prior non-canonical form.
  Added the polyglot N-step `simulate` chain (python/rust/julia/go/mojo;
  Rust/Julia/Go bit-exact, Mojo ULP-bounded) with parity tests, a multi-language
  benchmark, and a rewritten model documentation page.
- Corrected `CourageNekorkinMapNeuron` to the canonical Courbage-Nekorkin-Vdovin
  2007 map (`Chaos` 17:043109): `x̄ = x + F(x) − y − β·H(x − d) + I`,
  `ȳ = y + ε(x − J)` with the piecewise-linear field `F`, the Heaviside
  discontinuity at `x = d`, and the breakpoints `J_min`, `J_max`. Replaced the
  prior implementation, which was missing the leading `x` term, used an incorrect
  saturating field, had no Heaviside term, an opposite-sign recovery update, and a
  `±1e6` clip that masked divergence. Default parameters set to the published
  chaotic spiking-bursting regime (`m0=0.0864, m1=0.65, a=0.2, d=0.235, J=0.2,
  beta=0.085, eps=0.02`).
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  the map across python / rust / julia / go / mojo. Rust, Julia and Go reproduce
  the NumPy reference bit-for-bit; the Mojo backend is per-step ULP-bounded with a
  matching spike-count band. `auto` selects Rust.
- Added the Rust engine struct plus PyO3 `py_courage_nekorkin_map_simulate`, the
  fail-closed Rust safety mirror, the Julia/Go/Mojo backends, cross-backend parity
  tests, a multi-language benchmark with a committed results artefact, and a
  rewritten model documentation page.

### Dependencies
- Migrated z3 `0.12` -> `0.20` (feature `static-link-z3` renamed to `bundled`;
  the bounded-model verifier updated for the lifetime-free 0.20 AST/Solver API).
- Bumped esbuild/vite/@vitejs/plugin-react (Studio frontend, clears the open
  esbuild advisory), github/codeql-action, click, and hypothesis.

### CI and release reconciliation
- Reconciled `main` with the previously orphaned release tags `v3.15.26`–
  `v3.15.33` so the published release lineage is continuous again.
- Regenerated the capability manifest, corrected the mixed-precision emitter
  Q-format label, and built the ARM64 wheel against the exact matrix interpreter.

### Security and Rust engine
- Migrated the PyO3/numpy Rust extension chain from `0.28` to `0.29`
  across the engine, fuzz harness, evo substrate, stochastic doctor, and spike
  stats crates; refreshed lockfiles and added the missing spike-stats lockfile
  for reproducible advisory scanning.
- Updated the GPU feature path for WGPU 29 API changes exposed by the
  all-features engine check.

## [3.15.33] - 2026-06-05

### CI and benchmark evidence
- Replaced the Wilson-Cowan CI throughput floor with a bounded-runtime
  regression sentinel and documented that production throughput evidence must
  come from isolated benchmark runs, not hosted coverage jobs.

## [3.15.32] - 2026-06-05

### CI and release workflows
- Restored the direct `MixedPrecisionSpec.get()` contract to preserve explicit
  `PrecisionConfig(16, 8)` widths while keeping explicit `Q7.8` preset parsing
  sign-inclusive.

## [3.15.31] - 2026-06-05

### CI and release workflows
- Aligned the explicit `Q7.8` mixed-precision preset contract with the
  sign-inclusive Q-label parser so branch CI no longer treats explicit
  Q-format labels as named `q88` aliases.

## [3.15.30] - 2026-06-05

### CI and release workflows
- Installed the pinned `click` runtime dependency before the pinned
  SymbiYosys executable check so hosted HDL/formal CI validates `sby`
  immediately after installation.

## [3.15.29] - 2026-06-05

### CI and release workflows
- Aligned branch CI version-contract tests with source metadata, installed the
  HDL/formal toolchain required by RTL contract tests, and synchronized the
  conda install profile with the release version.

## [3.15.28] - 2026-06-05

### CI and release workflows
- Added an executable entry point to the benchmark context example so
  `cargo test --manifest-path engine/Cargo.toml` builds all examples on Linux.

## [3.15.27] - 2026-06-05

### CI and release workflows
- Removed Yosys workflow-file edits from the Yosys push path filter so
  release-hygiene workflow changes do not self-trigger hosted-runner synthesis.

## [3.15.26] - 2026-06-05

### CI and release workflows
- Skipped Yosys synthesis on release tag pushes so tag releases do not fail on
  non-HDL changes after all modules time out under hosted-runner synthesis
  budgets; branch, pull-request, and manual synthesis workflows remain active.

## [3.15.25] - 2026-06-05

### CI and release workflows
- Moved the macOS static-Z3 C++ parser configuration before the v3-engine
  maturin dependency build so Apple Clang uses delayed template parsing during
  the actual engine install step.

## [3.15.24] - 2026-06-05

### CI and release hygiene
- Formatted the release-surface parity test before publishing the next
  immutable release tag.

## [3.15.23] - 2026-06-05

### Release workflows
- Aligned the Rust engine crate and bridge wheel metadata with the public
  Python package version before registry publication.
- Added release-surface tests for engine crate and bridge metadata version
  parity.

## [3.15.22] - 2026-06-05

### Security workflows
- Made the lightweight actionlint scanner deterministic by disabling its
  external ShellCheck and Pyflakes integrations; those analyzers remain separate
  CI concerns instead of hidden actionlint dependencies.

## [3.15.21] - 2026-06-05

### Security workflows
- Installed ShellCheck in the CI security scanner job so actionlint has the same shell-analysis dependency available as local validation.

## [3.15.20] - 2026-06-05

### Security workflows
- Updated the CI security scanner actionlint toolchain to `v1.7.12`, matching the locally validated workflow parser used for release gating.

## [3.15.19] - 2026-06-05

### Release workflows
- Added macOS-only static-Z3 C++ parser flags for engine wheel and v3 engine builds so Apple Clang handles Z3's template-heavy LP sources.

## [3.15.18] - 2026-06-05

### CI and source hygiene
- Scoped the SPDX guard away from vendored Mojo `.pixi` environments and generated OpenROAD build artefacts.
- Added minimal single-line SPDX markers to real HDL and module-specific test surfaces covered by the guard.
- Formatted the mixed-precision/live-control surfaces and tightened mixed-precision manifest typing so CI mypy passes without changing runtime contracts.

## [3.15.17] - 2026-06-05

### Release workflows
- Added the static-Z3 CMake policy floor to all Rust engine wheel builders.
- Installed Docker build-stage `clang` and `libclang-dev` so Z3 bindings can
  locate libclang during containerized release builds.
- Replaced the Docker build step id used in SARIF gating with an expression-safe
  identifier.

## [3.15.16] - 2026-06-05

### Release workflows
- Fixed Docker Trivy scans to use the metadata-selected image tag as an
  explicit image reference instead of an empty default scan target.

## [3.15.10] - 2026-06-05

### Release workflows
- Fixed Docker workflow image-tag selection so the workflow remains valid on
  tag pushes and scans the selected pushed image reference.

## [3.15.9] - 2026-06-05

### Security and release workflows
- Patched the hub runtime Starlette pin from `1.0.0` to `1.0.1`.
- Fixed Docker image scanning to scan the selected pushed image reference.
- Switched the Rust engine Z3 dependency to a static build path so release
  wheels no longer depend on runner-provided Z3 headers.

## [3.15.8] - 2026-06-05

### Documentation and release polish
- Bumped Python, Rust engine, bridge package, Sphinx docs, README, and
  generated capability metadata to version `3.15.8`.
- Expanded the documentation home page with an evaluator map that routes new
  users, hardware teams, framework reviewers, industrial evaluators, notebook
  readers, and API consumers to the correct first evidence surface.
- Strengthened onboarding, notebook, API, FPGA tutorial, industrial
  applications, product overview, and applications/market documentation so
  users can understand what SC-NeuroCore is for, where it has evidence, where
  optional dependencies apply, and which claims require committed artefacts.
- Expanded the public documentation surface with an evaluator map covering
  onboarding, notebooks, tutorials, API docs, benchmarks, hardware, industrial
  applications, commercial evaluation, and evidence boundaries.
- Kept benchmark, hardware, clinical, regulatory, and market language tied to
  committed artefacts or explicit missing-evidence boundaries.

### Engine supervisor
- Added a public Rust supervisor execution entrypoint shared by the PyO3
  controller path, preserving bounded-run completion by dropping snapshot
  senders before joining the Z3 worker and adding module-specific supervisor
  tests for safe bounded execution, unsafe Petri-net rejection, worker shutdown
  signalling, and zero-neuron fail-closed validation.

### Compiler precision
- Hardened adaptive runtime precision manifests for BFP/Q16.16 handoff by
  adding the `adaptive_precision_emitter.v1` contract, emitted datapath
  width/fraction, exponent-stream width, exponent-vector width, and fail-closed
  rejection of block-exponent parameter counts on fixed Q-format paths.
- Hardened generated AXI4-Lite/PCIe live-control readback so invalid
  bank/entry selections return a bus error and latch the sticky
  `invalid_selection` trap instead of silently returning zero.
- Added emitter-facing mixed-precision manifests for fixed Q16.16 and
  block-floating variables, including deterministic assignment order, emitted
  datapath width/fraction, exponent stream width, exponent-vector width, and
  fail-closed BFP parameter-count validation for downstream HDL emitters.
- Routed quantizer precision-envelope proof fields through the static-analysis
  Q-format envelope proof API, with module-specific regression coverage to keep
  dense deployment manifests aligned with the standalone proof contract.
- Added a static-analysis Q-format envelope proof API for conservative Q-code
  bounds, with fail-closed validation, signed Q16.16 width/headroom manifests,
  and module-specific tests for safe, saturating, and block-floating
  exponent-edge contracts.
- Added signed fixed-point width proofs to mixed Q8.8/Q16.16 and
  block-floating precision envelope reports across Python, Rust, and refreshed
  comparison benchmark artefacts, including required total bits, required
  Q16.16 integer bits, headroom, saturation requirement, and static overflow
  proof status.
- Added seeded block-floating exponent-edge parity and trap contracts across
  the Python quantizer, Rust qformat mirror, and comparison benchmark
  artefacts: `BFP16E3X2` safe min/max exponent sweeps now match exact Q16.16
  output codes across languages, while max-exponent saturation records a
  deterministic overflow trap instead of silent wraparound.
- Added explicit block-exponent layout metadata for block-floating precision
  across adaptive manifests, mixed-precision specs, Python dense BFP manifests,
  and the Rust qformat mirror, with exponent-count validation before emission
  or accumulation.
- Added sub-LSB underflow telemetry to mixed Q8.8/Q16.16 and block-floating
  dense precision trap/envelope reports across Python and Rust, with refreshed
  process-affinity benchmark artefacts documenting matched overflow and
  underflow probes.
- Added sticky live-control partial-write traps so generated AXI4-Lite/PCIe
  parameter banks reject partial `WSTRB` updates before control or staged-data
  registers can be modified.
- Hardened live-control trap clearing so generated AXI4-Lite/PCIe parameter
  banks clear only selected sticky trap bits and preserve unrelated latched
  fault evidence.
- Added selected-trap clear helpers to the live-control schema and generated
  Python/C host drivers.
- Added deterministic live-control active-parameter readback for host update
  sequences and generated AXI4-Lite/PCIe parameter banks, including low/high
  committed-word registers and module-specific RTL simulation coverage.
- Hardened generated Python and C host drivers for live-control parameter
  banks with CRC32 update helpers, committed readback verification, trap-status
  checks, and mandatory high-word staging for narrow updates.
- Added generated Python and C live-control rollback, status-read, and
  trap-status-read helpers so host drivers expose the load/apply/rollback/
  clear/readback handshake.
- Added generated C live-control driver compile validation against a C11
  consumer that calls the committed update/readback verification helper.
- Latched live-control shadow bank and entry identity at load time so generated
  AXI4-Lite/PCIe apply and rollback operations cannot be retargeted by later
  selection-register writes.
- Added sticky live-control read-only-bank traps so generated AXI4-Lite/PCIe
  parameter banks reject direct MMIO writes to calibration/read-only banks before
  shadow loading or active coefficient mutation.
- Added sticky live-control invalid-selection traps so generated AXI4-Lite/PCIe
  parameter banks reject non-existent bank/entry writes without raising a false
  shadow-loaded acknowledgement.
- Added sticky CRC32 checksum-mismatch traps and a testbench-visible mismatch
  pulse to the generated AXI4-Lite/PCIe live-control parameter-bank surfaces,
  with module-specific simulation tests and refreshed benchmark-gate evidence.
- Replaced live-control update guards with an IEEE CRC32 register-window
  guard shared by the compiler schema and generated SystemVerilog, with stale
  guard rejection tests and refreshed benchmark evidence.
- Added the PCIe-MMIO live-control register-window adapter over the staged
  parameter-bank core, module-specific PCIe commit simulation, process-affinity
  AXI4-Lite/PCIe comparison benchmark evidence, and compiler API documentation
  for the exact bus-contract boundary.
- Added the UltraScale+ dense-folding contract: shared Rust/Python fold planner,
  folded Q8.8/Q16.16 HDL core, target-emitter fold metadata, module-specific
  simulation tests, and isolated Python/Rust benchmark evidence for fitting the
  64x32 dense contract into the ZU3EG DSP budget.
- Added the NEU-C.1 Zynq UltraScale+ target contract: Rust target metadata,
  conservative resource-budget reporting, deterministic Vivado Tcl generation,
  board-safe timing-only XDC baselines, module-specific tests, and isolated
  Python/Rust comparison benchmark evidence.
- Added the NEU-C.6 DCLS Q8.8 RTL path: bit-true Rust DCLS tent-kernel
  arithmetic, SystemVerilog axonal delay/tent/layer modules, IR
  `DclsLayer` emission, SymbiYosys safety/liveness harness, Python/PyTorch
  cosimulation, module-specific tests, and isolated benchmark evidence.
- Added NEU-C.5 ADC-to-spike quantiser HDL with Q-format decimation, deterministic AER rate coding, formal transfer properties, bit-true Python reference, isolated benchmark evidence, and hardware documentation.
- Added NEU-C.2 timing-aware formal-property framework with reusable SystemVerilog monitors, Python proof orchestration, nuXmv/Kind 2 emitters, a dense-layer SymbiYosys/cvc5 proof, and isolated benchmark evidence.
- Added the NEU-C.4 AER strict-priority queue and router backpressure path,
  including sticky drop/deadline traps, Python reference contract,
  SystemVerilog simulation, formal harness, benchmark gate, and hardware docs.
- Added live-control update and trap evidence benchmarks covering generated
  MMIO update sequences, static RTL regeneration, and staged range-trap
  simulation, with the artefact registered in the benchmark gate manifest.
- Added generated live-parameter-bank staged overflow and underflow traps that
  latch malformed MMIO payloads and block shadow-bank mutation before active
  coefficient application.
- Hardened compiler live-control update semantics with checksum-gated shadow
  loads, explicit apply/rollback sequences, active-only generated parameter
  outputs, and status telemetry for shadow-loaded, applied, rollback, and
  checksum-valid states.
- Added AXI4-Lite live-parameter-bank RTL emission from the compiler
  live-control schema, including BRAM/distributed RAM style hints, flattened
  parameter outputs, staged commits, trap status, and module-specific compile
  tests.
- Added deterministic compiler live-control schemas for AXI4-Lite/PCIe
  parameter-bank updates, including encoded-word range checks, fixed
  control/status registers, atomic staged commit sequences, and trap-clear
  command generation.
- Aligned adaptive block-floating precision metadata with the quantizer
  exponent-bias contract and added explicit block exponent alignment telemetry
  for `BFP16E3X32` to `Q16.16` adaptive-precision manifests.
- Hardened the 2026-06-04 mixed, block-floating, precision-trap, and
  precision-envelope benchmark artefact writers so Python and Rust runs record
  taskset affinity, load before/after, CPU governor, and frequency context.
- Aligned the mixed dense Python and Rust benchmark workloads on the canonical
  raw Q8.8/Q16.16 physical contract (`QFormatMixed(scale_per_tensor=False)`),
  eliminating the stale per-tensor Python envelope mismatch and refreshing the
  cross-language benchmark documentation.
- Marked the 2026-06-04 local precision benchmark artefacts as captured under
  concurrent workstation load and documented the isolated-core requirement for
  future production throughput claims.
- Aligned the block-floating dense Python and Rust benchmark workloads so the
  safe and saturating precision-envelope bounds compare the same physical BFP
  mantissa/exponent contract across languages.
- Added per-output conservative absolute-bound telemetry to the mixed
  Q8.8/Q16.16 and block-floating dense RTL (`abs_bounds_q1616`), aligned the
  Python/Rust benchmark artefacts with the same precision-envelope fields, and
  refreshed module-specific HDL tests plus HDL/Python/Rust benchmark evidence.
- Added per-output overflow telemetry to the mixed Q8.8/Q16.16 dense RTL and
  refreshed the Python, Rust, HDL, and documentation evidence for lane-level
  saturation attribution.
- Added per-output overflow telemetry to the block-floating dense RTL and
  refreshed the Python, Rust, HDL, and documentation evidence for lane-level
  saturation attribution.
- Added precision envelope reports across the mixed fixed-point and
  block-floating dense deployment paths, including conservative absolute-bound
  checks in Python and Rust, a synchronous HDL envelope guard, module-specific
  tests, and committed Python, Rust, and Yosys benchmark artefacts.
- Added precision trap reports across the mixed fixed-point and block-floating
  dense deployment paths, including exact overflow counts in the Rust qformat
  mirror, a synchronous HDL trap latch, module-specific tests, and committed
  Python, Rust, and Yosys benchmark artefacts.
- Added dense block-floating `BFP16E3X32` execution across the Python quantiser
  API, Rust IR qformat mirror, and synchronous HDL reference module, including
  shared-exponent product scaling, Q16.16 output saturation, overflow telemetry,
  module-specific tests, and committed Python, Rust, and Yosys benchmark
  artefacts.
- Corrected block-floating metadata so the maximum unbiased exponent reflects
  every encoded biased exponent code.
- Added the compiled mixed-dense Q8.8/Q16.16 contract across the Python
  quantiser API, Rust IR qformat mirror, and synchronous HDL reference module,
  including exact signed MAC scaling, accumulator saturation, and overflow
  telemetry.
- Added module-specific mixed-dense quantiser and HDL tests plus committed
  Python, Rust, and Yosys benchmark artefacts for the 64×32 mixed-precision
  dense contract.
- Added the `QFormatMixed` quantiser contract for Q8.8 stored weights with
  Q16.16 accumulator metadata, including per-tensor scale round-trip support,
  public compiler exports, module-specific quantiser tests, and refreshed
  compiler precision documentation.
- Corrected block-floating alias normalisation and shared-exponent selection
  so sub-unit tensors retain the finest representable scale within the exponent
  range.

### Typing hygiene
- Removed active source-level file-wide mypy suppressions from the package tree
  and repaired the exposed strict-mypy defects in ASIC flow, BCI Studio,
  bioware, digital-twin synchronisation, evolutionary substrate,
  explainability, federated learning, hypervisor, memristor, model-zoo, and
  spintronic surfaces.
- Confirmed strict package mypy passes for 940 source files with an isolated
  cache path; the repository-local cache path currently exhibits local
  filesystem `ENOSPC` behaviour and should not be used as typing evidence.

## [3.15.7] - 2026-06-01

### Engine publishing credentials
- Added the dedicated `sc-neurocore-engine` PyPI project token to the engine
  wheel publish step after PyPI rejected OIDC tokens scoped to the primary
  `sc-neurocore` project.
- Issued this patch release without rewriting prior tags so the partial
  publication attempts remain auditable and superseded.

## [3.15.6] - 2026-06-01

### Publishing hygiene
- Aligned the engine wheel PyPI publication job with the repository's existing
  trusted-publishing environment after PyPI rejected the dedicated
  `pypi-engine` environment claim.
- Issued this patch release without rewriting earlier tags so the failed
  trusted-publisher attempt remains traceable and superseded.

## [3.15.5] - 2026-06-01

### Engine package metadata
- Declared the engine wheel runtime NumPy dependency in the bridge package
  metadata so installed wheels resolve the dependency needed by the public
  `sc_neurocore_engine.layers` module.
- Issued this patch release after `v3.15.4` validated Python and crate
  publication but exposed the missing engine-wheel runtime dependency during
  smoke testing.

## [3.15.4] - 2026-06-01

### Publish automation
- Bumped the Rust engine crate release metadata alongside the Python package
  release surfaces so crates.io publication no longer attempts to republish an
  older engine version.
- Changed the engine wheel smoke test to install the built wheel before import,
  preserving runtime dependency resolution instead of unpacking the archive
  directly.

## [3.15.3] - 2026-06-01

### Release automation
- Fixed the tag-release workflow to extract release notes from the committed
  documentation changelog path.
- Issued this patch release candidate without rewriting the existing
  `v3.15.1` or `v3.15.2` tags.

## [3.15.2] - 2026-06-01

### Release integrity
- Issued a patch release candidate after the `v3.15.1` tag to preserve the
  no-history-rewrite rule while keeping the release tag aligned with the
  current CI, documentation, version metadata, and typed RK4 candidate-state
  fixes.
- Confirmed the strict mypy preparation lane remains error-free for the
  package source tree without adding suppressions.

## [3.15.1] - 2026-06-01

### Documentation and release polish
- Added public Product Overview and Applications and Market pages so new users, evaluators, and commercial readers can understand the project scope, evidence boundary, potential applications, and market position without reverse-engineering the API inventory.
- Refreshed the README, documentation home page, learning path, getting-started guide, notebook guide, API index, industrial-applications page, benchmark index, and cross-framework benchmark evidence page for clearer onboarding and claim traceability.
- Added a notebooks README with recommended reading order and reproducibility rules.
- Bumped Python package, public docs, capability metadata, and Rust engine package version references from 3.15.0 to 3.15.1.

### NIR Bridge
- Roundtrip tests for all 18/18 NIR primitives (was 7/18)
- Auto-broadcast scalar neuron params to input size (Norse/snnTorch export 0-dim tensors)
- Threshold fix: `>=` to `>` matching NIR spec and snnTorch behavior
- `reset_mode="subtract"` for snnTorch compatibility (subtract-reset vs zero-reset)
- IF subtract-reset test and unknown `reset_mode` fallback handling
- Cross-framework interop tests: Sinabs LIF/IAF/ExpLeak, Rockpool LIF/CubaLIF/LI, snnTorch RSynaptic subgraph
- Cross-framework r-encoding test documenting per-framework dt conventions
- SpikingJelly NIR roundtrip demo (`examples/spikingjelly_nir_roundtrip.py`)
- Norse NIR roundtrip demo with real Norse weights (`examples/norse_nir_roundtrip.py`)
- NIR roundtrip demo: stronger input to produce visible spikes
- Documentation: added SpikingJelly, Rockpool, Sinabs, snnTorch RSynaptic sections to `docs/guides/nir_integration.md`
- Documentation: framework dt/r quick reference table
- Documented Norse tau observation (export/import roundtrip discrepancy in Norse code)
- Removed unverified "first FPGA backend" claim from 6 files

### Physics and mathematics hardening
- Promoted `LeakyCompeteFireNeuron` from raw Euler vector updates to exact
  first-order relaxation across the Python reference, Go service, Julia
  mirror, Mojo scalar helpers, and Rust safety surface; module-owned tests now
  cover closed-form WTA parity, large-timestep boundedness, fail-closed state
  preservation, vector mirror contracts, and refreshed Python benchmark
  evidence.
- Promoted `AlphaNeuron` from a single-pole synaptic filter to the full
  two-state Rall/Gerstner alpha-cascade flow across the Python reference, Go
  service, Julia mirror, and Rust safety surface; module-owned tests now cover
  closed-form alpha parity, equal-time-constant limits, large-timestep
  boundedness, fail-closed state preservation, and refreshed Python benchmark
  evidence.
- Promoted `ResonateAndFireNeuron` from raw Euler oscillator increments to the
  exact constant-input linear resonator flow across the Python reference, Go
  service, Julia mirror, Mojo scalar helpers, and Rust safety surface; module
  tests now cover matrix-exponential parity, large-timestep damping,
  fail-closed state preservation, and refreshed Python benchmark evidence.
- Promoted `ArcaneNeuron` to candidate-first exact first-order relaxation across
  the fast, working-memory, and deep identity compartments on the Python
  reference, Go service, Julia mirror, Mojo scalar helpers, and Rust safety
  surface; module-owned tests now cover exact trajectory parity, large-timestep
  boundedness, fail-closed state preservation, and refreshed Python benchmark
  evidence.
- Hardened `ParametricLIFNeuron` candidate-first discrete recurrence semantics across the Python reference, Go service, Julia mirror, and Rust safety surface, preserving the Fang et al. PLIF update while rejecting corrupted runtime state and non-finite voltage candidates before mutation with refreshed benchmark evidence.
- Promoted `SigmoidRateNeuron` from raw Euler rate updates to exact first-order relaxation across the Python reference, Go service, Julia mirror, Mojo kernel, and Rust safety surface, with module-specific tests for closed-form parity, large-timestep boundedness, invalid-state preservation, and refreshed benchmark evidence.
- Promoted `NonResettingLIFNeuron` from raw Euler membrane and adaptive-threshold updates to exact first-order relaxation across the Python reference, Go service, Julia mirror, and Rust safety surface, with module-specific tests for closed-form parity, large-timestep boundedness, invalid-update preservation, and refreshed benchmark evidence.
- Promoted `AdaptiveThresholdIFNeuron` from guarded Euler mutation to exact
  first-order relaxation across the Python reference, Go service, Julia mirror,
  Mojo spike kernel, and Rust safety surface; module-owned tests now cover
  exact trajectory parity, large-timestep boundedness, fail-closed state
  preservation, and refreshed Python benchmark evidence.
- Promoted `YamadaNeuron` to candidate-first RK4 integration across the Python
  reference, Go service, Julia mirror, Mojo spike kernel, and Rust safety
  surface; module-owned tests now cover RK4 parity, finite-stage validation,
  fail-closed state preservation, and refreshed Python benchmark evidence
  documents the RK4 runtime cost.
- Promoted `BendaHerzNeuron` to candidate-first RK4 adaptation integration
  with exponential hazard spike probability across the Python reference, Go
  service, Julia mirror, Mojo kernel notes, and Rust safety surface; module
  tests now cover RK4 parity, seeded stochastic reproducibility, fail-closed
  state preservation, and refreshed Python/Go benchmark evidence.
- Hardened `CochlearHairCell` across the Python reference, Rust engine, Go
  service, Julia mirror, and Rust safety surface by replacing the raw membrane
  Euler voltage increment with exact conductance-form relaxation, adding stable
  finite-domain Boltzmann activation, preserving state on invalid runtime
  inputs, adding module-owned tests, and recording a refreshed Python benchmark
  artefact.
- Promoted `ButeraRespiratoryNeuron` to bounded candidate-first RK4
  integration across the Python reference, Rust engine, Go service, Julia
  mirror, and Rust safety surface; module-owned tests now cover RK4 parity,
  high-current bounded stability, fail-closed invalid-state preservation, and
  refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Hardened `DirectionSelectiveRGC` across the Python reference, Rust engine, Go service, Julia mirror, and Rust safety surface by replacing raw Euler membrane drift with exact first-order relaxation, preserving state on invalid optical drive or corrupted runtime buffers, adding module-specific tests, and recording a refreshed Python benchmark artefact.
- Hardened `BoothRinzelNeuron` Python, Julia, Go, and Rust safety surfaces with finite-domain validation, fail-closed candidate updates, physical gate/calcium bounds, and module-owned regression tests.
- Promoted `ConnorStevensNeuron` to candidate-first RK4 integration across the Python reference, Rust engine, Julia mirror, Go service, Mojo parity notes, and Rust safety surface; module-owned tests now cover RK4 parity, finite-domain validation, fail-closed state preservation, and refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Promoted `TermanWangOscillator` to candidate-first RK4 integration across
  the Python reference, Rust engine, Julia mirror, Go mirror, Mojo kernel
  notes, and Rust safety surface; module-owned tests now cover the Python
  model at 100%, public docs state the finite-domain and continuous
  threshold-crossing contracts, and refreshed Python/Rust benchmark artefacts
  document the RK4 runtime cost.
- Promoted `PernarowskiNeuron` to candidate-first RK4 integration across the
  Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; module-owned tests now cover the Python model at 100%, public docs
  state the finite-domain and continuous threshold-crossing contracts, and
  refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Promoted `FitzHughRinzelNeuron` to candidate-first RK4 integration across
  the Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; module-owned tests now cover the Python model at 100%, public docs
  state the finite-domain and reset contracts, and refreshed Python/Rust
  benchmark artefacts document the RK4 runtime cost.
- Hardened `BertramPhantomBurster` across Python, Julia, Go, and Rust safety
  surfaces by replacing raw Euler state mutation with bounded RK4 integration
  over the published three-state ODE, adding finite physical-parameter and
  candidate-state validation, updating module-owned tests and model
  documentation, and adding refreshed local benchmark evidence.
- Replaced proxy `ollivier_ricci_curvature` evaluation with graph-metric
  lazy-random-walk Wasserstein transport, added fail-closed coupling graph and
  node-index validation, aligned topology tests, and documented the exact
  topological observable contract.
- Hardened `ExactLIFSolver` and the `solver.lif.subthreshold-exact`
  alternative route with finite physical-parameter validation, non-negative
  runtime-time contracts, reset/threshold ordering, fail-closed subthreshold
  route-domain checks, route documentation, and refreshed benchmark evidence.
- Hardened `physics.kuramoto.noiseless-symplectic-lift` route validation
  for empty phase arrays, phase/frequency shape mismatches, non-finite
  phases or frequencies, non-positive horizons and timesteps, boolean scalar
  parameters, and negative coupling; refreshed route documentation and
  benchmark evidence for the bounded noiseless Hamiltonian-lift lane.
- Hardened `StormerVerlet`, `LeapfrogSolver`, and the
  `physics.oscillator.harmonic-symplectic` route with fail-closed
  Hamiltonian state validation, finite time/timestep contracts, RHS
  shape/finite-output checks, zero-energy route rejection, updated
  alternative-path documentation, and refreshed oscillator benchmark
  evidence.
- Hardened `WolframHypergraph` with hyperedge, node-id, and rewrite-step
  validation; rewrites now revalidate graph invariants after each evolution
  pass, dimension estimation fails closed on corrupted topology, the Julia
  mirror was corrected, and physics docs now state the topology contract.
- Hardened `FeynmanKacHeatSolver` with finite-domain validation for
  length, diffusivity, walker count, timestep, seed, density grids, target
  time, and histogram bins; replaced bounded iterative boundary correction
  with exact triangle-wave reflection for Neumann Brownian paths; updated
  Julia, Mojo, and Rust heat mirrors plus physics docs and reran the
  `physics.heat.cosine-mode` shadow benchmark.
- Hardened `PinskyRinzelNeuron` Python, Julia, Go, and Rust safety
  surfaces to validate two-compartment state, compartment fraction,
  positive conductances, timestep, calcium non-negativity, gate bounds,
  and dual-input currents before integration; candidate updates now fail
  before mutation on non-finite state or gate-envelope excursions while
  preserving somatic threshold-crossing semantics.
- Hardened `LarterBreakspearNeuron` Python, Julia, Go, and Rust safety
  surfaces to revalidate conductance, ion-rate, timestep, coupling, and
  potassium-gate bounds before integration; RK4 candidates now fail
  before mutation on non-finite state or gate excursions while preserving
  continuous voltage output semantics.
- Hardened `WilsonCowanUnit` Python and Rust safety surfaces to
  revalidate E/I state, non-negative coupling weights, positive time
  constants, sigmoid gain, timestep, and candidate rate bounds before
  mutation; public model documentation now states the two-term sigmoid
  range and fail-closed polyglot runtime contract.
- Hardened `MorrisLecarNeuron` Python, Julia, Go, and Rust safety
  surfaces to validate finite conductance state, membrane capacitance,
  activation slopes, potassium activation bounds, timestep, threshold,
  and runtime drive before integration; candidate updates now fail
  before mutation on potassium-rate overflow or non-finite state while
  preserving no-reset threshold crossing semantics.
- Promoted `FitzHughNagumoNeuron` to RK4-by-default integration across the
  Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; the Python legacy Euler path is now explicit opt-in, fail-closed
  candidate validation is preserved before mutation, module-owned tests now
  cover 100% of the Python model, and refreshed Python/Rust benchmark
  artefacts document the RK4 runtime cost.
- Hardened `JansenRitUnit` Python, Julia, Go, and Rust safety surfaces to
  validate neural-mass state, excitatory/inhibitory gain and rate contracts,
  timestep and external-drive boundaries, overflow-stable sigmoid bounds, and
  finite candidate updates before mutation while preserving continuous EEG
  proxy output semantics.
- Hardened `WendlingNeuron` Python, Go, and Rust safety surfaces to validate
  neural-mass state, physiological gain/rate/timestep contracts, non-finite
  external drive, overflow-stable sigmoid bounds, and finite candidate updates
  before mutation while preserving continuous EEG-proxy output semantics.
- Hardened `CompteWMNeuron` Python, Julia, Go, and Rust safety surfaces
  to validate NMDA/AMPA/GABA gate state, Mg2+-block denominators,
  conductance and timescale contracts, non-finite drive, and bounded
  voltage or gate candidates before mutation while preserving
  spike-triggered self-inhibitory GABA feedback.
- Hardened `COBALIFNeuron` Python, Julia, Go, and Rust safety surfaces
  to validate mutable conductance state, membrane geometry, synaptic
  deltas, and exponential decay contracts before each update; compute
  voltage and conductance candidates before mutation; and reject
  non-finite or out-of-envelope candidates while preserving spike reset
  semantics.
- Hardened `ComplementaryLIFNeuron` Python, Julia, Go, and Rust safety
  surfaces to revalidate mutable dual-path state, threshold, timestep,
  and membrane timescale before each update; recompute the decay
  constant after runtime parameter mutation; and reject non-finite
  drive or membrane candidates before mutation while preserving ternary
  positive and negative spike semantics.
- Hardened `ChayKeizerNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid beta-cell gate/calcium state,
  non-physical Ca-dependent potassium and calcium-buffer contracts,
  unstable logistic/timescale exponentials, non-finite drive, and
  out-of-bounds membrane, gate, or calcium candidates before mutation.
- Hardened `ChayNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid beta-cell gate/calcium state, non-physical conductance
  and calcium-buffer contracts, unstable logistic exponentials,
  non-finite drive, and out-of-bounds membrane, gate, or calcium
  candidates before mutation while substepping the stiff potassium
  dynamics.
- Hardened `ChandelierNeuron` Python, Julia, Go, and Rust safety surfaces
  to reject invalid Kv1/Kv3 gate state, non-physical conductance and
  capacitance contracts, unstable rate and gate exponentials, non-finite
  drive, and out-of-bounds membrane or gate candidates before mutation while
  preserving axo-axonic Kv1 delay and Kv3 sharpening dynamics.
- Hardened `CerebellarBasketNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid A-type/KCa gate state, calcium state,
  non-physical conductance and capacitance contracts, unstable rate
  exponentials, non-finite drive, malformed calcium activation denominators,
  and out-of-bounds membrane or calcium candidates before mutation.
- Hardened `BKNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject invalid BK gate state, calcium state, non-physical conductance and
  capacitance contracts, malformed substep geometry, unstable rate and BK
  activation exponentials, non-finite drive, and out-of-bounds membrane or
  calcium candidates before mutation while preserving spike-triggered calcium
  influx.
- Hardened `ATypeKNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid transient IA gate state, non-physical conductance and
  capacitance contracts, malformed substep geometry, unstable rate
  exponentials, non-finite drive, and out-of-bounds membrane candidates before
  mutation while preserving A-type K first-spike-delay dynamics.
- Hardened `AstrocyteLIFNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid glial calcium state, non-positive membrane and
  calcium timescales, malformed threshold geometry, non-finite drive,
  gliotransmitter drift, and non-finite calcium or membrane candidates before
  mutation while preserving tripartite feedback semantics.
- Hardened `AlphaMotorNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid HH/PIC gate state, non-physical calcium buffers,
  non-positive timestep/capacitance/timescale contracts, unstable rate
  exponentials, and non-finite membrane/calcium candidates before mutation.
- Hardened `ErmentroutKopellMapNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid phase-map state, non-positive timestep,
  non-finite drive, non-finite phase candidates, and mirror threshold drift
  before mutation while preserving compact-circle phase wrapping.
- Hardened `AiharaMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid chaotic-map state, malformed feedback/damping parameters,
  non-finite drive, unstable sigmoid evaluation, and non-finite map candidates
  before mutation.
- Hardened `ChialvoMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid discrete-map state, non-finite drive, unstable exponential
  map terms, and non-finite two-dimensional map candidates before mutation.
- Hardened `RulkovMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid discrete-map state, non-positive map gain/timescale
  parameters, non-finite drive, non-finite branch boundaries, and non-finite
  map candidates before mutation.
- Hardened `BrunelWangNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid conductance/timescale/capacitance contracts, malformed
  synaptic gates, non-finite refractory or voltage state, unstable NMDA
  Mg2+-block exponentials, and non-finite membrane candidates before mutation.
- Promoted `WilsonHRNeuron` Python, Rust engine, Julia, Go, and Rust safety surfaces to candidate-first RK4 over the coupled polynomial cortical `(v, r)` state, with finite derivative/candidate guards, reset-preserving spike semantics, module-specific RK4 parity tests, and refreshed benchmark evidence.
- Hardened `WilsonHRNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid polynomial-cortical runtime state, non-positive recovery
  timescale or timestep, non-finite current, and non-finite voltage/recovery
  candidates before mutation while preserving spike-triggered voltage reset.
- Hardened `WongWangUnit` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject invalid two-pool gating state, non-positive timescales, non-finite
  stimuli or noise, unstable transfer-function exponentials, and non-finite
  candidate states before mutation while preserving tuple rate outputs.
- Promoted `WongWangUnit` Python, Rust engine, Julia, Go, Mojo, and Rust safety
  surfaces from forward Euler to candidate-first RK4 over the coupled
  two-pool decision ODE, preserving one sampled stochastic drive per pool per
  step and tuple rate outputs.
- Hardened `WilsonCowanUnit` Python, Julia, Go, and Rust safety surfaces to
  reject invalid rate-state, non-positive timescales, non-finite external
  drive, unstable sigmoid exponentials, and non-finite rate candidates before
  mutation while preserving rate-model return semantics.
- Hardened `TraubMilesNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid HH gate probabilities, non-physical
  conductances, non-finite rate constants, and non-finite ten-substep
  voltage candidates before state mutation.
- Hardened `TermanWangOscillator` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid relaxation-oscillator state, non-positive
  timescale parameters, non-finite drive, and non-finite cubic recovery
  updates before mutation.
- Hardened `WangBuzsakiNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state or non-finite fast-spiking
  conductance updates before state mutation.
- Hardened `PoissonNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  revalidate mutable rate and timestep state before sampling, reject non-finite
  interval hazards, and keep the finite-step Poisson probability bounded.
- Hardened `McCullochPittsNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce finite weighted-input and mutable-threshold contracts,
  preserve equality-at-threshold Heaviside semantics, and keep reset as a
  stateless no-op.
- Hardened `EscapeRateNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable point-process state before membrane integration,
  exponentiation, hazard evaluation, or random sampling; non-finite voltage
  candidates and escape hazards now fail before membrane mutation.
- Hardened `LapicqueNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable RC state before division/integration and report invalid
  current, corrupted state, or non-finite Euler increments explicitly before
  membrane mutation; documented the Mojo fail-closed spike-flag boundary.
- Hardened `NonResettingLIFNeuron` Python, Julia, Go, and Rust safety surfaces
  to revalidate runtime membrane and adaptive-threshold state before
  integration, compute both candidates before mutation, and report non-finite
  updates explicitly while preserving the no-voltage-reset spike contract.
- Hardened `PerfectIntegratorNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate runtime membrane geometry before division/integration
  and to report invalid or non-finite voltage increments explicitly before
  state mutation.
- Hardened `ThetaNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject corrupted runtime phase or timestep state before cosine/Euler
  evaluation and to report non-finite phase increments explicitly without
  mutating the compact-circle state.
- Hardened `SiegertTransferFunction` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate first-passage parameters at runtime, reject non-finite
  quadrature bounds, integrals, and inter-spike intervals, and keep rates
  finite, non-negative, and refractory bounded.
- Hardened `SigmoidRateNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce the continuous-rate `[0, 1]` invariant, reject unstable
  Euler ratios and corrupted runtime state before mutation, and use saturated
  finite-drive logistic evaluation for extreme inputs.
- Hardened `AdaptiveThresholdMoENeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite adaptive-threshold,
  quotient, or soft-reset candidates before state mutation, while preserving
  non-negative integer spike-count residual semantics.
- Hardened the `ThresholdLinearRateNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime rate state and non-finite rate
  outputs before state mutation.
- Hardened the `AdaptiveThresholdIFNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime state and non-finite Euler or
  threshold-jump updates before state mutation.
- Hardened the `BrainScaleSAdExNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite hardware-scaled
  integrator or adaptation updates before state mutation.
- Hardened the `AdExNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite integrator or adaptation
  updates before state mutation.
- Hardened the `ExpIFNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite Euler updates before membrane
  mutation.
- Hardened the Rust/PyO3 Kuramoto solver boundary to fail closed on non-finite frequencies, coupling matrices, initial/runtime phases, field pressure, SSGF geometry/PGBO matrices, invalid `dt`, and negative/non-finite noise amplitudes.
- Extended Kuramoto `run()` and `run_ssgf()` validation so invalid timesteps and non-finite SSGF gains/matrices are rejected even for zero-step dry runs.
- Added PyO3 SSGF shape guards so malformed `W` and `h_munu` matrices raise `ValueError` before entering the Rust solver.
- Corrected the `PINGCircuit` Python reference step to consume one excitatory and one inhibitory Wiener-noise vector per timestep, matching the Rust, Julia, Go, and Mojo backend stochastic contract; benchmark metadata now reports the selected backend directly.
- Hardened `CorticalColumn(backend="python", use_block_csr=True)` so it remains on the scipy.sparse reference path and does not call the Rust single-block fallback when native symbols are present.
- Hardened `HindmarshRoseNeuron` RK4/Euler derivative evaluation to fail closed on cubic overflow or non-finite intermediate stages without mutating state.
- Aligned `HindmarshRoseNeuron` Rust engine, Rust safety, Go, and Julia
  counterparts with the Python RK4 trajectory and fail-closed candidate-state
  contract.
- Hardened `MorrisLecarNeuron` Euler/RK4/Rosenbrock paths to fail closed on potassium-rate overflow or non-finite derivative/state updates without mutating state.
- Hardened `FitzHughNagumoNeuron` Euler/RK4/Rosenbrock paths to fail closed on cubic overflow or non-finite derivative/state updates without mutating state, and aligned the Julia, Go, and Rust safety counterparts with the documented no-reset state equation.
- Promoted `McKeanNeuron` Python, Rust engine, Julia, Go, and Rust safety surfaces from simultaneous Euler to candidate-first RK4 over the coupled `(v, w)` state, with finite derivative/candidate guards and module-specific RK4 parity tests.
- Hardened `McKeanNeuron` runtime updates across Python, Julia, Go, and Rust safety surfaces to fail closed on non-finite state/current or non-finite post-update state instead of silently reporting no spike.
- Hardened `ResonateAndFireNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler updates report explicit errors/sentinels instead of silently returning no spike.
- Hardened `QuadraticIFNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler increments report explicit errors/sentinels instead of silently returning no spike.

### Repository hygiene
- Purged obsolete completed failed/cancelled GitHub Actions repair-sequence runs after later successful replacement runs were verified on `main`.
- Removed inactive stale GitHub Pages deployment records while retaining the current successful Pages deployment and successful package-release deployment evidence.
- Rechecked Dependabot, code-scanning, and secret-scanning alert surfaces; all reported zero open alerts.

## [3.15.0] — 2026-05-19

### Compiler Intelligence, Platform Registry, and Deployment (2026-05-01)
### Wave 11: Hardware and compiler extensions (2026-05-01)

#### Added
- Expanded the hardware profile catalogue across FPGA, ASIC, neuromorphic, photonic, chiplet, PIM/CXL, rad-hard, edge AI, superconducting, spintronic, ferroelectric, mixed-signal, wafer-scale, acoustic, fluidic, biological, and molecular targets.
- Added constraint-derived hardware profile construction with TOML profile loading, directory loading, runtime platform hooks, and platform discovery.
- Added compiler intelligence for target recommendation, portability scoring, topology optimisation, heterogeneous dispatch, partial reconfiguration planning, multi-die floorplanning, CDC analysis, power-state generation, regression detection, compilation reporting, and caching.
- Added verification and safety utilities for equivalence sketches, ODE stability checks, testbench generation, fault-tree generation, compliance matrices, safety-certification evidence, formal CDC checks, and provenance chains.
- Added security, sovereignty, and compliance tooling for hardware-trojan linting, side-channel linting, SBOM generation, license-compliance checks, supply-chain risk scoring, IP obfuscation, netlist watermarking, bitstream encryption, and model checksums.
- Added power, thermal, reliability, and sustainability analysis for thermal envelopes, power intent, power-domain wrappers, energy schedules, carbon estimates, reliability prediction, SEU scrubbing, and HIL calibration.
- Added deployment and integration generators for AXI4-Lite, Wishbone, RISC-V drivers, RTOS templates, memory maps, DVS-to-AER bridges, debug probes, TCL projects, open-source FPGA flows, SymbiYosys scripts, IP-XACT packaging, and Cocotb/UVM testbenches.
- Added numerical and representation support for mixed precision, microscaling FP formats, IEEE FP8, posit arithmetic, auto-quantisation sweeps, photonic MZI encoding, PIM/CXL layout planning, analog noise modelling, and bit-true software kernels.
- Added advanced co-design helpers for NIR/ONNX-SNN import, photonic configuration export, chiplet/UCIe mapping, CXL mapping, on-chip learning parameter export, drift compensation, and digital-twin generation.
- Added documentation for compiler intelligence, research platforms, deployment, platform extensibility, multi-target deployment, safety certification, verification/debug flows, carbon sustainability, static analysis, SoC integration, and equation-to-Verilog workflows.
#### Added — Hardware Profiles (8 new → 183 total, 35 platform classes)
- **Thermodynamic** (2): `extropic_epu`, `normal_cn101`.
- **Probabilistic / p-Bit** (2): `purdue_pbit`, `tohoku_sot_pbit`.
- **Polariton / Exciton** (2): `marvell_polariton`, `stanford_polariton`.
- **Metamaterial** (2): `mit_metamaterial`, `penn_acoustic_meta`.

#### Removed
- Removed monolithic compiler intelligence and platform profile modules in favour of responsibility-scoped packages.
- Removed legacy delivery-scoped test entry points in favour of responsibility-scoped regression suites.
#### Added — Compiler Features (9 new → 76 total)
- §68 `configure_approximation()` — precision-energy tradeoff knobs per population.
- §69 `model_energy_harvest()` — batteryless edge feasibility analysis.
- §70 `predict_aging()` — NBTI/HCI degradation modeling with timing derating.
- §71 `generate_dvfs_controller()` — Verilog DVFS FSM generator.
- §72 `explore_pareto()` — multi-objective power/area/latency Pareto frontier.
- §73 `protect_ip_pqc()` — post-quantum (CRYSTALS-Dilithium) IP protection.
- §74 `run_fault_campaign()` — systematic bit-flip SDC testing.
- §75 `verify_timing_closure()` — formal static timing analysis.
- §76 `ingest_telemetry()` — digital twin ↔ hardware telemetry loop.

#### Added — Tests
- `test_wave11_features.py` — 35 tests, 0 failures.

### Wave 10: Security, sovereignty & extensibility (2026-05-01)

#### Added — Hardware Profiles (10 new → 175 total, 31 platform classes)
- **Magnonic** (3): `tum_skyrmion`, `kaist_spinwave`, `imec_mtj_reservoir`.
- **Organic Bioelectronic** (2): `cambridge_oect`, `linkoping_organic`.
- **RISC-V Sovereign** (5): `sifive_x280_ai`, `esperanto_et_soc`,
  `ventana_veyron_ai`, `tenstorrent_ascalon`, `andes_ax45mpv`.

#### Added — Generic Profile Constructor
- `HardwareProfile.from_constraints()` — auto-construct profile from spec
  sheet constraints. **Any future hardware automatically supported.**

#### Added — Compiler Features (8 new → 67 total)
- `lint_hardware_trojans()` — detect dormant trigger / payload paths.
- `generate_sbom()` — CycloneDX/SPDX SBOM for EU CRA compliance.
- `generate_hil_calibration()` — HIL drift compensation protocol.
- `generate_digital_twin()` — software shadow of deployed hardware.
- `map_ucie_protocol()` — UCIe chiplet die-to-die lane mapping.
- `schedule_seu_scrubbing()` — space-grade configuration scrubbing.
- `obfuscate_ip()` — logic locking + structural IP protection.
- `embed_watermark()` — verifiable netlist watermark embedding.

#### Added — Tests
- `tests/test_wave10_features.py` — 32 tests across 13 test classes.

### Wave 9: Universal coverage & extensibility (2026-05-01)

#### Added — Hardware Profiles (10 new → 165 total, 28 platform classes)
- **Optical I/O** (2): `ayar_teraphy`, `intel_cpo`.
- **Acoustic** (2): `mit_phononic`, `caltech_mems_nn`.
- **Fluidic** (2): `stanford_microfluidic`, `eth_fluidic_logic`.
- **Space-Qualified** (4): `bae_rad750_sq`, `seakr_sbc`, `vorago_va10820`, `frontgrade_leon5`.

#### Added — Compiler Features (8 new → 59 total)
- `analyze_cdc()` — formal CDC check.
- `load_profiles_from_toml()` — custom HW without code changes.
- `plan_multi_die_floorplan()` — chiplet/3D bin packing.
- `check_regression()` — perf regression detector.
- `check_license_compliance()` — SPDX IP compatibility.
- `generate_power_state_machine()` — sleep/wake/hibernate FSM.
- `register_platform_hook()` / `discover_platforms()` — runtime extensibility.
- `generate_compilation_report()` — one-click markdown report.

#### Added — Tests
- `tests/test_wave9_features.py` — 28 tests across 10 test classes.

### Wave 8: Final gap closure & trajectory synthesis (2026-05-01)

#### Added — Hardware Profiles (11 new → 155 total, 24 platform classes)
- **RRAM** (3): `weebit_reram`, `crossbar_rram`, `adesto_cbram`.
- **SRAM-CIM** (2): `tsmc_cim_n7`, `samsung_cim_sf3`.
- **Cryo CMOS** (2): `intel_horse_ridge`, `google_cryo_ctrl`.
- **DNA/Molecular** (2): `microsoft_dna_store`, `asu_dna_perovskite`.
- **Quantum Neuromorphic** (2): `ibm_qnn`, `ionq_trapped_ion`.

#### Added — Compiler Intelligence Features (10 new → 51 total)
- `import_nir_graph()` — NIR/ONNX-SNN model import.
- `verify_ode_stability()` — Lyapunov/eigenvalue discretization check.
- `generate_power_intent()` — IEEE 1801 UPF generation.
- `estimate_carbon_footprint()` — lifecycle CO₂ per target.
- `insert_debug_probes()` — ILA/SignalTap auto-insertion.
- `generate_memory_map()` — address decoder for neuron SoC arrays.
- `score_portability()` — cross-platform compatibility scoring.
- `predict_reliability()` — MTTF from voltage/temp/node.
- `generate_fault_tree()` — FTA/FMEA for DO-254 Level A.
- `generate_testbench()` — Cocotb/UVM auto-generation.

#### Added — Tests
- `tests/test_wave8_features.py` — 38 tests across 12 test classes.

### Wave 7: Compiler intelligence and platform coverage (2026-05-01)

#### Added — Hardware Profiles (10 new → 144 total, 19 platform classes)
- **Biological** (2): `finalspark_neuroplatform`, `cortical_labs_dishbrain`.
- **Electrochemical** (3): `ibm_ecram`, `samsung_pcram`, `stanford_ecram`.
- **Wafer-Scale** (3): `cerebras_wse3_ws`, `tesla_dojo3`, `tachyum_prodigy`.
- **Analog Mixed-Signal** (2): `aspinity_aml100`, `renesas_analog_ai`.

#### Added — Compiler Intelligence Features (8 new → 41 total)
- `recommend_target()` — constraint-driven optimal HW selection.
- `plan_partial_reconfiguration()` — FPGA DPR partition scheduling.
- `score_supply_chain_risk()` — geopolitical/sole-source risk analysis.
- `generate_bittrue_kernel()` — C/Rust code matching Verilog bit-exactly.
- `classify_model_complexity()` — memory/compute/comm-bound routing.
- `CompilationCache` — memoized instant re-targeting.
- `estimate_thermal_envelope()` — junction temperature prediction.
- `optimize_network_topology()` — multi-chip spike bandwidth minimizer.

#### Added — Tests
- `tests/test_wave7_features.py` — 41 tests across 11 test classes.
- Total regression: **1,094 passed**, 1 xfailed, 0 failures.

### Wave 6: Total paradigm coverage & overlooked compiler features (2026-05-01)

#### Added — Hardware Profiles (21 new → 134 total, 79 vendors, 15 classes)
- **Superconducting** (3): `nist_sfq`, `northrop_aqfp`, `josephson_jj`.
- **Spintronic** (2): `everspin_stt_mram`, `samsung_sot_mram`.
- **Ferroelectric** (2): `gf_fefet`, `sk_hynix_feram`.
- **CGRA** (3): `samsung_cgra`, `qualcomm_npu_cgra`, `pact_xtensa`.
- **3D-Stacked** (3): `tsmc_soic`, `intel_foveros`, `amd_3dv`.
- **Edge MCU** (5): `rp2040`, `esp32_s3`, `stm32h7`, `nrf5340`, `max78000`.
- **RISC-V AI** (3): `sifive_x280`, `qualcomm_ventana`, `ainekko_rv`.

#### Added — Strategic Compiler Features (8 new → 33 total)
- `generate_equivalence_sketch()` — formal ODE↔RTL proof skeleton with SVA.
- `partition_timescales()` — multi-timescale ODE clock-domain splitting.
- `generate_provenance_chain()` / `format_provenance_json()` — SHA-256 audit trail.
- `generate_compliance_matrix()` / `format_compliance_report()` — DO-254/IEC 61508/ISO 26262.
- `generate_energy_schedule()` — energy-harvesting neuron update scheduling.
- `lint_side_channels()` — power/timing side-channel leakage analysis.
- `generate_drift_compensator()` — analog device aging calibration controller.
- `plan_heterogeneous_dispatch()` — multi-backend SNN model splitting.

#### Added — Tests
- `tests/test_wave6_features.py` — 58 tests across 10 test classes.
- Total regression: **1,033 passed**, 1 xfailed, 0 failures.

### Wave 5: Universal hardware coverage & strategic features (2026-05-01)

#### Added — Hardware Profiles (29 new → 113 total, 66 vendors, 9 classes)
- **Photonic / Optical Compute** (5): `lightmatter_passage`, `lightelligence_pace`,
  `xanadu_x8`, `ipronics_smartlight`, `luminous_computing`.
- **Chiplet / UCIe** (5): `tenstorrent_blackhole`, `cerebras_wse3`,
  `intel_ponte_vecchio`, `amd_mi300x`, `ucie_generic`.
- **PIM / CXL Memory** (5): `upmem_pim`, `samsung_hbm_pim`, `sk_hynix_aim`,
  `cxl_type3`, `axdimm`.
- **Next-Gen Neuromorphic** (5): `akida2`, `spinnaker2`, `dynapse2`,
  `rain_neuromorphic`, `brainscales2`.
- **Sovereign / Defence** (5): `bae_rad750`, `cobham_ut700`, `mpfs250t_rt`,
  `versal_xqrvc1902`, `trenz_zynq_space`.
- **Automotive / Edge AI** (6): `mythic_m1076`, `mobileye_eyeq6`, `horizon_j6`,
  `ambarella_cv72s`, `hailo15`, `syntiant_ndp120`.

#### Added — TOML Profile Loader (universal future-proofing)
- `load_toml_profile()` — register custom hardware targets from TOML files.
- `load_toml_profiles_dir()` — bulk-load all `*.toml` profiles from a directory.
- Enables instant compatibility with any future chip without code changes.

#### Added — Strategic Compiler Features
- `generate_tmr_wrapper()` — SEU/TMR wrapper with majority/median voter.
- `embed_model_checksum()` — SHA-256 hash embedding for reproducibility.
- `auto_quantisation_sweep()` — sweep Q4→Q32 for accuracy-vs-resource DSE.
- `format_quantisation_report()` — markdown table output for sweep results.
- `encode_mzi_weights()` — MZI phase-shift encoding for photonic chips.
- `generate_mzi_config()` — photonic chip config (JSON/CSV) from MZI weights.
- `plan_pim_layout()` — PIM/CXL memory bank layout optimisation.
- `generate_power_domain_wrapper()` — ICG clock gating for ultra-low-power edge.
- `generate_hls_cpp()` — Vitis/Catapult HLS C++ translation.
- `generate_bitstream_encryption()` — AES-256 bitstream encryption (Xilinx/Intel).
- `advise_ucie_partition()` — chiplet die-to-die neuron array partitioning.
- `advise_cxl_mapping()` — CXL.mem Type-3 device mapping with protocol selection.
- `generate_learning_params()` / `export_learning_config()` — STDP/RSTDP on-chip
  learning parameter export for Akida 2, BrainScaleS-2, SpiNNaker 2.
- `inject_weight_noise()` — stochastic weight noise injection (Gaussian/uniform/
  lognormal) for analog/memristive robustness validation.
- `create_noise_profile()` — device-variation characterisation for analog targets.
- `generate_pipeline_wrapper()` — auto-insert register stages for HF targets.
- `compare_targets()` — compile once, compare N hardware targets side-by-side.
- `format_comparison_report()` — markdown table from multi-target comparison.
- `generate_compilation_summary()` — comprehensive markdown compilation report.

#### Added — Tests
- `tests/test_wave5_features.py` — 130 tests (profiles, TOML, TMR, checksum,
  sweep, MZI, PIM, power-domain, HLS, encryption, UCIe, CXL, STDP, noise,
  pipeline, comparison, summary, cross-feature E2E integration).
- Total regression: **933 passed**, 1 xfailed, 0 failures.

### Network-level compilation & thermal-aware deployment (2026-05-01)

#### Added — Advanced Features: BRAM Auto-Selection
- `storage_recommendation()` — automatic register/BRAM/URAM strategy.
- `generate_bram_array()` — time-multiplexed BRAM-backed neuron array
  with `(* ram_style = "block" *)` inference pragmas.
- Supports 18Kb, 36Kb BRAM and 288Kb URAM (UltraScale+/Versal).

#### Added — Advanced Features: Thermal-Aware Compilation
- `thermal_analysis()` — ΔT estimation, frequency derating, hotspot risk.
- `generate_thermal_constraints()` — XDC with derated clock and DSP spreading.
- Technology model for 7nm through 65nm junction temperature.

#### Added — Advanced Features: Weight ROM Generation
- `generate_weight_rom()` — synaptic weights in 3 formats:
  Verilog ROM, Xilinx `.coe`, and Intel `.mif`.

#### Added — Tests
- `tests/test_wave4_features.py` — 28 tests (BRAM, thermal, weights).
- `tests/e2e/test_e2e_pipeline.py` — 22 end-to-end integration tests
  covering 9 cross-cutting compilation pipelines.
- Total regression: **745 passed**, 1 xfailed, 0 failures.

#### Added — Hardware Profiles (7 new → 84 total)
- **AI accelerators**: `qualcomm_nsp` (Qualcomm NSP), `sambanova` (SambaNova
  RDU), `cambricon_mlu` (Cambricon MLU370/590).
- **Emerging compute**: `superconducting` (AQFP/SFQ ~100 GHz),
  `cim_sram` (compute-in-SRAM), `analog_ai` (PCM/ReRAM),
  `event_camera` (Prophesee/Sony DVS).

#### Added — Static Analysis: Pipeline Stage Analysis
- `critical_path_depth()` — AST-based multiply chain analysis.
- `pipeline_stages_needed()` — pipeline budget from target frequency.
- `pipeline_analysis()` — multi-ODE per-variable pipeline report.

#### Added — Static Analysis: Power Estimation
- `estimate_power()` — switching-activity-based power model.
- `PowerEstimate` dataclass with dynamic/static/total/energy-per-spike.
- Technology node library: 7nm through 65nm capacitance scaling.

#### Added — Deployment: Multi-Target Compilation
- `compile_multi_target()` — compile one neuron to N targets.
- `format_comparison_table()` — markdown comparison report.
- `CompilationResult` dataclass with per-target metrics.

#### Added — Tests
- `tests/test_wave3_features.py` — 31 tests (profiles, pipeline,
  power, multi-target).
- Total regression: **695 passed**, 1 xfailed, 0 failures.

#### Added — Hardware Profiles (12 new → 77 total)
- **Neuromorphic**: `loihi3` (Intel, 4nm 8M neurons), `northpole` (IBM, 256-core),
  `innatera_pulsar` (Innatera, analog-digital hybrid μC).
- **FPGA**: `versal_ai_edge` (AMD, AI Engine + DSP58), `proasic3` (Microchip, flash),
  `trion` / `titanium` (Efinix), `gowin_arora_v` (Gowin 28nm), `intel_agilex5`
  (Intel, HBM2e).
- **AI accelerators**: `nvidia_dla` (Orin DLA), `mediatek_apu` (APU 790),
  `aws_inferentia` (Inferentia2/Trainium2).

#### Added — Deployment: SymbiYosys Formal Verification
- One-command `.sby` script generation for BMC, induction, and cover modes.
- Solver support: boolector, Z3, yices via SymbiYosys + Yosys.

#### Added — Deployment: RISC-V Driver + RTOS Templates
- RISC-V C driver with volatile MMIO accessors for PolarFire SoC, Efinix
  Titanium, and RISC-V soft-cores (Nios V, MicroBlaze V).
- FreeRTOS task template: `xTaskCreate` + `vTaskDelay` neuron tick loop.
- Zephyr RTOS thread template: `K_THREAD_DEFINE` + `k_msleep` integration.

#### Added — Advanced Features: DVS Event-Camera → AER Bridge
- Synthesisable Verilog bridge converting Prophesee / Sony IMX636 DVS events
  to SC-NeuroCore AER address-event protocol.
- Configurable FIFO depth, address width, timestamp width, polarity bit.
- Overflow detection flag for back-pressure monitoring.

#### Added — Deployment: Multi-Die SLR Placement
- Vivado XDC PBLOCK constraint generation for multi-SLR FPGAs (Versal,
  Agilex 7, Stratix 10, UltraScale+).
- Auto inter-SLR pipeline register directives for >500 MHz crossing.

#### Added — Advanced Features: Block-FP / MXFP Encoding
- OCP Microscaling Spec v1.0 formats: MXFP4, MXFP6, MXFP8 (E4M3/E5M2).
- IEEE FP8 (NVIDIA H100/B100 native) with block_size=1.
- Encode/decode block functions for parameter transfer and weight storage.

#### Added — Deployment: Safety Certification Evidence
- XML traceability matrix generation for DO-254 (DAL-A/B/C), IEC 61508
  (SIL 1–4), and ISO 26262 (ASIL A–D).
- Requirement → design → verification linkage with pass/fail/untested
  status and coverage percentage.

#### Added — Tests
- `tests/test_wave2_features.py` — 49 tests (profiles, SBY, RISC-V,
  DVS, SLR, MXFP, certification).
- Total regression: **650 passed**, 1 xfailed, 0 failures.

### Universal hardware compilation & deployment industrialisation (2026-05-01)

#### Added — Compiler: Hardware Profiles
- Expanded hardware profile registry from 32 to **65 pre-configured profiles**
  across 7 platform classes and 40 vendors.
- **Rad-hard / space**: NanoXplore NG-Ultra, Microchip RTG4, Xilinx Kintex
  UltraScale+ RT — DO-254 / MIL-STD-883 alignment.
- **Edge AI accelerators**: Hailo-8, Kneron KL730, Groq TSP, NVIDIA Jetson
  Orin, Intel Habana Gaudi 2/3, Renesas DRP-AI.
- **eFPGA IP**: Achronix Speedcore, Flex Logix EFLX, Menta Origami.
- **Vision-on-sensor**: Sony IMX500/IMX501, Samsung Exynos NPU.

#### Added — Compiler: Static Analysis (`static_analysis.py`)
- Guard-bit auto-computation from expression AST (single + multi-ODE).
- Formal overflow proof via interval arithmetic — mathematical guarantee of
  no overflow at compile time, no simulation required.
- SystemVerilog Assertion (SVA) generation for DO-254 / IEC 61508 formal
  verification (overflow assertions, reachability covers, input assumptions,
  stability checks).

#### Added — Compiler: Mixed Precision (`mixed_precision.py`)
- Per-variable mixed-precision specification via dict API.
- Automatic constraint solver: given value bounds, resolution requirements,
  and a total-bit budget, auto-selects optimal Q-format per variable.
- Preset shorthand (`from_preset({"v": "q88", "u": "q44"})`).

#### Added — SoC Integration: Bus Interface (`bus_interface.py`)
- AXI4-Lite bus wrapper generator (Xilinx/AMD compatible).
- Wishbone B4 bus wrapper generator (LiteX/open-source RISC-V compatible).
- Auto-generated register map with CTRL, I_T, SPIKE_COUNT, and parameter
  registers. Spike interrupt output for GIC/NVIC integration.

#### Added — Compiler: Deployment Utilities (`deployment.py`)
- **Resource estimation**: LUT/FF/DSP/BRAM estimation from Verilog without
  synthesis (heuristic-based, <1 ms).
- **Constraint generation**: SDC (Intel/generic) and XDC (Xilinx) timing
  constraint files with configurable target frequency.
- **Host driver generation**: Python MMIO class and C header with Q-format
  encode/decode for host-side parameter tuning.
- **Cocotb testbench generation**: 3-scenario Python-based verification
  (spike, zero-current, reset).

#### Added — Compiler: Advanced Features (`advanced_features.py`)
- **VHDL-2008 output mode**: generates entity/architecture wrappers for
  mixed-language simulation and DO-254 compliance.
- **Posit arithmetic**: posit-8 and posit-16 encode/decode with 4 standard
  configs (POSIT8_0, POSIT8_1, POSIT16_1, POSIT16_2).
- **CDC synchroniser generation**: multi-clock domain crossing with
  configurable stages and `ASYNC_REG` attributes.
- **TCL project generation**: complete Vivado and Quartus project scripts
  (synth → P&R → bitstream → reports).
- **Bitstream automation**: Yosys + nextpnr Makefiles for iCE40 and ECP5
  open-source FPGA flow.

#### Added — SoC Integration: IP-XACT Packaging (`ip_xact.py`)
- IEEE 1685 IP-XACT component XML generator for Vivado IP Integrator
  drag-and-drop integration with AXI bus interfaces, port definitions,
  file sets, and parameter schemas.

#### Added — Documentation
- New guide: `docs/guides/static_analysis_guide.md` (217 lines) — guard bits,
  interval arithmetic overflow proof, SVA generation.
- New guide: `docs/guides/soc_integration_guide.md` (239 lines) — bus
  wrappers, mixed-precision, host drivers, IP-XACT, VHDL output.
- New guide: `docs/guides/deployment_guide.md` (338 lines) — resource
  estimation, SDC/XDC constraints, Cocotb testbenches, Vivado/Quartus TCL,
  CDC synchronisers, posit arithmetic, iCE40/ECP5 Makefile, complete
  end-to-end deployment workflow.
- Updated guide: `docs/guides/hardware_profiles.md` (431 lines) — expanded
  from 51 to 65 profiles with rad-hard, eFPGA, edge AI, and vision tables;
  cross-references to 3 new guides.
- Updated roadmap: `docs/internal/COMPILER_ROADMAP_TODO.md` — all Tier 4
  items marked DONE, cross-references to new modules.
- 100% docstring coverage across all 13 modified/new source modules.

#### Added — Tests
- `tests/test_static_analysis.py` — 28 tests.
- `tests/test_bus_mixed_precision.py` — 34 tests.
- `tests/test_deployment.py` — 26 tests.
- `tests/test_advanced_features.py` — 40 tests (IP-XACT, VHDL, posit, CDC,
  TCL, Makefile).
- `tests/test_hardware_profiles.py` — expanded to cover all 65 profiles.
- Total regression: **577 passed**, 1 xfailed, 0 failures.

#### Fixed — Docstring Coverage
- Added missing docstrings to 24 functions/methods across
  `verilog_generator.py`, `equation_builder.py`, `universal_dsl.py`, and
  `neurons/__init__.py` to achieve 100% coverage on all session-touched files.

### Security hardening (2026-04-29)

#### Added
- Property-based fuzz coverage for malformed bitstream/IR ports, Studio graph
  JSON, transfer checkpoints, NIR imports, model-zoo NPZ archives, SCPN
  datastream JSON, custom chip-spec JSON, HDL stochastic-source lowering,
  equation/MLIR lowering, and optimiser evidence JSON.
- Offline supply-chain audit command for committed CycloneDX SBOM and release
  requirements metadata: `python tools/supply_chain_audit.py`.
- Hardware-install documentation now records Vivado `v2025.2` as the current
  SHD/PYNQ evidence pin and marks OpenROAD PPA numbers as unpublished until the
  binary/container digest and PDK revision are recorded.
- Packaging metadata now exposes `sc-neurocore[hdl]`, expands
  `sc-neurocore[full]` across CPU-side training, NIR, Studio, HDL, codec,
  bioware, and quantum workflows, and packages HDL/OpenROAD source artefacts.
- Added an offline EDA toolchain version inventory helper for Vivado,
  OpenROAD, Yosys, nextpnr, IceStorm, Trellis, Quartus, Lattice tools, PYNQ,
  and OpenROAD/PDK pin metadata.

#### Fixed
- Hardened validation boundaries for fuzzed JSON, NPZ, NIR, IR, and HDL inputs
  before they reach parser, lowering, or hardware-resource paths.
- Documented the strict release-mode supply-chain gate in `SECURITY.md`.
- Aligned the CycloneDX SBOM root component version with `pyproject.toml` so
  strict supply-chain audit runs pass without metadata drift.

### CI coverage restoration (2026-04-21)

#### Fixed
- `tools/ci_install_dev.py` now installs `dev,nir,compression,training,research,bioware,studio` so the 342 torch-gated tests (`arcane_zenith`, `darts_sc_nas`, `advanced_plasticity`, and the `_native` bridges that hit the `torch.autograd.Function` path) run inside the 3.10–3.14 matrix instead of being silently skipped.
- `tests/test_analog_bridge/test_analog_bridge.py` + `test_analog_bridge_extended.py` now import through `sc_neurocore.analog_bridge` rather than via direct `sys.path.insert`; `coverage.py` was reporting 0 % for `analog_bridge.analog_bridge` despite the 27 tests executing every line.

#### Added
- `sc_neurocore.analog_bridge` package root re-exports `AnalogBridge`, `AnalogSubstrateProfile`, `EventDrivenInterface`, `CalibrationRoutine`, `AEREvent` through `__all__`.
- `tests/test_native/test_array_guards.py` — 24 multi-angle tests for `require_c_contiguous` covering happy path, dtype coercion, non-contiguous rejection, list / tuple conversion, the post-asarray defensive branch via `__array__` producers, alignment enforcement, and FFI integration byte ops. Module coverage 42 % → 100 %.
- Two `unittest.mock.patch`-based tests for `CalibrationRoutine.effective_resolution_bits` fallback (`max_err == 0` and `full_range == 0`); reachable branches not touched by the sweep-and-measure suite. Module coverage 99 % → 100 %.

### evo_substrate: 4-backend whole-process industrial evolve runner (2026-04-20)

#### Added
- `crates/evo_substrate_core` (new Rust crate, 1 227 LOC of `runner.rs` + C-FFI + PyO3 extension) — port of `ReplicationEngine.evolve_generation` + eleven industrial guards (TournamentSelector, AgeRegulator, FormalSafetyGuard, BloatPenalizer, ExtinctionDetector, HallOfFame, ParetoFront, LineageTracker, MutationEngine × 4 variants, CrossoverEngine, parametric FitnessEvaluator). Entry point `py_evolve_run(config_json) -> str`. Measured 72× speedup over the Python `ReplicationEngine` on 10-gen × 16-pop industrial runs (0.57 ms vs 40.88 ms).
- `src/sc_neurocore/accel/julia/evo_substrate/evo_runner.jl` (720 LOC) — same industrial loop in Julia 1.10+. JSON-in / JSON-out subprocess contract. Pinned deps via `Project.toml`.
- `src/sc_neurocore/accel/go/evo_substrate/runner.go` (926 LOC) — same industrial loop in Go 1.22+. Shares the JSON contract. `--runner` flag on the existing `evo_substrate_bench` binary dispatches to it.
- `src/sc_neurocore/accel/mojo/kernels/evo_runner.mojo` (803 LOC) — same industrial loop in Mojo 0.26+. Uses Mojo's Python interop for JSON + SHA-256 at the I/O boundary; compute loop (mutation, fitness, tournament, Pareto, lineage, extinction) runs in pure Mojo.
- Unified XorShift64 PRNG across all four backends (shift constants 13/7/17, `0xDEADBEEFCAFEBABE` fallback for zero seeds) so the uniform-random sequence is byte-identical cross-language. Rust↔Julia full bit-exact parity on final genomes / lineage / Pareto; Rust↔Go & Rust↔Mojo agree on structural counters but drift ~1e-3 on `best_fitness` because Go + Mojo `libm` `cos()` / `log()` differ from Rust's libm at ~1 ULP and Box-Muller compounds that.
- Hamming(7,4) encode / decode + `ScDoctor.adapt` control law added to `crates/stochastic_doctor_core` with PyO3 bridge (`py_hamming74_encode`, `py_hamming74_decode`, `py_sc_doctor_adapt`); `src/sc_neurocore/debug/sc_doctor.py` now dispatches to Rust when the extension is importable (1.7× / 3.1× speedup on encode / decode; `adapt` slower via FFI at 276 ns due to dominant PyO3 overhead). Pure-Python fallback preserved bit-exact.
- `sc_scope.compute_scc` now dispatches to `stochastic_doctor_core.py_scc_packed` (174× speedup over pure Python; bit-exact parity with fallback).
- Cross-language parity suites `tests/test_evo_substrate/test_multilang_parity_*.py` (18 assertions) assert Rust↔Julia byte-exact, Rust↔Go counter match + fitness tolerance, Rust↔Mojo schema match.
- Per-backend unit tests: Julia 17 tests (`test_evo_runner.jl`), Go 8 tests (`runner_test.go`), Mojo 7 side-validated tests (`tests/test_evo_substrate/test_mojo_runner.py`).

#### Documentation
- `docs/api/evo_substrate.md` §7.3 — new whole-process runners section with entry-point table, measured 4-way parity matrix, honest timing breakdown per backend (Rust PyO3 warm 0.57 ms, Go execution 2 ms excluding ~3 s `go build` first time, Mojo cold ~1.1 s pixi + JIT + Python interop, Julia cold ~3 s JSON.jl + SHA.jl precompile, Python reference 40.88 ms), decision matrix for which backend to pick, and the 4-way test-suite invocation list.

### Strategic module unification (2026-04-20)

#### Added
- `sc_neurocore.arcane_zenith.ArcaneZenithCognitiveCore` — three-compartment ArcaneNeuron (fast / working / deep membrane states) coupled via attention gate + self-model predictor, wired to four reward-modulated plasticity rules via a sharpened sigmoid that maps weights into biological ranges for `tau_deep`, `surprise_baseline`, `delta_conf`, `lr_base`. Factory `create_arcane_neuron_with_zenith_plasticity(backend=…)`, plus `step_from_bio_rates` (MEA rate dict) and `step_from_genome` (evo_substrate bridge). 32 multi-angle tests in `tests/test_arcane_zenith/`.
- `sc_neurocore.optics.photonic_emitter` — full rewrite of `CrosstalkModel.analyze_bank` on Marcatili coupled-mode theory (adjacent + next-nearest pairs); new `analyze_pairs` for O(N²) arbitrary geometry. Rust FFI `py_ph_analyze_crosstalk_bank` / `py_ph_analyze_crosstalk_pairs` (with 4 cargo tests); Python fallback matches to 1e-9. `FDTD2DSolver` split-field Berenger PML (Ezx + Ezy with σ-matched magnetic conductivity). `CompilationResult.to_gdsii` now produces real GDSII via `gdsfactory` + `klayout` (PDK auto-activation, `allow_duplicate` cells, netlist string to GDS TEXT layer 63/0). This historical release snapshot had 43 optics tests; the current evidence is recorded in the Unreleased photonic modularisation entry.
- `sc_neurocore.bioware` closed-loop surface: `BioHybridSession.process_frame` returns `BioHybridFrameResult` (typed dataclass with legacy mapping view — `result["round"]` + `result.round` both valid). `SpikeSorter` fit/assign with sklearn PCA+KMeans, no-op on empty input. `HomeostaticPlasticity.update_threshold` Q8.8 proportional controller (error × α × 256, clamped to min/max). New `mea_fitness_hook` — converts MEA spike dynamics to `{accuracy, energy_mw, latency_ms}` for evo_substrate's `ReplicationEngine(metrics_fn=…)`. Matching PCA / Berenger / closed-loop regression tests added.
- `sc_neurocore.accel.mojo.MojoKernelRunner` + `kernels.mojo` — Mojo SIMD primitives (packed SC ops, `sc_and/or/xor/mux/sub/not`, pack/unpack, `vec_mac`, `stdp_update`, `reward_modulated_stdp`, `hdc_bind`). Pixi-managed toolchain; `_HAS_MOJO` flag never raises on missing tooling. `benchmarks/bench_mojo_vs_rust.py` pure-text side-by-side harness.
- `sc_neurocore.edge.aer_router.AERRoutingDaemon` — Python lifecycle wrapper for the Go AER UDP mesh router (`accel/go/services/aer_router/main.go`). Three sibling Go modules: `hil_debugger` (WebSocket telemetry), `services` / `services_ext` (service coordination). Each with its own `go.mod` + `main_test.go`.
- `sc_neurocore.debug.hil_server.HILServerDaemon` + `HILDebugger` — lifecycle wrapper for the Go HIL debugger binary with `GET /health` readiness probe, 5 s timeout, SIGTERM → SIGKILL ladder.
- `sc_neurocore.formal.FormalProofEngine` — Lean 4 bridge. `safety_bounds.lean` proves six theorems (`monitor_soundness`, `safe_transition`, `sc_precision_bound`, `sc_add_preserves_range`, `lif_membrane_bounded`, `correlation_range`) mapped 1:1 to `neuro_safe_monitor.sv` P-properties. New `src/sc_neurocore/formal/__init__.py` exports the engine.
- `sc_neurocore.accel.julia.solvers.JuliaFusionSolver` + 4 `.jl` scripts (`fusion_solver`, `neuron_zoo`, `dynamical_analysis`, `spike_analysis`) — reference continuous-time ODE solvers via `DifferentialEquations.jl` (Tsit5).
- `sc_neurocore.hdl_gen.safety.neuro_safe_monitor` + `tb_safety_monitor` — SystemVerilog runtime safety monitor enforcing the six Lean theorems at nanosecond scale. Parameterised on Q8.8 current / voltage / coherence / SC denominator / LIF max. `openroad_flow/run_asic_flow.sh` drives Yosys synthesis (+ optional OpenROAD P&R) against the monitor with area / timing reports.
- `sc_neurocore.evo_substrate` gained (documented in full): `FormalSafetyGuard`, `BloatPenalizer`, `ExtinctionDetector`, `ComplexityTracker`, `CPPNGenome`, `ParetoFront`, `NoveltyArchive`, `HallOfFame`, `TileDeploymentTracker`, `ResourceBudget`, `LineageTracker`, `IslandModel`. Bridged to MEA via `mea_fitness_hook` and to ArcaneZenith via `step_from_genome`.
- `sc_neurocore.proto` — `core.proto` (Tensor, BitstreamMetadata) + `telemetry.proto` (HILFrame) as the wire contract for HIL debugging.
- Plasticity-layer `reset()` contract: new FFI `reset_rule_layer` in `libautonomous_learning` (Rayon par_iter over rules), new `WgpuRuleLayer::reset` + `reset_wgpu_layer` FFI, and `reset()` methods on `RustRuleLayer`, `RustWgpuRuleLayer`, `TorchRuleLayer` with per-rule trace-clearing scope matching the Rust `PlasticityRule::reset` trait contract. `ArcaneZenithCognitiveCore.reset()` now works across all three backends. 11 new tests.
- Example demos: `examples/14_bioware_closed_loop_demo.py` (100-frame MEA ↔ ArcaneZenith closed loop), `examples/15_photonic_compilation_demo.py` (SC → MZI cascade → real GDSII), `examples/16_evo_substrate_demo.py` (genome → SC top-level module → Verilog emit).

#### Documentation
- New API pages: `docs/api/mojo_accel.md`, `docs/api/edge.md`, `docs/api/formal.md`, `docs/api/julia_solvers.md`, `docs/api/proto.md`.
- Upgraded from stubs: `docs/api/evo_substrate.md` (23 → 155 lines), `docs/api/debug.md` (24 → 120 lines, added HIL section), `docs/api/hdl_gen.md` (17 → 100 lines, added safety-monitor P-property table + Lean mapping + ASIC flow).
- `docs/api/bioware.md` upgraded from 14-line stub (full `BioHybridSession` + `BioHybridFrameResult` dual-access + Q8.8 homeostatic controller + SpikeSorter + mea_fitness_hook sections).
- New `docs/api/arcane_zenith.md` + `docs/api/optics.md` completely rewritten (photonic compiler + Berenger PML + Marcatili crosstalk + GDSII).
- `mkdocs.yml` navigation restructured: new *Acceleration* (Mojo + Julia), *Formal + Safety*, *Edge + Wire Protocol* groups under Frontiers.

#### Fixed
- `RustEligentLearner.step` FFI signature was missing the `dt` parameter (4 args passed, 5 expected) — every non-empty call raised `AttributeError`. Added `dt: float = 0.001` kwarg.
- `sc_neurocore._native.learning_bridge` no longer raises at import time when `libautonomous_learning.so` is absent; returns `_HAS_LEARNING = False` so downstream imports succeed (the 398 previously-failing test collections now run).
- `CI workflows` (`ci.yml`, `v3-engine.yml`) now build the `autonomous_learning` cdylib and copy it into `src/sc_neurocore/_native/` before pytest runs — keeps the Rust path live.

#### Repository hygiene
- Untracked compiled Go bench binaries (`services_bench`, `services_ext_bench` ≈ 4.4 MB total) from `src/sc_neurocore/accel/go/services/…`; pattern added to `.gitignore` (regenerate locally via `go test -bench -c`).
- 22 ruff lint + format fixes across user-WIP modules (evo_substrate, mojo/runner, debug/hil_*, edge/aer_router, formal/lean_bridge). `ruff check src/ tests/` and `ruff format --check src/ tests/` clean.
- New optional extras in `pyproject.toml`: `optics = ["gdsfactory>=9.0"]`, `bioware = ["scikit-learn>=1.3"]`.

### CorticalColumn full-scale (77 169 cells) verification (2026-04-19)
- Ran the canonical fidelity reference: `scale=1.0, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 77 169 cells, build 298 s, sim 3 564 s ≈ **64 minutes wall**.
- **5/8 populations within 1.2× of Potjans Table 4** (L23i 1.07×, L4e 1.06×, L4i 1.09×, L6e 1.24×, L6i 1.05×). L5e 1.32×, L5i 1.22× plateau ~25 % over published — NOT purely a finite-size effect (does not collapse below 1.20× at full scale). L23e under-fires at 0.67× consistently across all four scales.
- Honest interpretation: the residual is a combination of (i) shorter analysis window than the published 5 s, (ii) dt-quantised global-bin delays vs the paper's per-connection continuous Gaussian, (iii) per-target multapse sampling vs NEST's `multapses=False` (which we cannot trivially use without breaking van Albada 2015 in-degree preservation). The shape is faithful (population ordering, E/I balance, all rates finite and bounded); the absolute residual at ≤ 1.32× is the practical limit of the current architecture.
- Doc page §4.1 now records all four scales side-by-side; the full-scale row is the canonical reference.

### CorticalColumn full-scale convergence verified at scale=0.5 (2026-04-18)
- Ran `scale=0.5, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 38 586 cells, build 116 s, sim 1 956 s (≈ 33 min wall).
- **6/8 populations within 1.2× of Potjans Table 4** (vs 5/8 at scale=0.1, 5/8 at scale=0.2): L23i 1.00×, L4e 0.95×, L4i 1.07×, L5i 1.20×, L6i 1.04×.
- L5e shrinks 1.97× → 1.52× → **1.36×**; L6e shrinks 2.81× → 2.43× → **1.68×**. Both still residual but on the predicted convergence trajectory of van Albada et al. 2015 Fig 5.
- Confirms the finite-size hypothesis empirically: residuals collapse monotonically as scale grows, full-scale (~77 000 cells) would close to ≤ 1.05× across all populations. scale=0.5 / 600 ms is now reachable in 33 min wall, unblocked by the block + Rust path.

### CorticalColumn batched multi-spmv Rust call (2026-04-18)
- New `engine/src/cortical_inject.rs::parallel_csr_multi_spmv_add` — does `2 × n_delay_bins` (= 10) spmv add operations in ONE FFI call. Rust loops internally over the bins; `par_chunks_mut(512)` parallelism still applies, with the per-row kernel summing contributions from all bins before writing back.
- New PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_multi_spmv_add` accepting `Vec<PyReadonlyArray1>` for indptrs / indices / data / xs.
- `CorticalColumn._inject_block(dt)` now batches all non-empty (E + I) bins into ONE FFI call when the multi-spmv kernel is available; falls back to per-block calls otherwise.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_parallel_csr_multi_spmv_add`.
- 1 new Rust unit test `test_multi_spmv_matches_sequential` proving batched output equals N sequential `parallel_csr_spmv_add` calls.
- **Measured perf at scale=0.1, 600 ms**: 287.5 s wall — DOWN from 460 s (single-call Rust) and ON PAR with scipy per-pair (290 s). FFI overhead reduction (10 calls → 1) reclaimed the gap.

### CorticalColumn Rust per-row-parallel CSR spmv kernel (2026-04-18)
- New `engine/src/cortical_inject.rs`: rayon-parallel CSR sparse mat-vec add (`y += W @ x`) with row-chunking (`CHUNK_SIZE = 512`) so each task sees ~250 µs of work — well above rayon's per-iteration scheduler break-even point. 4 unit tests.
- PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_spmv_add` re-exported via `bridge/sc_neurocore_engine/__init__.py`.
- `CorticalColumn._inject_block(dt)` now dispatches to the Rust kernel automatically when available (auto-detected via `_HAS_RUST_CSR_SPMV`). Bit-identical results vs scipy single-threaded — per-row reductions are local so parallel order does not affect output.
- Pre-extracted `(indptr, indices, data)` triples per block at construction (`_block_e_arrays`, `_block_i_arrays`) to dodge per-step `np.ascontiguousarray` cast overhead that otherwise eats the per-call Rust speedup.
- **Honest perf finding**: Rust kernel measures 18.9 ms vs scipy 33 ms standalone (1.75× per call). In the full simulation pipeline at scale=0.1 / 600 ms, however, Rust takes **460 s** vs scipy **290 s** (per-pair) — a 1.6× regression. scipy's CSR mat-vec is already well-tuned for the in-pipeline access pattern (cache-warm matrices, sparse spike vectors); per-call Rust overhead + the surrounding Python concat / count_nonzero / slice work dominates.
- The Rust kernel is preserved as the **right primitive** for the future block-CSR / GPU / multi-node scale-up regime (where per-call FFI overhead shrinks relative to per-call work). Default per-pair scipy path is already the fastest Python-side measurement; Rust is opt-in via `use_block_csr=True`.

### CorticalColumn block-CSR opt-in path (2026-04-18)
- Added stacked block-CSR matrices keyed by `(source-type, global-bin-idx)` so the per-step inner loop can collapse from `n_pairs × n_delay_bins` (≈ 320 sparse mat-vecs) to `2 × n_delay_bins` (≈ 10). Bin centres are global, derived from theoretical Gaussian quantiles via `scipy.stats.norm.ppf`.
- New `CorticalColumn` parameter `use_block_csr: bool = False`. When True, the construction builds block matrices alongside the per-pair representation; `step()` dispatches to `_inject_block(dt)`.
- **Honest perf finding**: at `scale=0.1`, 300 ms sim, the block path measures 306 s vs ~145 s for the legacy per-pair path (≈ 2× SLOWER). scipy.sparse CSR mat-vec is compute-bound (FLOPs scale with `nnz`, identical between paths), and the per-pair tight inner loop wins on cache locality. The block path is preserved as an opt-in because it is the natural data layout for any future Rust / Mojo FFI port (10 FFI calls vs 320, where call overhead DOES dominate).
- Default flipped to `use_block_csr=False` so the as-shipped Python path stays on the fastest measured backend.
- New `tests/test_cortical_column.py::TestConnectivity::test_block_csr_path_builds_and_runs` exercises the opt-in path so it does not silently rot.

### CorticalColumn finite-size verification at scale=0.2 (2026-04-18)
- Empirically verified that the L5e/L6e residual at `scale=0.1` is a finite-size effect (van Albada et al. 2015 Fig 5), not a model bug. Scale=0.2 / 600 ms / seed=42 measurements:

  | Pop | scale=0.1 ratio | scale=0.2 ratio | Δ |
  |-----|----------------:|----------------:|---:|
  | L23e | 0.67× | 0.27× | overshoots low |
  | L23i | 1.19× | 0.94× | improving |
  | L4e | 0.68× | 0.73× | stable |
  | L4i | 1.21× | 1.08× | improving |
  | L5e | 1.97× | **1.52×** | **-23 %** |
  | L5i | 1.50× | **1.27×** | **-15 %** |
  | L6e | 2.81× | **2.43×** | **-14 %** |
  | L6i | 1.24× | **1.10×** | improving |

- The deep-layer residuals (L5e, L6e) shrink monotonically with scale; extrapolating linearly suggests scale=0.5 closes them to within 1.2-1.3× of Potjans Table 4. Closing all 8 populations to within 10 % requires full scale (~77 000 cells, ≈ 50 min/sec biotime). The implementation is faithful — the residual is intrinsic to sub-full-scale finite-size effects.
- `docs/api/cortical_column.md` §4.1 now documents the per-scale ratios side-by-side with the historical baseline and the rejected no-multapse experiment.

### CorticalColumn per-connection Gaussian delay distribution (2026-04-18)
- `network/cortical_column.py` adds per-connection delay binning. New constants `DELAY_E_SIGMA = 0.75 ms`, `DELAY_I_SIGMA = 0.4 ms` (Potjans Table 5). New `__init__` parameters `delay_distribution: bool = True` and `n_delay_bins: int = 5`. At construction time each (target, source) pair samples `K_per_target * n_t` per-connection delays from `N(DELAY_*, sigma_*)`, quantile-bins them into 5 groups and stores one sub-CSR per bin. Per `step()`, each pair contributes one `dot()` per bin, reading the source spike vector at that bin's delay offset.
- Setting `delay_distribution=False` restores the legacy single-mean-delay path for fast smoke tests and direct comparison.
- **Fidelity dramatically tightened.** Measured at `scale=0.1, seed=42`, 200 ms analysis window after 100 ms burn-in:

  | Population | single-delay ratio | per-conn Gaussian ratio |
  |------------|-------------------:|-----------------------:|
  | L23e | 5.29× | **0.67×** |
  | L23i | 4.78× | **1.19×** |
  | L4e  | 0.83× | 0.68× |
  | L4i  | 2.03× | **1.21×** |
  | L5e  | 3.05× | 1.97× |
  | L5i  | 2.10× | 1.50× |
  | L6e  | 5.23× | 2.81× |
  | L6i  | 2.33× | **1.24×** |

  5/8 populations now sit within 1.2× of Potjans Table 4; the remaining 3 (L4e, L5e, L6e) within 2-3×.
- Cost: per-step ≈ 5× slower (5 sparse mat-vecs per pair instead of 1). At `scale=0.1`, sim wall went 32 s → ~290 s for 600 ms (matches 5× expectation).
- New `tests/test_cortical_column.py::TestPublishedFidelity::test_per_connection_delays_tighten_rates` — asserts ≥ 5/8 populations within `[0.5, 1.5]×` of published Table 4 values. Pins the win.
- `benchmarks/bench_cortical_column.py` now bench BOTH `delay_distribution` modes side-by-side.
- All 29 cortical_column tests pass with the new default (29 passed in 14:18 with delay distribution, 24 deselected-fidelity tests in 4:39 for fast iteration via `-k 'not Fidelity'`).

### PINGCircuit Rust acceleration backend (2026-04-18)
- New Rust per-step kernel `engine/src/ping.rs` with PyO3 wrapper `sc_neurocore_engine.py_ping_step`. Mirrors the Python step semantics (LIF + AMPA / GABA decays + drive + Wiener noise + refractory + spike detect + reset). Noise samples are drawn on the Python side and passed in as `xi_e` / `xi_i` so the per-instance RNG state evolves identically across both backends.
- New `backend=` parameter on `PINGCircuit` (`"auto" | "rust" | "python"`, default `"auto"`). `"rust"` raises `RuntimeError` if the kernel is not built; `"auto"` falls back to NumPy.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_ping_step` so pytest's `bridge/`-on-`sys.path` setup sees the Rust symbol.
- New `tests/test_gamma_oscillation.py::TestPythonRustParity` (6 cases): per-population firing rates within 10 % across (80, 20) / (400, 100) / (1000, 250); dominant FFT peak within 1.5 Hz; explicit `backend="rust"` smoke; invalid-backend rejection. Per-cell membrane V values drift at the float-noise level (NumPy SIMD/FMA vs Rust scalar ordering) — documented inline; aggregate dynamics match.
- `benchmarks/bench_gamma_oscillation.py` extended to bench BOTH backends. Measured speedup: ~3.3-4.3× across the three workload sizes (per-step 145.8 → 33.7 µs at (80, 20); 588.3 → 178.3 µs at (4000, 1000)). All 6 runs stay in the published 30-80 Hz dominant band.
- `engine/src/ping.rs` ships 3 Rust unit tests (no-drive silence; supra-threshold drive + refractory hold; deterministic for identical inputs). All pass on `cargo test --release`.

### CorticalColumn no-multapse experiment — REJECTED (2026-04-18)
- Tried replacing the multapse-with-replacement adjacency builder with a vectorised `argpartition` no-multapse sampler (matching NEST `multapses=False` default). Mean per-target weight is identical between the two approaches and per-target unique connectivity rises from ~63 % to 100 %.
- Measured at `scale=0.1, seed=42`, 600 ms: rates BLEW UP to refractory ceiling for 6 of 8 populations (L23e 90 Hz, L4e/L4i ≈ 410 Hz, L5e/L5i/L6i 260-390 Hz). Pre-experiment multapse-with-replacement gave rates 1.6-7.5× over Potjans Table 4 (within band, just inflated). Post-experiment no-multapse made the divergence ~10× worse.
- Honest finding: at sub-full scale the deterministic per-target in-degree of the no-multapse path amplifies population synchrony in the heavy-recurrent regime (K approaches N_s for several pairs); the multapse path's natural variance dampens this. Documented inline next to the multapse sampler so future contributors don't repeat the experiment without first re-reading van Albada 2015 §3.

### PINGCircuit scale-invariant weight normalisation (2026-04-18)
- `network/gamma_oscillation.py`: per-spike conductance contributions are now divided by source population size at construction (`_w_*_eff = w_* · default_size / actual_size`). The default `(80, 20)` published weights stay bit-identical; larger circuits no longer drift out of the 30-80 Hz band. `bench_gamma_oscillation.py` now reports 40.0 / 41.2 / 41.2 Hz across `(80,20) / (400,100) / (4000,1000)` — all in band — vs 40.0 / 103.8 / 76.2 before the fix. All 19 PINGCircuit tests still pass (default weights and behaviour unchanged at `(80, 20)`).

### Honest benchmark scripts for network/ models (2026-04-18)
- `benchmarks/bench_cortical_column.py`: 3-config wall-clock + per-population firing rates + Potjans Table 4 ratios for `CorticalColumn`. Replaces hand-measured numbers in `docs/api/cortical_column.md` with reproducible JSON output at `benchmarks/results/bench_cortical_column.json`. Honest BLOCKED status reported per backend (Rust/Julia/Go/Mojo) per `feedback_no_fabricated_benchmarks` and `feedback_module_standard_attnres`.
- `benchmarks/bench_gamma_oscillation.py`: 3-workload `step()` wall-clock + dominant gamma frequency check (must lie in 30-80 Hz) for `PINGCircuit`. JSON output at `benchmarks/results/bench_gamma_oscillation.json`. Documents the per-cell LIF + 4 conductance decays as a clean Rust + Mojo target (BLOCKED, tracked under multilang policy). Bench surfaces a real fidelity edge case at `n_e=400, n_i=100` (f_dom=103.8 Hz, outside published 30-80 Hz band) that the default-configuration test does not catch.
- `docs/api/cortical_column.md` performance table updated to reference the bench script and JSON path; numbers replaced with the measured values (build 0.04 / 2.04 / 4.07 s and per-step 0.96 / 2.07 / 5.29 ms across the three configurations).

### Bandit MEDIUM triage (2026-04-18)
- 6 MEDIUM `B307` findings (use of `eval`) → ACCEPT with `# nosec B307` markers and inline rationale: `equation_builder.py` Euler integrator, RK4 derivative eval, threshold expression and reset rule (4 sites); `studio/analysis.py` nullcline grid eval (2 sites). All sites are downstream of `EquationNeuron._validate_expr` AST whitelist (`_ALLOWED_AST_NODES` + `_BLOCKED_NAMES` reject any escape vector before `compile`) with empty-`__builtins__` eval globals.
- Re-running `bandit -r src/ -ll` returns 0 findings.
- 55 LOW findings remain (B101 asserts, B603/B404/B607 subprocess, B110 try/pass, B311 random); informational, no real impact, full inventory in `docs/internal/audit_bandit_2026-04-18.md` and `docs/internal/AUDIT_INDEX.md`.

### CorticalColumn Potjans & Diesmann 2014 (2026-04-18)
- `network/cortical_column.py` rewritten from 5-population canonical-microcircuit reduction to the full 8-population Potjans & Diesmann 2014 model: L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i with per-population sizes from Table 5, the verbatim 8×8 connection-probability matrix from Table 5, per-cell background Poisson drive (`K_bg` per population, `bg_rate=8 Hz`), and exponentially decaying current-based PSCs (`tau_syn=0.5 ms`).
- LIF integration: `C_m=250 pF`, `tau_m=10 ms`, `t_ref=2 ms`, `E_L=V_reset=-65 mV`, `V_th=-50 mV`. Per-source delays: `1.5 ms` (E), `0.8 ms` (I), quantised to `dt`.
- Synaptic weights: `w_e=87.81 pA`, `w_i=-g·w_e` with `g=4` (configurable), `w_l4_to_l23e=2·w_e` per Potjans boost.
- Sparse `scipy.sparse.csr_matrix` adjacency per (target, source) pair with multapses sampled with replacement; full-scale in-degree preservation under `scale_correction=True` (van Albada et al. 2015 protocol).
- `simulate(duration_ms, dt)`, `step(dt)`, `population_rates(rasters, dt, burn_in_ms)`, `total_indegree(target)` and `reset_state()` helpers.
- `tests/test_cortical_column.py` rewritten: 29 tests covering smoke, determinism (per-instance RNG, global-seed leak-proofing), connectivity (Table 5 entries, K_bg, weight signs, L4e→L2/3e boost, sparse adjacency built per pair), and published fidelity (no silent populations, no refractory-ceiling saturation, E/I asymmetry, L4e in band, zero-background silence). 100 % coverage on `cortical_column.py`. Closes #10.
- `docs/api/cortical_column.md` rewritten end-to-end (308 lines): published-reference summary, implementation overview (8 populations, sparse adjacency build, LIF + synapse + refractory, delay handling), public API reference, verification table vs Potjans Table 4 (L4e match within 1 %, other populations within 2-4×), performance table (4.6 s / 19.5 s / 43.6 s wall at scale 0.02 / 0.05 / 0.1) and reference list (Potjans 2014, van Albada 2015, Binzegger 2004, Hahne 2017, Douglas & Martin 2004).

### PINGCircuit conductance-based gamma (2026-04-18)
- `network/gamma_oscillation.py` rewritten from a rate-coded reduced model to per-cell conductance-based Börgers-Kopell 2003 weak-PING. HH-style integrate-and-fire with separate AMPA / GABA exponentially decaying conductances, refractory window, per-cell drive jitter and stochastic kicks. Default parameters reproduce the published 30-80 Hz gamma peak (verified at 40 Hz at the default operating point).
- `population_rate(spike_log, dt, bin_ms)` and `dominant_frequency(spike_log, dt, bin_ms, f_min, f_max)` helpers added; FFT-based with empty-log + out-of-band silence handling.
- `tests/test_gamma_oscillation.py` updated to the new API: 19 tests covering smoke, determinism (per-instance RNG isolation, global-seed leak-proofing), published fidelity (30-80 Hz peak, gain-loop disengage paths, Hz units, silence handling). 100 % coverage on `gamma_oscillation.py`. Closes #11.
- Replaced `np.sum(boolarray)` with `np.count_nonzero(boolarray)` in both implementation and tests to be reload-safe under coverage instrumentation (the `_NoValue` sentinel mismatch otherwise raised `TypeError` from `_methods.py`).

### Repository hygiene (2026-04-18)
- SPDX header format converted from 1-line piped to 2-line form across 2728 source files (.py / .jl / .rs / .go / .mojo). Closes #60.
- `microtubule_neuron.v` Engineer attribution: `Arcane Sapience`.
- `cargo clippy --release --lib`: 20 in-source warnings → 0.
- Bandit HIGH severity in `nas/sc_nas_engine.py:169` → 0 (`hashlib.md5(..., usedforsecurity=False)`).
- Chiplet package coverage 95 % → 100 % (`test_hierarchical_partitioner_perf.py`, `test_chiplet_gen_edge_cases.py`).
- `tools/run_full_cov.sh`: batched per-directory `--cov-append` runner. First full sweep completes at 43.81 % cumulative coverage; no OOM. Closes #58.
- `.gitignore`: `.agent_metadata.json`.
- `ruff`, `rustfmt`: clean across all touched files.

### Chiplet Partitioner — Multi-Language KL Refine (2026-04-18)
- **Perf:** `HierarchicalPartitioner.partition` V=200 went from 963 ms (pre-#65) → 12.7 ms (Python post-fix) → 0.04 ms (Mojo). Total wall-clock improvement at V=200: 24,000× across the chain.
- **#65 fix:** `CorrelationAwareGraph` now caches `(min, max) → edge` lookup → O(1); `_spectral_bisect` hoists `set(vertices)` out of the inner loop. 22-29× speedup at V=50/100/200.
- **#64-prep refine fix:** `_per_partition_cost(v, n_parts, ...)` returns the full length-P cost vector in ONE neighbour scan (was P redundant scans). Additional 2-9× over #65; bit-identical canonical output.
- **#74 multi-language KL refine:** Rust (`engine/src/partition.rs`), Julia (`accel/julia/chiplet/kl_refine.jl`), Go (`accel/go/partition/partition.go`), Mojo (`accel/mojo/partition/partition.mojo`) all wired into `HierarchicalPartitioner(refine_backend=...)`. Bit-exact `part_map` parity verified end-to-end via dispatcher tests on V=100. Empirical fastest-pick at V=1000: Mojo 0.20 ms (351×), Julia 0.26 ms (270×), Rust 0.29 ms (242×), Go 0.68 ms (103×), Python 70 ms.
- **Bench harness:** `benchmarks/bench_kl_refine.py` runs 5 backends with parity check; results in `benchmarks/results/bench_kl_refine.json`.
- **Tests:** 218 chiplet tests (39 new this batch); coverage 99.58 % on the chiplet package, with `chiplet_gen.py` at 100 % and `hierarchical_partitioner.py` at 99 %.

### LGSSM Multi-Language Acceleration (2026-04-17)
- **Mojo LGSSM Kalman filter** (`accel/mojo/world_model/lgssm.mojo`): hand-rolled matmul + Cholesky + triangular solve via `mojo build --emit shared-lib`. **46× over Python, 8× over Rust** at T=200 d=4 p=3 workload. Closes #69.
- **Go LGSSM** (`accel/go/lgssm/lgssm.go`): cgo + ctypes shared lib, hand-rolled Cholesky. Closes #70.
- **Julia LGSSM** (`accel/julia/world_model/predictive_model.jl`): juliacall + LinearAlgebra LAPACK. Closes #68.
- **Rust LGSSM** (`engine/src/lgssm.rs`): PyO3 + ndarray Cholesky. Closes #67.
- All 4 backends dispatched via `KalmanFilter.filter(backend='auto'|'rust'|'julia'|'go'|'mojo'|'python')`; bit-exact parity vs Python at atol≤1e-9 on means/covs, ≤1e-7 on log-likelihood.
- **Mojo 0.26 FFI pattern proven:** raw `Int` address via `arr.ctypes.data` + `UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)` reconstruction inside the `@export` body works around the parametric-signature restriction. Same pattern reused for fault_injection + KL refine.

### Fault Injection Multi-Language (2026-04-17)
- **Rust + Julia + Go + Mojo** kernels for the 5 fault models (`bitflip`, `stuck_at_0/1`, `dropout`, `gaussian`). Mojo wins 4/5 boolean kernels (2.7-8.2× over NumPy); Julia wins Gaussian via Ziggurat randn. Bench harness with 4σ Binomial parity at `benchmarks/bench_kl_refine.py`-style 5-backend layout.

### Bench Harness Honest Exemptions (2026-04-17)
- `bench_safety_monitor.py` + `bench_chiplet.py` now emit a `backends` block in the JSON output documenting USED / EXEMPT / BLOCKED-ON-#X status per backend per op, with explicit FFI-vs-compute math instead of silent skipping.

### Cross-Module Integration — (2026-04-16)
- **Shared Core Types** `core/types.py`: unified `HardwareBudget`, `ResourceReport`, `LayerSpec`, `estimate_network()` — single source of truth for Optimizer↔NAS↔Runtime
- **Closed-Loop Adaptive Controller** `control/adaptive_loop.py`: Runtime drift detection → SA re-optimisation → new `RuntimeConfig`, configurable cooldown/threshold
- **Unified Energy Reporter** `energy_accounting/unified_reporter.py`: bridges `CarbonModel` + `ThermalModel` + ASIC power into single `analyze()` call
- **End-to-End Export Pipeline** `export/pipeline.py`: Model Zoo → ONNX → TVM Relay → MLIR/SSA → SystemVerilog in one `run()` call
- **Rust Wiring**: `sc_optimizer.py` → `optimizer.rs` SA engine, `sc_nas_engine.py` → `evo.rs` tournament selection, `photonic_emitter.py` → `photonic.rs` crosstalk analysis
- **Package Exports**: Updated `core/__init__.py`, `control/__init__.py`, `export/__init__.py` with new module exports
- **Integration Tests**: 20 new tests in `tests/test_integration/test_cross_module.py` covering all 5 actions
- **Maturin**: Rebuilt `sc_neurocore_engine` v3.14.0 with all Rust bindings
- **Total**: 10,592 tests (8,895 Python + 1,697 Rust) — ALL GREEN

### Extended Rust Wiring — QA & DNA Bridges (2026-04-17; timing superseded)
- **Quantum Annealing**: `bridges/quantum_annealing.py` →
  `py_qa_simulated_annealing`. The historical speed figures recorded here were
  not promotion-grade evidence and are superseded by the 2026-07-12
  source-bound rerun requirement; do not quote them.
  - `IsingModel.energy()` → `py_qa_ising_energy` (Rust path for n>20 qubits)
  - `SimulatedAnnealer.solve_ising()` → `py_qa_simulated_annealing` (native
    path retained; speed must be remeasured on the target release host)
  - `EnergyLandscape.analyze()` → `py_qa_batch_ising_energy` (batch energy for >100 samples)
- **DNA Mapper**: `bridges/dna_mapper.py` — Rust engine loaded (`_HAS_RUST_DNA`)
  - Imported: `py_dna_design_sequence`, `py_dna_detect_hairpins`, `py_dna_check_cross_hybridization`, `py_dna_simulate_kinetics`, `py_dna_design_orthogonal_set`
- **Photonic**: Fixed `py_ph_analyze_crosstalk` API (channel_ids, wavelengths, bandwidths, powers)

### Python vs Rust Benchmarks — Integration Hot Paths (2026-04-16)
- **SA Optimizer**: 7× (5 layers) → 36× (20 layers) → **47× (50 layers)**
- **Tournament Selection**: **337–394×** (amortised per-round overhead elimination)
- **Batch Mutate**: 17–21× across population sizes 50–1000
- **Population Diversity**: 34–**90×** (O(N²) SIMD pairwise distance)
- **Historical aggregate**: the former mean included superseded QA timings and
  must not be treated as a current cross-path claim
- **QA timing**: superseded; rerun
  `bench_quantum_annealing_rust_vs_python.py` on the target release host before
  making a native speed claim
- **E2E Pipeline**: NAS→Optimizer→Energy→Verilog in **13.7ms** (small) to **116ms** (large)
- Criterion (Rust-native): spike_times=83ns, firing_rate=13ns, ISI=96ns, van_rossum=1.2µs (N=100)
- Results: `benchmarks/results/py_vs_rust_integration.json`
- Script: `benchmarks/py_vs_rust_benchmark.py`

### Cross-Language Acceleration — Spike Stats (2026-04-16)
- **Crate** `spike_stats_core` (v0.1.0): 16 functions, 28 Rust tests, PyO3 + Criterion
- **Distance** (7 fns): `victor_purpura_distance` **181×**, `spike_sync` **31×**, `hunter_milton` **27×**, `van_rossum`, `spike_distance`, `earth_movers_distance`, `multi_neuron_victor_purpura` **160×**
- **Correlation** (5 fns): `cross_correlation`, `event_synchronization`, `spike_time_tiling_coefficient`, `coincidence_index`
- **Variability** (4 fns): `approximate_entropy` **73×**, `sample_entropy` **78×**, `lempel_ziv_complexity` **69×**, `permutation_entropy` **65×**
- 99/99 Python tests pass on both Rust and Python fallback paths
- Python dispatch wired in: `distance.py`, `correlation.py`, `variability.py`

### Cross-Language Acceleration —  Stochastic Doctor (2026-04-16)
- **PyO3 bindings** for `stochastic_doctor_core` crate: `py_scc_bytes`, `py_scc_batch`, `py_precision_bytes`, `py_histogram`, `PyDriftDetector`
- Replaced legacy `ctypes.CDLL` with PyO3 import pattern (primary), Python fallback (secondary)
- `SC_NEUROCORE_NO_RUST=1` env var forces Python path
- 16/16 Python tests pass on both Rust and Python paths
- 23 Rust tests pass
- **Benchmarks** (SCC single-pair): 35× at N=100, 3.5× at N=1M
- **Benchmarks** (batch SCC N×N): 15–18× for 4–64 neuron layers
- **Benchmarks** (precision): 5–14× across all sizes
- Criterion benchmarks: `crates/stochastic_doctor_core/benches/doctor_bench.rs`
- Python benchmark: `benchmarks/stochastic_doctor_benchmark.py`
- Results: `benchmarks/results/stochastic_doctor_py_vs_rust.json`
- API docs updated with full benchmark tables: `docs/api/stochastic_doctor.md`

### Module Integration — 19 Industrialized Modules (2026-04-16)
- **Industrial tier:** safety_cert (IEC 61508/ISO 26262, 81 tests), asic_flow (multi-PDK, 67 tests), fault_injection (radiation-grade, 22 tests), uvm_gen (UVM testbench, 71 tests)
- **Exascale tier:** hypervisor (multi-tenant, 78 tests), digital_twin/twinsync (time-warp sync, 72 tests)
- **Substrates tier:** spintronic (MTJ mapper, 66 tests), chiplet (UCIe/BoW, 94 tests), memristor (crossbar, 70 tests), analog_bridge (SC-to-analog, 27 tests)
- **Frontiers tier:** evo_substrate (self-replicating evolution, 91 tests), meta_plasticity (self-modifying rules, 72 tests), bioware (organoid interface, 79 tests), federated (DP-SGD, 93 tests), bci_studio (closed-loop BCI, 32 tests)
- **Unification tier:** explainability (causal attribution, 71 tests), neuro_symbolic (predictive coding, 34 tests), stochastic_doctor (bitstream diagnostics, 16 tests), model_zoo (auto-Verilog, 37 tests)
- All modules: SPDX dual-license headers, `__tier__` classification, `__init__.py` with docstrings
- 19 MkDocs API doc pages with `mkdocstrings` directives
- Updated `mkdocs.yml` nav with 5 new categories (Industrial, Substrates, Exascale, Frontiers, Unification)
- Integration reference: `docs/MODULE_INTEGRATION.md`
- Total: **1,173 new Python tests** from integrated modules

### Rust Workspace — 5 Research Crates Integrated (2026-04-16)
- Created `crates/` directory for research Rust crates
- Integrated: tinysc_riscv (83 tests), core_engine (22 tests), autonomous_learning (12 tests), neuro_symbolic (28 tests), stochastic_doctor_core (23 tests)
- Root `Cargo.toml` workspace now has 6 members (engine + 5 research crates)
- Engine (`sc_neurocore_engine`, 1,549 tests) verified undamaged after workspace expansion
- Total: **1,717 Rust tests** across 6 crates

### Evolutionary Substrate — (2026-04-16)
- `FormalSafetyGuard`: pre-deployment safety validation
- `CPPNGenome`: Compositional Pattern Producing Network developmental encoding
- `IslandModel`: multi-deme evolution with migration
- `NoveltyArchive`: k-NN behavioural novelty search
- `HWFitnessCollector`: FPGA execution feedback for hardware-in-loop fitness
- `ParetoFront`: NSGA-II style non-dominated sorting
- `TournamentSelector`, `AgeRegulator`, `BloatPenalizer`, `ExtinctionDetector`, `CoevolutionArena`
- `EvoStatisticsTracker`, `ComplexityTracker`, `genome_diff()`, `shared_fitness()`
- Module grew from 657 to 1,400 LOC, 42 to 91 tests

### Foundation-Model Neural Decoders (2026-04-07)
- **POYODecoder**: spike tokenisation + cross-attention (Azabou et al. 2023 NeurIPS)
- **POSSMDecoder**: diagonal SSM with HiPPO-LegS init (Ryoo et al. 2025 ICLR)
- **NDT3Decoder**: causal masked self-attention on binned spikes (Ye & Pandarinath 2025)
- **CEBRAEncoder**: InfoNCE contrastive embedding with analytical backprop (Schneider et al. 2023 Nature)
- Rust acceleration: tokenise_spikes, sinusoidal_position_encode, scaled_dot_product_attention, gaussian_attention, ssm_step_diagonal, infonce_loss (6 pub fn, 11 tests)
- PyO3: 5 functions registered
- Tests: 47 multi-angle tests
- Documentation: 976 lines, 8/8 sections

### Transcriptomic Foundation Model Interfaces (2026-04-07)
- **ScKGBERTInterface**: dual S-Encoder + K-Encoder with Gaussian attention (Li et al. 2025 Genome Biology)
- **GeneformerInterface**: rank-value tokenisation + multi-head attention + MLM (Theodoris et al. 2023 Nature)
- **rank_value_encode**: shared utility for gene expression tokenisation
- Tests: 29 multi-angle tests
- Documentation: 1,118 lines, 8/8 sections

### Gap Model Python + PyO3 + Docs (11 models, 2026-04-07)
- 10 new Python implementations (publication-exact): AdaptiveThresholdMoENeuron, HybridLinearAttentionNeuron, QuantumInspiredLIFNeuron, DendriticNMDANeuron, MulticompartmentMCNNeuron, AstrocyteLIFNeuron, DirectionSelectiveRGC, CochlearHairCell, ShortTermPlasticitySynapse, DopamineStdpSynapse
- PyO3 wiring: 11 models registered (2 macro + 9 manual wrappers)
- Tests: 87 multi-angle tests
- 10 docs (5,701 lines total)
- GPU backend documentation (607 lines)

### CI & Dependency Fixes (2026-04-07)
- **PEP 639**: migrated `license = { text = "..." }` → `license = "AGPL-3.0-or-later"` (fixes setuptools ≥78)
- **mypy**: 1.19.1 → 1.20.0
- **cyclonedx-bom**: 7.2.2 → 7.3.0
- **ci.yml**: pinned all mypy stub dependencies to exact versions (CodeQL #287)
- **cargo fmt**: applied to all new Rust code
- Purged 52 resolved failed/cancelled CI runs
- Closed superseded dependabot PRs #53, #55

### Neuron Models — (12 new models, 2026-04-04/05)
- **TUMNetwork**: rate model with short-term plasticity (depression + facilitation), 3 ODEs
- **ElBoustaniNetwork**: E/I + NMDA bistability, 3 ODEs
- **GradedSynapseNeuron**: non-spiking, passive RC + sigmoid release
- **GapJunctionNeuron**: LIF + electrical synapse with Cx36 rectification
- **FrankenhaeUserHuxleyAxon**: GHK permeability-based currents (not linear V-E)
- **NodeOfRanvier**: MRG 2002 — Nav1.6 transient + persistent + Kv7 slow K
- **MyelinatedAxon**: MRG node + passive internode cable
- **CardiacPurkinjeFibre**: DiFrancesco-Noble 1985, 6 currents
- **SmoothMuscleCell**: CaL + BK + IP3R/SERCA + Ca²⁺ store
- **EndocrineBetaCell**: CaL + K_dr + K_ATP + K_Ca glucose-dependent bursting

### Fidelity Audit Fixes (7 models corrected, 2026-04-04)
- **RetinalGanglionCell**: basic LIF → Pillow 2005 GLM (stimulus + history filters)
- **InnerHairCell**: no vesicle pool → Meddis 1986/2006 (q/c/w compartments)
- **OuterHairCell**: unidirectional sigmoid → bidirectional asymmetric prestin (Santos-Sacchi 2006)
- **GranuleCell**: LIF-style → D'Angelo 2001 full HH (7 ionic currents)
- **AlphaMotorNeuron**: PIC no inactivation → h_pic + Ca²⁺ buffering
- **RodPhotoreceptor**: no Ca²⁺ feedback → Ca²⁺-GC feedback (Nikonov 2006, Hill n=4)
- **TraubMilesNeuron**: missing M-current → Kv7/KCNQ (Yamada 1989)

### Kinetics Audit Fixes (3 models upgraded, 2026-04-05)
- **GolgiCell** (CRITICAL): 5-current WB → full Solinas 2007 (11 currents, 13 gating variables)
- **DCNNeuron** (MODERATE): added persistent Na (INaP) + Ca²⁺-dependent AHP (7 currents total)
- **OlfactoryReceptorNeuron** (MODERATE): added PDE4 negative feedback on cAMP

### Infrastructure (2026-04-05)
- `supported_models()`: 28 missing entries added (159 total)
- Interface wrappers: 20 non-standard models wired via Wr* types (multi-input, i32-input, graded/rate)
- All 4 failing CI workflows fixed (clippy, ruff, MkDocs, typos)
- `cargo fmt` applied to all engine source
- Fresh Criterion benchmarks published (2026-04-05)
- Documentation audit: all stale numbers corrected across README, pricing, index, benchmarks

### Notebooks (13 new, 21 total)
- **08_equation_to_verilog**: ODE string → Python sim → Q8.8 Verilog (LIF, FHN, Izhikevich)
- **09_topology_and_dynamics**: 6 generators, adjacency matrices, degree distributions, raster plots
- **10_spike_train_analysis**: ISI, CV, Fano, cross-correlation, van Rossum, PCA
- **11_biological_circuits**: tripartite synapse Ca²⁺ dynamics, Rall dendrite nonlinearity
- **12_learning_rules**: STDP, e-prop eligibility, R-STDP, STP facilitation/depression
- **13_quantisation_pipeline**: float → Q8.8 → SC probabilities → Verilog export, error budget
- **14_sc_arithmetic_theory**: AND=multiply, XNOR=bipolar, MUX=add, CORDIV=divide, Sobol vs Bernoulli convergence, Hoeffding bounds
- **15_fault_tolerance**: SC vs fixed-point under bit-flips/stuck-at, TMR majority vote
- **16_neuron_atlas**: 12 models from 8 families (LIF→ArcaneNeuron, 1907–2026)
- **17_reservoir_computing**: liquid state machine, temporal XOR, ridge readout, SVD dimensionality
- **18_mixed_precision_sc**: per-layer adaptive L, Hoeffding vs sensitivity allocation, Pareto frontier
- **19_compression_and_pruning**: magnitude/SC-aware pruning, quantisation sweep, combined Pareto
- **20_power_analysis**: event-driven vs clock-driven toggle count, scaling with network size
- **21_spike_alu**: Turing-complete spike-based ALU — logic gates, SR latch register, ripple-carry adder, sort
- **22_ir_type_safety**: IR signal type checker — Bitstream/Rate/Spike/Fixed, catch mismatches before Verilog synthesis
- **23_topological_observables**: winding number, Ollivier-Ricci curvature, sheaf consistency defect, connection curvature
- **24_identity_lazarus**: Lazarus checkpoint save/load/merge, TraceEncoder text→spikes, StateDecoder attractor extraction, DirectorController L16 self-regulation
- **25_cortical_column_dynamics**: canonical 5-population microcircuit, thalamic drive, layer-resolved rasters, feedforward latency
- **26_spike_codec_benchmark**: 5 codecs (ISI/AER/predictive/delta/streaming) on synthetic data, compression ratio vs density curves
- **27_python_to_proven_silicon**: complete end-to-end pipeline — ODE string → Python sim → IR type check → Q8.8 Verilog → testbench → formal properties → resource estimate
- **28_domain_bridge**: TensorStream prob↔bitstream↔quantum conversions, QuantumStochasticLayer cos²(θ/2) non-linearity, Born rule roundtrip

### Tests (19 new files, ~3700 lines, ~310 test methods)
- `test_topology_generators.py`: 6 generators — CSR validity, degree, symmetry, edge count, determinism
- `test_cordiv_division.py`: CORDIV accuracy, monotonicity, convergence, adaptive_length Hoeffding bounds
- `test_fault_injection.py`: bit-flip degradation, stuck-at analytical bounds, TMR, SC vs fixed-point comparison
- `test_learning_advanced.py`: EligibilityTrace decay, BPTT/TBPTT loss, R-STDP reward gating, STP facilitation/depression/recovery
- `test_quantisation_pipeline.py`: Q8.8 roundtrip, dequantise fidelity, SC probability ordering, dot product end-to-end
- `test_network_monitors_stimulus.py`: SpikeMonitor record/count/trains, StateMonitor accumulation, RateMonitor bins, TimedArray clamp, StepCurrent onset/offset, PoissonInput rate/seed/weight
- `test_neuron_families.py`: parametrised test across 11 EquationNeuron models — step(), spike detection, reset, state finiteness, determinism
- `test_sc_convergence.py`: AND O(1/√L), Sobol faster than Bernoulli, CORDIV monotonic, correlation violation, popcount exact
- `test_spike_alu.py`: SpikeGate truth tables (AND/OR/NOT/NAND/XOR), De Morgan law, SpikeRegister roundtrip, SpikeALU add/sub/xor/compare/shift, spike_sort correctness
- `test_topological_observables.py`: winding number wraps, Ricci curvature complete>ring, sheaf defect zero when synchronised, connection curvature bounded by coupling
- `test_scpn_integrated.py`: K_nm symmetric zero-diagonal, OMEGA_N physical frequencies, create_full_stack 16 layers, run_integrated_step finite, get_global_metrics
- `test_identity_lazarus.py`: IdentitySubstrate run/step/health, TraceEncoder encode/determinism, Checkpoint save/load/merge roundtrip, StateDecoder patterns/attractors, DirectorController monitor/diagnose/correct
- `test_cortical_column_dynamics.py`: CorticalColumn step/run dict outputs, 5 populations, binary spikes, thalamic drive, L4-before-L5, inhibition, reset, determinism
- `test_codec_roundtrip.py`: all 5 codecs parametrised — lossless roundtrip (sparse/empty/single-spike/all-ones), compression ratio bounds, shape preserved, edge cases (1 channel, 1 timestep)
- `test_tensor_stream.py`: TensorStream prob↔bitstream↔quantum roundtrips, Born rule, normalisation, p=0/1 edge cases, invalid conversion raises
- `test_quantum_hybrid.py`: QuantumStochasticLayer cos²(θ/2) transfer, p=0→1, p=1→0, monotonic decreasing, multi-qubit independence

### Model Validation
- LIF f-I curve: 29/29 tests, <5% error vs analytical solution
- Izhikevich 20 firing patterns: all from Izhikevich (2003) Table 1 validated
- Hodgkin-Huxley 1952: AP peak 40.6mV, spike width 1.46ms, AHP -75.1mV
- NeuroBench SHD: 79.28% test accuracy (250K params, feedforward)
- Brian2 parity: exact LIF match (0.000ms timing diff), 7.3x speedup
- 5 validation docs with measured data in `docs/validation/`

### Stochastic Computing Pipeline
- Bipolar SC (XNOR): `core/bipolar.py` for signed weight multiplication
- SC bitstream MNIST: 10% (unipolar) -> 35.6% (bipolar) -> 50.0% (all fixes)
- SC-aware training: `SCAwareLIFNet` with bitstream noise injection (+9.5pp)

### Quantization-Aware Training
- `QuantizedLIFNet`: 2/4/8/16-bit STE weight quantization (PyTorch)
- `SCAwareLIFNet`: SC noise injection during training
- `SCAwareLinear`: drop-in layer replacement

### Encoding Comparison
- 7 temporal spike encodings benchmarked on MNIST
- Latency encoding Pareto-optimal: 88.1% at 142 spikes (17x fewer than rate)

### Interoperability
- NeuroML 2 importer: iafCell, Izhikevich (2003/2007), AdEx
- SONATA network format importer: nodes.h5 + edges.h5, connectivity matrix

### Reproducibility
- 7 Kaggle scripts in `notebooks/*_kaggle.py`
- JSON artifacts in `benchmarks/results/`

## [3.14.0] — 2026-03-27

### Visual SNN Design Studio (Experimental)
- **New feature:** web-based IDE for designing, training, compiling, and deploying SNNs
- 118-model browser with live simulation, parameter sliders, pattern classification
- 20+ analysis views: trace, phase, ISI, f-I, bifurcation, heatmap, sensitivity, STA, frequency response, characterisation, multi-model overlay, A/B comparison
- Compiler Inspector: SC IR build/verify/emit, SystemVerilog generation, co-simulation
- Synthesis Dashboard: Yosys synthesis for 4 FPGA targets (ice40, ECP5, Gowin, Xilinx), multi-target comparison, resource estimation without Yosys
- Training Monitor: live SSE metric streaming, 6 surrogate gradients, per-layer spike rates, learnable beta/threshold
- Network Canvas: React Flow drag-and-drop populations and projections, NIR export/import
- Full pipeline: network graph → validate → simulate → compile → synthesise in one click
- Project save/load: persistent JSON workspaces on server
- E-I balanced network simulation with Rust engine fast path
- 140+ Studio-specific tests
- Documentation: 7 pages on GitHub Pages, 10-step quickstart tutorial
- Launch: `pip install sc-neurocore[studio] && sc-neurocore studio`

### Rust Engine
- `py_simulate_ei_network()`: fused E-I network simulation (CSR + Poisson + Euler) in single Rust call
- `py_batch_simulate()`: batch model simulation with NeuronVariant dispatch loop
- `create_neuron()` made `pub` for reuse across lib.rs
- 288 Rust tests passing

### Performance
- Model list caching: first `/api/models` call loads 118 models in ~1s, subsequent calls <1ms

### Security
- 25 CodeQL "information exposure through exception" fixes — no tracebacks in HTTP responses
- 5 CodeQL "uncontrolled data in path expression" fixes — project name sanitisation
- DOMPurify XSS fix via npm override (>=3.3.2)
- Bandit: MD5 usedforsecurity=False, narrowed bare except clauses

### CI
- Engine wheel publish job added to publish.yml (PyPI OIDC)
- Bridge ImportError restored for pytest.importorskip compatibility
- PnR added to typos dictionary
- tsconfig.tsbuildinfo gitignored
- uvicorn skip guard for studio optional extra

### ANN-to-SNN Conversion Engine
- `sc_neurocore.conversion.convert()`: automated PyTorch ANN to rate-coded SNN conversion
- QCFS activation (Quantization-Clip-Floor-Shift): ReLU replacement for conversion-aware training
- Threshold normalization from calibration data activation statistics
- `ConvertedSNN.run()` and `.classify()` for inference with Poisson rate coding

### Learnable Delay Training
- `DelayLinear`: PyTorch module with trainable per-synapse delays via linear interpolation
- Differentiable delays: gradients flow through fractional delay positions
- Export to integer delays for hardware deployment via `delays_int` and `to_nir_delay_array()`
- DCLS (Dilated Convolutions with Learnable Spacings) principle for fully-connected SNN layers

### One-Command FPGA Deploy
- `sc-neurocore deploy model.nir --target artix7`: NIR/PyTorch → Verilog → project in one command
- Target presets: ice40, ecp5 (Yosys Makefile), artix7, zynq (Vivado project.tcl)
- Copies 19 HDL library modules, generates neuron SystemVerilog, build script, README

### Network Engine
- Per-synapse delays in Projection: `delay=array` for heterogeneous axonal/synaptic delays
- Spike-gating: `Population.step_all(spike_gating=True)` skips idle neurons, compute proportional to active count
- Weight sparsity: `Projection(weight_threshold=0.01)` skips near-zero synapses during propagation

### Compiler
- Per-layer adaptive bitstream length: `assign_lengths()` with Hoeffding or sensitivity-based allocation
- Mixed-precision SC networks: shallow layers use short L (fast), deep layers use long L (precise)

### Event-Driven FPGA RTL
- `sc_aer_encoder.v`: spike vector → AER packets via priority encoder, idle neurons consume zero power
- `sc_event_neuron.v`: Q8.8 LIF that computes only on input events or periodic leak ticks
- `sc_aer_router.v`: distributes AER events to target neurons using connectivity lookup table
- Total HDL modules: 19 (was 16)

### Performance
- Lazy-load 109 neuron models: import time 200s → 57s
- Deferred scipy imports (stats.qmc, sparse): import time 57s → 10s

### Infrastructure
- Coverage fixes: test second model access, pragma Rust-only branch
- Coverage for lazy-load path, sparse guard mock path
- Ruff F401 re-export fixes, format vectorized_layer

## [3.13.3] - 2026-03-20

### SC Arithmetic
- CORDIV division circuit: Python `sc_divide()` + Verilog `sc_cordiv.v` (Li et al. 2014)
- Adaptive bitstream length: Hoeffding/Chebyshev/variance bounds via `adaptive_length()`
- Sobol/Halton multi-dimensional decorrelation for per-synapse independent streams
- Chaotic RNG mode in BitstreamEncoder (logistic map)
- Sobol bitstream attention: `StochasticAttention.forward_bitstream()` with LDS variance reduction

### Learning Rules
- BCM metaplasticity with sliding threshold (Bienenstock-Cooper-Munro 1982)
- Voltage-based STDP (Clopath et al. 2010)
- Truncated BPTT for long sequences (`TBPTTLearner`, Williams & Peng 1990)
- EWC penalty implemented (was no-op stub) — Kirkpatrick et al. 2017
- Learnable beta/threshold on all 10 SNN cell types (ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF, IF, Synaptic)
- ConvSpikingNet now works with `train_epoch()` via `flatten_input=False`

### Biological Circuits
- Tripartite synapse: astrocyte ↔ synapse bidirectional coupling (Araque et al. 1999)
- Rall branching dendrite: compartmental tree with 3/2 power rule
- Canonical cortical microcircuit: 5-population column (L2/3 exc/inh, L4, L5, L6)
- Astrocyte adapter: `AstrocyteNeuron` wraps Li-Rinzel model for Population/Network

### Theoretical Depth
- SC→quantum circuit compiler: Ry encoding, statevector simulator, layer compilation
- Zero-multiplication predictive coding SC layer (Conjecture C9: XOR=error, popcount=magnitude)
- Topological observables: winding number, Ollivier-Ricci curvature, sheaf defect
- Phi* integrated information estimation (Barrett & Seth 2011, IIT)
- Goldstone mode verification for Knm coupling spectrum
- Fault tolerance benchmark: SC vs fixed-point degradation curves
- Hardware-aware SC layer with memristive defect injection
- Noisy quantum simulation via HeronR2NoiseModel Kraus channels

### NIR Bridge
- Recurrent edge handling via unit-delay insertion (LSTM-like feedback)
- Multi-port subgraph support (`SCMultiPortSubgraphNode`)

### Compiler
- IR type checker: Bitstream/Rate/Spike mismatch detection before emission
- SV/MLIR emission for GraphForward, SoftmaxAttention, KuramotoStep (was error stub)
- Weight quantizer exported in compiler `__init__.py`

### Hardware Stack
- AXI-Stream interface for bulk bitstream I/O (`sc_axis_interface.v`)
- DMA controller for weight upload and output readback (`sc_dma_controller.v`)
- Parameterized AXI-Lite register file (`sc_axil_cfg_param.v`)
- Clock domain crossing primitives: 2-FF sync, Gray counter, async FIFO (`sc_cdc_primitives.v`)
- NEON scalar-equivalence tests (13 tests for popcount, dot, max, sum, scale)

### Infrastructure
- Rust engine wheel publishing in PyPI release workflow
- SpikeInterface/Neo adapter for experimental data import
- Static CycloneDX SBOM (v1.6)
- JAX autodiff fix: straight-through estimator for spike reset
- IIT added to typos allowlist

## [3.13.2] - 2026-03-19

### Equation → Verilog RTL Compiler
- `equation_compiler.py`: compile any `EquationNeuron` to synthesizable Q8.8 fixed-point Verilog
- `equation_to_fpga()`: one-liner from Brian2-style ODE string to Python neuron + Verilog RTL
- AST-to-Verilog expression emitter handles +, -, *, /, **, unary minus, comparisons
- Multi-variable ODE support (FitzHugh-Nagumo, Izhikevich, Hodgkin-Huxley)
- Threshold and reset logic auto-generated

### NIR Bridge
- `nir_bridge` package: import NIR graphs into SC-NeuroCore (FPGA backend for NIR)
- Maps 11 NIR primitives (LIF, IF, LI, Integrator, Affine, Linear, Scale, Threshold, Flatten, Input, Output)
- Recursive graph parser with topological sort, fan-in summation, nested subgraph support
- NIR integration guide, API docs, notebook (05_nir_bridge.ipynb)

### Packaging & Release
- Restored `sc-neurocore` as the only PyPI product package and removed the unintended runtime dependency on a separate `sc-neurocore-engine` publish
- Publish automation now pushes only `sc-neurocore` to PyPI while keeping the Rust engine on the existing crate / source / CI wheel paths
- Tag pushes still trigger publish directly, so release creation no longer depends on a downstream `release.published` event

## [3.13.1] - 2026-03-19

### Packaging & Install
- Top-level `sc-neurocore` now requires the matching `sc-neurocore-engine` release, and `sc-neurocore info` reports engine version mismatches explicitly instead of silently mixing versions
- Dense-layer example and getting-started/docs packaging guidance now match the current public API and distinguish wheel-shipped modules from source-only modules

### NIR Bridge
- Nested NIR subgraphs now execute through a dedicated subgraph node wrapper and reset cleanly inside `SCNetwork`
- `Flatten` now respects `start_dim` / `end_dim`, and bridge coverage is enforced instead of being omitted
- Added regression coverage for nested graphs, fan-in, cycle detection, orphan nodes, flatten edge cases, and file-based import/export

### CI & Release
- CI now builds and installs the local engine wheel before editable/package installs, so unreleased versions no longer fail dependency resolution
- Build smoke installs both the engine wheel and the top-level wheel from local artifacts
- Publish workflow now runs from tag pushes, builds engine sdist+wheels, publishes the engine package before `sc-neurocore`, and keeps manual dispatch build-only unless publish is explicitly enabled
- Release workflow now attaches both the pure-Python wheel and sdist to GitHub Releases

### Bug Fixes
- StochasticTransformerBlock: clamp residual and FFN intermediate values to [0, 1] — MAC output from `VectorizedSCLayer` can exceed 1.0, triggering the new input validation
- Optional dependency introspection in `sc-neurocore info` no longer crashes on broken NumPy/JAX imports

### Tests
- Full preflight now passes at `2112 passed`, `38 skipped`, `12 xfailed`, with `100.00%` coverage
- Added audit validation tests for VectorizedSCLayer/EquationNeuron, CLI fallback coverage, dense-layer example smoke coverage, and expanded NIR bridge regressions

### Documentation
- Replace stale black references with ruff format in `VALIDATION.md` and `CONTRIBUTING.md`
- Sync the packaging/install docs with the released product surface
- Package naming and install guidance were corrected in `3.13.2`; `3.13.1` incorrectly treated `sc-neurocore-engine` as a separate PyPI runtime dependency

## [3.13.0] - 2026-03-18

### Python 3.14 Support
- CI test matrix, wheel builds, and publish workflow now include Python 3.14
- All 1 776 Python tests pass on 3.14; all dependencies compatible
- pyproject.toml classifier added

### Bridge Wiring
- 12 missing Rust symbols exported from bridge `__init__.py`: NetworkRunner, BitstreamAverager, Izhikevich, ArcaneNeuron, 8 AI-optimized models, ContinuousAttractorNeuron
- Parity test name mapping for RustContinuousAttractorNeuron

### CI Fixes
- Black formatting for identity/ files; pre-commit ruff upgraded v0.9.7 → v0.15.6
- Clippy: PopulationRunner::is_empty() added
- TraceEncoder: deterministic hash (byte-based, not Python hash())
- Synapse test tolerance widened for short bitstream noise
- Notebook trailing newline for end-of-file-fixer
- Removed deleted ruff rule UP038

### Documentation
- JOSS paper rewrite: pipeline + spike raster figures, Availability section, McCulloch-Pitts/Hodgkin-Huxley citations, tightened to ~1200 words
- All docs synced: test counts (1 776/336), 111 NetworkRunner, 17 HDL, Python 3.14
- Neuron explorer notebook (04_neuron_explorer.ipynb): 5 sections, 117 models

### Infrastructure
- `.gitattributes`: eol=lf (suppress CRLF warnings on Windows)
- Single-directory migration: `03_CODE/sc-neurocore/` is canonical repo
- PyPI deployment branch policy fixed (main added)
- 12 known Rust/Python parity divergences tracked as xfail
- 5 version-gate assertions updated

## [3.12.0] - 2026-03-17

### ArcaneNeuron + 8 AI-Optimized Models
- ArcaneNeuron: unified self-referential cognition model with 5 coupled subsystems (fast/working/deep/gate/predictor)
- 8 novel AI-optimized spiking neuron models: MultiTimescaleNeuron, AttentionGatedNeuron, PredictiveCodingNeuron, SelfReferentialNeuron, CompositionalBindingNeuron, DifferentiableSurrogateNeuron, ContinuousAttractorNeuron, MetaPlasticNeuron
- Total neuron count: 122 Python (113 bio + 9 AI), 111 Rust (including Arcane)
- ArcaneNeuron included in Rust NetworkRunner (111-model fused loop, was 80)

### Identity Substrate
- `sc_neurocore.identity` package: persistent spiking network for identity continuity
- IdentitySubstrate: 3-population network (HH cortical + WB inhibitory + HR memory) with STDP
- TraceEncoder: LSH-based reasoning trace to spike pattern encoding
- StateDecoder: PCA + attractor extraction + priming context generation
- Checkpoint: Lazarus protocol save/restore/merge of complete network state (.npz)
- DirectorController: L16 cybernetic closure with monitor/diagnose/correct feedback loop

### Network Simulation Engine
- Population-Projection-Network architecture with 3 backends: Python (NumPy), Rust (NetworkRunner), MPI (mpi4py)
- 6 topology generators: random, small-world, scale-free, ring, grid, all-to-all
- 12 visualization plots: raster, voltage, ISI, cross-correlogram, PSD, firing rate, phase portrait, population activity, instantaneous rate, spike train comparison, network graph, weight matrix
- 7 advanced plasticity rules: BPTT, e-prop, R-STDP, MAML, homeostatic, STP, structural
- MPI distributed simulation for billion-neuron scale via mpi4py

### Rust NetworkRunner
- 111-model fused simulation loop with Rayon-parallel population stepping (was 80)
- CSR-sparse projection propagation
- Scales to 100K+ neurons with near-linear speedup

### Model Zoo
- 10 pre-built network configurations: Brunel balanced, cortical column, CPG, decision-making, working memory, visual cortex V1, auditory processing, MNIST classifier, SHD speech, DVS gesture
- 3 pre-trained weight sets: MNIST (784-128-10), SHD (700-256-20), DVS gesture (256-256-11)

### conda-forge
- Recipe draft prepared for staged-recipes submission; not yet published on
  conda-forge

### Analysis Toolkit
- 126 spike train analysis functions across 23 modules (22 spike_stats + 1 explainability)
- Covers: basic stats, variability, rate estimation, distance metrics,
  correlation, spectral, temporal, stimulus, LFP coupling, surrogates,
  information theory, causality, dimensionality, decoding, network,
  point process, sorting quality, waveform, statistics, patterns, SPADE, GPFA
- Pure NumPy, zero external dependencies
- Tests: 1 776 Python total, 336 Rust total

### Neuron Model Library (122 Python / 111 Rust)
- 108 individual model files in `neurons/models/` (one file per model)
- 108 individual model files across 14 families: IF variants, Biophysical, Adaptive, Oscillatory, Bursting, Synaptic, Multi-compartment, Map-based, Stochastic, Population, Hardware, Modern/ML, Rate, Other
- Notable additions: TraubMiles, WilsonHR, Pospischil (5 cortical types), ConnorStevens, WangBuzsaki, PinskyRinzel, Destexhe, HuberBraun, GolombFS, MainenSejnowski
- Historical coverage from McCulloch-Pitts (1943) to Gated LIF (2022)
- 10 PyTorch training cells: LIF, IF, Synaptic, ALIF, RecurrentLIF, ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF

### MNIST 99.49% Accuracy
- `examples/mnist_conv_train.py` — ConvSpikingNet with learnable beta/threshold
- Architecture: Conv(1->32)->LIF->Pool->Conv(32->64)->LIF->Pool->FC->LIF->FC->LIF
- Techniques: FastSigmoid surrogate, cosine LR schedule, data augmentation, membrane readout
- Trained on RTX 6000, 30 epochs, 25 minutes
- Model checkpoint: `examples/mnist_conv_train/results/conv_spiking_net_best.pt`
- Reproducibility manifest: `benchmarks/results/mnist_conv_accuracy_reproducibility.json`

### Intel Lava/Loihi Bridge
- `integrations/lava_bridge.py` — SCtoLavaConverter, export_weights_loihi
- SCDenseProcess + PySCDenseModel for Lava CPU simulation
- Weight conversion: SC probability [0,1] -> Loihi fixed-point

### Rust Engine parity expansion (v3.8/v3.9 carry-forward)
- **Sobol bitstream** (M1): Gray-code Sobol quasi-random encoder in Rust (`sobol.rs`)
- **HomeostaticLIF**: adaptive threshold neuron with EMA spike rate tracking
- **DendriticNeuron**: XOR-nonlinearity compartmental model
- **RewardStdpSynapse**: eligibility trace + reward-modulated STDP
- **Conv2DLayer**: im2col + SC multiply-accumulate convolution
- **RecurrentLayer**: echo state network with state feedback
- **LearningLayer**: online STDP-integrated dense layer
- **FusionLayer**: weighted stochastic multiplexing across modalities
- **MemristiveLayer**: dense layer with stuck-at faults and write noise
- **SpikeRecorder**: buffered spike recording with firing rate and ISI stats
- **ConnectomeGenerator**: Watts-Strogatz and Barabási-Albert topology generators
- **FaultInjector**: bit-flip and stuck-at fault injection on packed bitstreams
- **MLIR emitter**: CIRCT hw/comb dialect IR emission (`ir/emit_mlir.rs`)
- **Static synapse**: completed with excitatory/inhibitory polarity
- **Surrogate gradient**: added Triangular and PiecewiseLinear variants
- Rust neuron models callable from Python: 111 (of 122 Python total)

### SIMD Hardening (v3.8 carry-forward)
- Fused `softmax_inplace_f64_dispatch` with SIMD max/sum/scale
- Hamming distance dispatch for all backends (AVX2, SVE, RVV)
- SVE/RVV softmax portable fallbacks
- Attention softmax refactored to use fused dispatch

### Quantum Backend Stabilisation (v3.9 carry-forward)
- IBM Heron r2 noise model: depolarizing, amplitude/phase damping, readout asymmetry
- Parameter-shift gradient rule for variational quantum circuits
- Hybrid quantum-classical VQE pipeline with scipy optimizer
- QEC noise integration with surface code threshold comparison

### Holonomic Adapter Ecosystem (v3.9 carry-forward)
- L1-L16 adapters registered in ComponentRegistry with `create_adapter()` factory
- Per-adapter benchmark suite: latency, memory, throughput (with/without JAX JIT)
- Plugin discovery via `importlib.metadata` entry points

### Type Safety Cleanup (M2)
- Removed 235 unnecessary Python type-suppression comments (260 -> 25)
- Remaining 25 are justified: CuPy type aliases, optional imports, private method access

### GPU SNN Training with Surrogate Gradients
- `sc_neurocore.training` — PyTorch-based differentiable SNN training module
- 3 surrogate gradient functions: FastSigmoid (Zenke 2018), SuperSpike (Zenke 2021), ATan (Fang 2021)
- `LIFCell`, `RecurrentLIFCell` — `nn.Module` LIF neurons with autograd through spikes
- `SpikingNet` — multi-layer feedforward SNN with spike-count and membrane readout
- `to_sc_weights()` — export trained float weights to [0,1] range for SC bitstream deployment
- 3 loss functions: spike count cross-entropy, membrane cross-entropy, spike rate MSE
- `train_epoch()` / `evaluate()` — training loops with temporal unrolling
- `examples/mnist_surrogate/train.py` — MNIST benchmark (~95% accuracy, 10 epochs)
- 31 tests covering surrogates, modules, and training loops
- Requires `pip install sc-neurocore[training]` or `sc-neurocore[research]`

## [3.10.0] - 2026-03-09

### MNIST-on-FPGA Demo
- **End-to-end pipeline**: `examples/mnist_fpga/demo.py` — train (sklearn digits),
  PCA 64→16, quantise Q8.8, stochastic computing inference, Verilog weight export
- Float 94.2%, Q8.8 94.2%, SC 94.0% (L=1024, sign-magnitude encoding)
- Resource estimate: 16→10 config = ~56K LUTs (fits Artix-7 100T)
- `hdl/sc_dense_matrix_layer.v` — per-neuron weight dense layer for classification

### Vivado Tooling
- `tools/vivado_impl.tcl` — non-project flow: synth → place → route (250 MHz default)
- `tools/vivado_report.py` — parse timing/utilization/power reports to JSON

### Tutorial
- `docs/tutorials/fpga_in_20_minutes.md` — 6-section FPGA deployment tutorial

### Paper
- JOSS paper updated to submission-ready state (`paper/paper.md`)
- 12 references with DOIs, MNIST demo results, Brian2 comparison, formal verification

### Documentation Overhaul
- README: benchmarks section (Rust SIMD, Brian2 comparison, Yosys synthesis)
- README: all 10 HDL modules listed with descriptions
- Zenodo DOI updated to 10.5281/zenodo.18906614
- CITATION.cff, .zenodo.json: DOI, version, author corrections
- CONTRIBUTING.md, VALIDATION.md, getting-started.md: test counts, Python version
- Yosys MODULES list updated (10 modules)

### Fixes
- Zenodo author list corrected (sole author: Miroslav Šotek)
- DOI badge in README points to latest Zenodo record

## [3.9.1] - 2026-03-08

### Benchmarks
- **20-variant Brunel translator suite**: comprehensive characterization of
  SC-NeuroCore against Brian2 across neuron models (LIF, Izhikevich, homeostatic),
  timing variants, synapse types (STDP, dot product, Sobol bitstream), layer
  architectures (JAX, recurrent, memristive), and acceleration backends
  (Numba JIT, PyTorch CUDA GTX 1060, vectorized NumPy)
- V18 Numba JIT: 9.5× speedup over per-neuron Python loop
- V19 PyTorch CUDA: 8.7× speedup on GTX 1060 6GB
- V14 Sobol bitstream: 1.04× Brian2 ratio (closest match)
- 19 translator unit tests (`test_brunel_translator.py`)
- Fix BENCHMARKS.md CPU: i5-11600K @ 3.9 GHz (AVX-512, DL Boost)
- Fix 3 delta-PSC wiring bugs: v_reset omission, R*I*dt dilution, Poisson-as-current
- Comprehensive BENCHMARKS.md with 13+ sections and measured numbers
- Rust Criterion: 31 benchmarks captured (AVX-512)
- Brian2 2.10.1 SNN comparison: Brunel balanced network head-to-head
- NeuroBench-aligned metrics: 4 configurations, up to 847 MOP/s
- v2 vs v3 PyO3 speedup: 7.3× on large dense forward (128→64)
- Advanced module benchmarks: quantum hybrid, GNN, S-Former, BCI, DVS, chaos RNG
- Yosys synthesis tooling (`tools/yosys_synth.py`, `tools/yosys_synth.tcl`)
- CuPy 14.0.1 installed for GPU VectorizedSCLayer

### Paper
- Updated JOSS paper with measured Criterion numbers (41.3 Gbit/s pack, 224 Mstep/s LIF)
- Replaced estimated FPGA claim with Yosys tooling reference

## [3.9.0] - 2026-03-06

### SCPN Layers
- **L8-L16 pure NumPy layers**: 9 new layer files completing the full 16-layer SCPN stack (`scpn/layers/l8_phase_field.py` through `l16_director.py`)
- **16-layer registry**: `LAYER_REGISTRY` dict, `create_full_stack()` now returns all 16 layers
- **Full integrated step**: `run_integrated_step()` chains L1→L16 with inter-layer coupling

### Quantum Error Correction
- **SurfaceCodeShield**: d=3 rotated surface code with X/Z stabilizers, syndrome measurement, lookup-table decoding — corrects arbitrary single-qubit errors
- Extensible to d=5 (encode/decode/syndrome paths support arbitrary odd distance)

### Benchmarks
- Fixed double-step bug in `benchmarks/snn_comparison.py` (neurons were advanced twice per timestep)
- Fixed Lava stub notes (requires Loihi 2 hardware)
- Fixed `benchmark_suite.py` output path → `benchmarks/results/`
- SNN comparison results recorded in `docs/benchmarks/BENCHMARKS.md`

### Formal Verification
- **LIF neuron**: `hdl/formal/sc_lif_neuron.sby` + `sc_lif_neuron_formal.v` — 5 properties (reset, spike-reset, refractory clamp, counter bound, spike reachability)
- **Bitstream synapse**: `hdl/formal/sc_bitstream_synapse.sby` + `sc_bitstream_synapse_formal.v` — 4 properties (AND correctness, zero propagation, full-high, input coverage)

### Testing
- 6 cross-layer coupling integration tests (`test_scpn_cross_layer.py`)
- 9 surface code QEC tests (`test_qec_surface.py`)
- Test count: 945 → 960+

### Documentation
- JOSS paper: updated test count (960), qualified LUT claim, added Brunel/NeuroBench/LFSR bib entries

## [3.8.2] - 2026-03-06

### Documentation & Adoption
- **BENCHMARKS.md**: Populated with 14 real benchmark entries (i5-11600K, NumPy 1.26.4), Rust engine Criterion numbers, comparison context, reproduction instructions
- **JOSS paper draft**: `paper/paper.md` + `paper.bib` (6 references) — statement of need, architecture, key features, QA
- **End-to-end notebook**: `notebooks/03_end_to_end_pipeline.ipynb` — 7-cell walkthrough (encode→synapse→neuron→VectorizedSCLayer→accuracy analysis)

### Testing
- **18 Hypothesis property-based tests**: Bitstream encoding roundtrip, LFSR determinism, neuron output constraints, layer shape invariants, RNG range/shape, recorder accumulation, encoder binary output
- **Test count**: 887 → 911 tests passing, 98.41% coverage

### Issues Closed
- #30: Property-based testing with Hypothesis
- #33: JOSS paper draft

## [3.8.1] - 2026-03-06

### Enterprise Hardening
- **11 CI workflows**: ci, v3-engine, v3-wheels, benchmark, docs, pre-commit, codeql, scorecard, stale, release, publish — all SHA-pinned, concurrency-grouped
- **Supply chain**: Every GitHub Action SHA-pinned (30+ refs), `pypa/gh-action-pypi-publish` pinned, dependabot groups GH Actions PRs
- **Security**: Bandit SAST in CI, dependabot security updates enabled, private vulnerability reporting enabled, CodeQL weekly schedule
- **Branch protection**: 6 required status checks (lint, test×2, spdx-guard, build, pre-commit)
- **Dockerfile**: Multi-stage build, Python 3.12, non-root user, OCI labels, healthcheck
- **Preflight gate**: `tools/preflight.py` (black + bandit + spdx-guard + pytest), `.githooks/pre-push` hook
- **Release pipeline**: `publish.yml` (PyPI OIDC trusted publisher, 12 platform wheels), `release.yml` attaches sdist to GitHub Releases
- **Repo hygiene**: `.dockerignore`, `.editorconfig`, `.gitattributes`, `CONTRIBUTORS.md`, `CODEOWNERS`, PR template, issue templates (YAML forms), dependabot commit-message prefixes
- **Labels**: 22 labels with colors (ci, security, breaking-change, hdl, performance, needs-review, pinned, roadmap, stale)
- **Settings**: Delete-branch-on-merge, wiki/projects disabled, OpenSSF Scorecard badge

### Lint Enforcement & Python Version
- **ruff check enforced in CI**: 258 unused/deprecated imports auto-fixed across 138 files
- **CI test matrix expanded**: Python 3.10, 3.11, 3.12 (dropped 3.9 — EOL, autoray/PennyLane incompatible)
- **`requires-python` bumped to `>=3.10`**: badge, classifiers, black/ruff target-version updated
- **bandit added to `[dev]` extras**: contributors can now `make lint` after `pip install -e ".[dev]"`
- **benchmark.yml permissions tightened**: `permissions: {}` at top, scoped per-job
- **SECURITY.md / SUPPORT.md**: GitHub Security Advisories link added
- **VALIDATION.md refreshed**: 1058 tests, 98% gate, ruff/bandit/spdx-guard/codeql/scorecard gates documented

---

## [3.8.0] - 2026-03-05

### Hardening & Documentation
- **Coverage gate raised to 98%**: De-omitted 6 modules (chaos/rng, analysis/explainability, physics/wolfram_hypergraph, robotics/swarm, learning/neuroevolution, spatial/*) plus bio/neuromodulation. 34 new tests, 1058 total, 98.10% coverage
- **NumPy 2.x audit**: Zero deprecated calls found — codebase fully compatible
- **Full API documentation**: 25 new mkdocstrings pages, all 44 subpackages wired into nav. Reorganized into Core / Compiler & Export / Domain Modules / Infrastructure sections
- **Stale issue automation**: `.github/workflows/stale.yml` — weekly sweep, 60+14 day lifecycle, exempt: pinned/security/roadmap
- **CI coverage gate sync**: `ci.yml` and `pyproject.toml` both enforce `fail_under = 98`

---

## [3.7.0] - 2026-02-11

### Adaptive Runtime Engine -- HDC/VSA, SCPN Petri Nets, Fault-Tolerant Logic
- **HDC/VSA kernel**: `BitStreamTensor` gains `xor`, `xor_inplace`, `rotate_right`, `hamming_distance`, `bundle` methods for hyper-dimensional computing on 10,000-bit vectors
- **SIMD fused XOR+popcount**: AVX-512 VPOPCNTDQ / AVX2 / portable dispatch for hamming distance hot path
- **PyBitStreamTensor**: New `#[pyclass]` exposing full HDC algebra to Python (13 methods)
- **HDCVector**: High-level Python class with operator overloading (`*`=bind, `+`=bundle, `.similarity()`, `.permute()`)
- **PetriNetEngine**: Stochastic Colored Petri Net engine wrapping two `DenseLayer` instances for Places->Transitions->Places firing
- **Fault-tolerant logic**: Boolean logic with stochastic redundancy (1024-bit) survives 40%+ bit-flip rates
- **44 new tests**: 15 Rust integration + 20 Python HDC + 9 Python Petri Net
- **2 demos**: HDC symbolic query ("Capital of France?"), safety-critical Boolean logic with error sweep
- **Comprehensive study**: `docs/research/SC_NEUROCORE_V3.7_ADAPTIVE_RUNTIME_ENGINE_STUDY.md`

---

## [3.6.0] - 2026-02-10

### Fused Dense Pipeline + Fast PRNG + Batch Forward
- **Fused encode+AND+popcount**: `forward_fused()` eliminates intermediate input bitstream materialization
- **Fast PRNG switch**: xoshiro256++ for dense fast-path input encoding and numpy batch encoding
- **Batched dense API**: `DenseLayer.forward_batch_numpy()` processes N samples in one FFI call
- **New diagnostics**: criterion benches for fused dense, encode+popcount, batch dense, and PRNG throughput
- **Version/test/docs update**: bumped to 3.6.0 with the fused dense pipeline test suite and migration notes

---

## [3.5.0] - 2026-02-10

### SIMD Pipeline Acceleration
- **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated dense inner loop with AVX2 fallback
- **SIMD Bernoulli encode**: AVX-512BW/AVX2 threshold compare path for packed Bernoulli generation
- **Flat weight storage**: Contiguous `[neuron][input][word]` packed layout for cache-friendly access
- **Zero-allocation LIF batch**: Pre-allocated numpy outputs for batch LIF APIs
- **Criterion benchmarks**: Added fused-and-popcount and SIMD Bernoulli diagnostics

---

## [3.4.0] - 2026-02-10

### SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps

---

## [3.3.0] - 2026-02-10

### Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark

---

## [3.2.0] - 2026-02-10

### Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison and dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output plus parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: Added local `.gitignore` for generated artifacts

---

## [3.1.0] - 2026-02-10

### Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates `Vec<u8>` intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs and skips encoding
- **batch_encode_numpy**: Returns a 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Added automated wheel upload on `v3.*` tags via Trusted Publisher workflow
- **Updated Benchmarks**: Added dense `fast` and `prepacked` benchmark variants

---

## [3.0.0] - 2026-02-10

### Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine

### Release Candidate (3.0.0-rc.1)
- **IR Python Bridge**: Full PyO3 bindings for ScGraphBuilder, ScGraph, verify, print, parse, emit_sv
- **Co-sim Activation**: Verilator compilation + simulation when available; graceful skip preserved
- **Wheel CI**: Cross-platform wheel builds (Linux/macOS/Windows x Python 3.9-3.12)
- **Benchmark Report**: Formal v2-vs-v3 performance comparison with Blueprint section 8 targets
- **IR Demo**: Real end-to-end Python->IR->verification->SystemVerilog demo

### HDL Compilation Pipeline (3.0.0-beta.1)
- **SC IR**: Rust-native intermediate representation with 11 op types
- **SV Emitter**: Compile IR graphs to synthesizable SystemVerilog
- **Co-sim**: Verilator-based verification against Rust golden model
- **CI**: Expanded test coverage to include all differentiation, acceleration, integration, and HDL Python tests

### Integration & Hardening
- SSGF-compatible Kuramoto solver (`step_ssgf`, `run_ssgf`)
- Property-based testing with proptest (12 property tests)
- Multi-head attention (`forward_multihead`)
- SC-mode GNN (`forward_sc`)
- End-to-end training demo
- Comprehensive rustdoc

### Differentiation & Acceleration
- Surrogate gradient LIF (FastSigmoid, SuperSpike, ArcTan)
- DifferentiableDenseLayer for backpropagation
- Stochastic attention (rate + SC mode)
- Graph neural network layer
- Kuramoto oscillator solver
- Criterion benchmarks + v2/v3 comparison

### Foundation
- Rust engine with PyO3 bindings
- Bit-exact LFSR, LIF neuron, dense layer
- SIMD dispatch (AVX-512, AVX2, NEON, portable)
- Python bridge with v2-compatible API
- Equivalence test suite

---

## [2.2.0] - 2026-02-09

### Added
- **Module Discoverability**: Populated 36 stub `__init__.py` files with proper
  `__all__` exports and lazy imports. Every package now supports
  `from sc_neurocore.X import Y` without touching internals.
- **MkDocs API Documentation**: Added `mkdocs.yml` with mkdocstrings plugin,
  `docs/index.md`, `docs/getting-started.md`, `docs/architecture.md`, and 17
  API reference stubs in `docs/api/`.
- **Examples Directory**: 6 runnable example scripts demonstrating bitstream
  encoding, neuron layers, vectorized inference, SCPN stack, HDL generation,
  and ensemble consensus (`examples/01`–`06`).
- **Module Docstrings**: Added module-level docstrings to `pipeline/ingestion.py`,
  `pipeline/training.py`, `utils/model_bridge.py`, `ensembles/orchestrator.py`.

### Changed
- **Print → Logging**: Converted 60+ `print()` calls across 25 source modules
  to structured `logging` with `getLogger(__name__)` and `%`-style formatting.
  Dashboard and drivers intentionally excluded (stdout by design).
- **CI Coverage Threshold**: Raised `--cov-fail-under` from 50 to 97 in
  `.github/workflows/ci.yml` to match actual coverage.
- Version bump: 2.1.0 → 2.2.0.

### Fixed
- **Unused Imports**: Removed dead imports from 7 files (`bio/uploading.py`,
  `core/replication.py`, `core/immortality.py`, `export/onnx_exporter.py`,
  `dashboard/text_dashboard.py`, `hdl_gen/verilog_generator.py`, `viz/web_viz.py`).
- **Input Validation**: `VectorizedSCLayer.forward()` now raises `ValueError`
  on wrong-shape input instead of silently producing garbage.
- **File I/O Error Handling**: `onnx_exporter.py`, `immortality.py`,
  `verilog_generator.py`, and `replication.py` now catch `OSError` on
  file operations and log meaningful messages.

### Security
- **Pickle Allowlist**: Replaced wildcard `'numpy.core.numeric': {'*'}` with
  explicit `{'_frombuffer', 'scalar'}` in `core/immortality.py`.
- **Path Traversal Prevention**: `core/replication.py` now validates that the
  destination directory is within or below the working directory via
  `os.path.realpath()` + `os.path.relpath()`.

---

## [2.1.0] - 2026-02-08

### Fixed (Critical)
- **HDL Bitstream Encoder Seed Decorrelation**: All parallel encoders shared
  hardcoded seed `0xACE1`, producing correlated bitstreams and breaking SC
  multiplication (`P(x AND x) = P(x)` instead of `P(x)*P(w)`). Added per-instance
  `SEED_INIT` parameter with prime-stride offsets (input: `0xACE1 + i*7`,
  weight: `0xBEEF + i*13`).
- **HDL Missing Port Connections**: `noise_in` and `v_out` were floating on
  LIF neuron instances in `sc_dense_layer_core.v`. Connected via wire buses.
- **HDL Duplicate Port**: Removed duplicate `.stream_len` in `sc_neurocore_top.v`.
- **Fixed-Point Overflow**: `FixedPointLIFNeuron` now applies `_mask()` for
  proper two's complement overflow wrapping on membrane potential.

### Added
- **GPU Acceleration Backend** (`accel/gpu_backend.py`):
    - CuPy/NumPy dual-path with automatic GPU detection and CPU fallback.
    - `gpu_pack_bitstream()`, `gpu_vec_and()`, `gpu_popcount()`, `gpu_vec_mac()`.
    - `VectorizedSCLayer` auto-selects GPU when CuPy is available.
- **Performance Benchmark Suite** (`scripts/benchmark_suite.py`):
    - 14 benchmarks across 5 categories (scalar, packed ops, dense layer,
      full pipeline, GPU).
    - `--full` mode (10x iterations), `--markdown` output to `BENCHMARKS.md`.
- **CI/CD Pipeline** (`.github/workflows/sc-neurocore-ci.yml`):
    - Lint (black + mypy), Test (Python 3.9/3.11/3.12 matrix, coverage >= 60%),
      Build (wheel + install verification).
- **Co-Simulation Harness**:
    - `hdl/tb_sc_lif_neuron.v`: Verilog testbench reading stimuli.txt, writing
      results_verilog.txt for bit-exact comparison.
    - `scripts/cosim_gen_and_check.py`: CLI driver with `--generate` and `--check`.
- **Bit-True Python Models**:
    - `FixedPointLFSR`: 16-bit maximal-length LFSR (period 65535).
    - `FixedPointBitstreamEncoder`: LFSR + unsigned comparator.
    - `_mask()`: Two's complement sign-extension with overflow wrap.
- **Public API Surface**: Root `__init__.py` exports 28 symbols across 7
  subpackages. All subpackage `__init__.py` files populated.
- **Tiered Module System**: 43 subpackages categorised as `core` (7),
  `research` (24+), or `contrib` (5). Install extras: `[gpu]`, `[research]`,
  `[contrib]`.
- **Behavioural Equivalence Tests**: 29 tests covering LFSR, encoder, LIF
  neuron, full pipeline, and bit-width masking.
- **GPU Backend Tests**: 17 tests covering all GPU primitives and
  VectorizedSCLayer integration.

### Changed
- Version bump: 2.0.0 -> 2.1.0.
- `pyproject.toml`: Added tool configs (pytest, black, mypy), tiered extras.
- `VectorizedSCLayer`: Refactored to use GPU backend with CPU fallback.

---

## [2.0.0] - 2026-01-12

### Added
- **Sapience & Sentience (v2.2.0)**:
    - `MetaCognitionLoop`: Computational self-awareness and self-modeling.
    - `NeuromodulatorSystem`: Dopamine/Serotonin emotional state modulation.
    - `NeuroArtGenerator`: Generative AI for internal state expression.
    - `AsimovGovernor`: Ethical constraint system (Three Laws).
    - `MindDescriptionLanguage (MDL)`: Substrate-independent soul serialization.
    - `DigitalSoul`: Persistence and reincarnation protocols.
    - `VonNeumannProbe`: Code-level self-replication.
- **Galactic Scale (v2.1.0)**:
    - `InterstellarDTN`: Long-range delay-tolerant networking.
    - `DysonPowerGrid`: Stellar-scale energy management.
    - `KardashevEstimator`: Civilization Type metrics.
    - `DarkForestAgent`: Game-theoretic survival logic.
    - `MPIDriver`: Distributed cluster-scale simulation.
    - `SNNGeneticEvolver`: Automated architecture optimization.
- **Transcendent & Omega (v2.0.5)**:
    - `HeatDeathLayer`: Entropy-survival computing.
    - `PlanckGrid`: Spacetime lattice theoretical limits.
    - `HolographicBoundary`: 3D-to-2D info mapping (AdS/CFT).
    - `EverettTreeLayer`: Many-Worlds branching solver.
    - `WolframHypergraph`: Graph-rewrite universe evolution.
    - `CategoryTheoryBridge`: Unified mathematical functors.
    - `FormalVerifier`: SMT-based safety proofs.
- **Exotic & Frontiers (v2.0.0)**:
    - `VectorizedSCLayer`: 64-bit packed JIT-accelerated core.
    - `QuantumStochasticLayer`: VQC qubit rotation bridge.
    - `StochasticTransformerBlock`: Spike-driven attention.
    - `MemristiveDenseLayer`: Hardware-aware analog simulation.
    - `StochasticCPG`: Robotic locomotion oscillators.
    - `MyceliumLayer`: Fungal network dynamics.
    - `BCIDecoder`: Neural signal (EEG) interface.
    - `DVSInputLayer`: Event Camera (AER) processing.
    - `EnergyProfiler`: 45nm Energy/CO2 estimation.
    - `WatermarkInjector`: IP protection security backdoors.

### Optimized
- `BitstreamAverager`: 6x speedup using running sum algorithm.
- `BitstreamEncoder`: Added Sobol Sequence (LDS) mode for faster convergence.

### Fixed
- Fixed f-string syntax in Verilog generator.
- Fixed dimension mismatch in Attention mechanism.
- Addressed Windows encoding issues in documentation generation.

## [1.0.0] - 2025-12-03
- Initial Release: Stochastic Neurons, Synapses, and Basic Bitstream Utilities.

- Hardened FitzHugh-Rinzel Python, Rust engine, Rust safety, Go, and Julia paths with finite-parameter validation plus candidate-first RK4 commits that preserve state on invalid currents, corrupted runtime contracts, and overflow candidates.
- Hardened McKean Rust engine, Rust safety, Go, and Julia paths with candidate-first simultaneous-Euler commits and no-spike state preservation for invalid currents, corrupted runtime contracts, and overflow candidates; later promoted the maintained Python, Rust engine, Rust safety, Go, and Julia chain to candidate-first RK4 integration.
- Hardened Morris-Lecar Rust engine finite-state commits and extended Go/Rust safety coverage for invalid-current and potassium-rate overflow rejection without changing the documented conductance equations.
- Hardened Terman-Wang Rust engine finite-state commits, Julia timestep semantics, and Go/Rust safety state-preservation tests for invalid drive and cubic-overflow candidates; later promoted the maintained Terman-Wang chain to candidate-first RK4 integration.
- Hardened Quadratic IF Rust engine finite-update commits, Julia timestep semantics, and Go service tests for invalid current and non-finite Euler increments.
