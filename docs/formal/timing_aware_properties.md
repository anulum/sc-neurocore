# Timing-Aware Formal Properties

SC-NeuroCore now carries a timing-aware formal framework for FPGA control loops. The framework targets bounded latency, hard deadlines, and bounded liveness for generated or hand-authored HDL surfaces where a missed response is a control-system fault, not a cosmetic regression.

## Production contract

The framework has four surfaces:

1. `SystemVerilog`: reusable monitors in `hdl/formal/timing/timing_wrapper_lib.sv` and assertion macros in `hdl/formal/timing/timing_assertions.svh`.
2. `SymbiYosys/cvc5`: executable proof examples, currently anchored by the dense-layer latency contract in `hdl/formal/timing/example_dense_layer_core_latency.sby`.
3. `Python`: `TimingProperty` and `TimingProofOrchestrator` provide deterministic proof orchestration and fail-closed dependency reporting.
4. `nuXmv` and `Kind2`: emitters generate model-checker forms for the same timing property specification. Local execution is recorded as unavailable when those external binaries are absent.

A timing property is valid only when it states a real event-response contract. The dense-layer example proves these contracts:

1. `start_pulse` must lead to `run_done` within the bounded dense-layer stream latency.
2. `start_pulse` must lead to `step_valid` inside the first-step liveness window.
3. `run_done` must not remain concurrent with `running`.

## Monitor semantics

`sc_latency_monitor` starts one obligation when `start_event` rises while no obligation is active. It clears the obligation on `end_event`. If the obligation remains active after the configured cycle bound, the sticky `violation` flag is asserted.

`sc_deadline_monitor` and `sc_bounded_liveness_monitor` are named wrappers over the same obligation semantics. This keeps deadline and liveness contracts auditable while avoiding divergent RTL behaviour between property classes.

## Consumable surface for external integration

A downstream repository (for example SCPN-MIF-CORE) binds these properties against its own RTL in the open-source Yosys/SymbiYosys flow, which has no concurrent SVA. Every property here is therefore a counter/shift-register monitor plus a **procedural immediate** `assert (!violation)` inside `always @(posedge CLK)` — read with `read_verilog -formal -sv` alongside the consumer's RTL. The framework owns the reusable property; the consumer owns the instantiation against its own signals.

### Bounded latency

`` `SC_ASSERT_LATENCY_LE(NAME, CLK, RST_N, START_EVENT, END_EVENT, BOUND_CYCLES) `` expresses "END_EVENT within BOUND_CYCLES of START_EVENT", e.g. an actuator asserted within N cycles of a sensor event. `SC_ASSERT_DEADLINE_LE` and `SC_ASSERT_BOUNDED_LIVENESS` share the shape. Cycle-accurate only — nanosecond timing is a post-route/silicon fact, not a framework claim.

### Two-flop CDC synchroniser

`` `SC_ASSERT_CDC_TWO_FLOP(NAME, DST_CLK, RST_N, ASYNC_IN, META_Q, SYNC_OUT, SYNC_DEPTH_N) `` binds `sc_cdc_two_flop_monitor` over the destination-domain synchroniser flops the consumer owns (`META_Q` = first flop, `SYNC_OUT` = last flop). The monitor builds a `SYNC_DEPTH_N`-deep reference delay of `ASYNC_IN` and, once warmed up, asserts:

- `SYNC_OUT` is `ASYNC_IN` delayed by exactly `SYNC_DEPTH_N` flops — this pins the synchroniser depth (a one-flop chain fails) and the absence of combinational contamination on the data path;
- `SYNC_OUT` is a pure registered copy of `META_Q` — no combinational/glitch path escapes past the last flop, so the output is stable for the whole destination cycle.

A `cover` on the synchronised-output transition keeps the proof non-vacuous. This is RTL-level structural verification; physical metastability resolution time stays a silicon/STA fact, so the monitor never asserts that the first flop is metastability-free. `example_cdc_two_flop_synchroniser.{sv,sby}` is the worked proof; the test suite also proves a one-flop synchroniser is rejected, so the property has teeth. Multi-bit non-gray buses need a handshake/FIFO template rather than this single-bit/gray-coded one.

### SymbiYosys task convention

Tasks register in a consumer's `formal_manifest.py` drift gate with the triple `(mode, engine, depth)` used by `example_*.sby`:

```
[options]
mode bmc            # or `prove` for k-induction; the monitors reset cleanly and are induction-friendly
depth <BOUND + margin>   # CDC: SYNC_DEPTH + 4; latency: BOUND + 2

[engines]
smtbmc cvc5         # boolector is interchangeable

[script]
read_verilog -formal -sv -I. timing_wrapper_lib.sv <wrapper>.sv <rtl...>.sv
prep -top <formal_wrapper>
```

### Ownership

For a CDC crossing into a consumer's fabric, the consumer owns the synchroniser RTL instantiation (e.g. `mif_aer_cdc_synchroniser.sv`) and sc-neurocore owns the property template plus the upstream stream (the AER router/encoder up to the crossing boundary).

## Dependency policy

`SymbiYosys` and `cvc5` are executable proof dependencies for the dense-layer example. `nuXmv` and `Kind 2` are optional external model-checker runtimes in this repository state: the emitters are tested and benchmarked, while runtime execution is reported as unavailable unless the binaries are installed on the host.

The orchestrator never treats a missing executable as a proof pass. It returns `exit_code=127`, records the missing dependency names, and marks the proof as failed.

## Benchmarking and core isolation

Timing-framework benchmark evidence belongs in `benchmarks/results/local_python_2026-06-04_timing_formal_framework.json`. The benchmark records:

1. SystemVerilog proof runtime through SymbiYosys/cvc5.
2. Python property construction and proof orchestration.
3. nuXmv model-emitter throughput.
4. Kind 2 Lustre-emitter throughput.
5. Runtime CPU affinity, cgroup cpuset, load average, and governor sample.

Because the workstation can be under load, timing benchmark comparisons are only valid when run under a runtime cpuset shield. The required local lane pins the process to CPUs `10-11` and keeps the host system slice off those CPUs during the run.

## Replication

Run the proof directly:

```bash
sby -f hdl/formal/timing/example_dense_layer_core_latency.sby
```

Run the benchmark with isolated benchmark cores:

```bash
sudo systemctl set-property --runtime system.slice AllowedCPUs=0-9
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-9
sudo systemctl set-property --runtime init.scope AllowedCPUs=0-9
sudo systemd-run --wait --collect --pipe --slice=benchmark.slice \
  -p AllowedCPUs=10-11 -p CPUAffinity='10 11' \
  --uid=anulum --gid=anulum --working-directory="$PWD" \
  --setenv=PYTHONPATH="src:hdl/formal" \
  /usr/bin/taskset -c 10-11 \
  .venv/bin/python benchmarks/bench_timing_formal_framework.py \
  --json benchmarks/results/local_python_2026-06-04_timing_formal_framework.json
sudo systemctl set-property --runtime system.slice AllowedCPUs=0-31
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-31
sudo systemctl set-property --runtime init.scope AllowedCPUs=0-31
```

If a different workstation has a different CPU topology, replace `10-11` with dedicated benchmark cores and record the resulting `host_context.cgroup_effective_cpuset` in the benchmark JSON.
