# LapicqueNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`

**Primary source:** Lapicque (1907); English translation DOI
[`10.1007/s00422-007-0189-6`](https://doi.org/10.1007/s00422-007-0189-6)

**Interpretive companion:** Brunel and van Rossum (2007), DOI
[`10.1007/s00422-007-0190-0`](https://doi.org/10.1007/s00422-007-0190-0)

## Identity boundary

Lapicque's paper treats nerve excitation as the first attainment of a
polarization threshold in a leaky-capacitor circuit. It does not define an
automatic post-event reset or a repetitive spike generator. SC-NeuroCore
therefore exposes two deliberately separate profiles:

```python
from sc_neurocore.neurons.models import LapicqueNeuron, SCLapicqueLIFNeuron

source = LapicqueNeuron.lapicque_1907()  # counted source identity
compat = SCLapicqueLIFNeuron()           # count-neutral SC hard-reset LIF
legacy = LapicqueNeuron()                # preserved alias of the SC profile
```

The zero-argument legacy constructor remains compatible with existing network,
training, and user code. In the compiled `NetworkRunner`, the exact canonical
name `LapicqueNeuron` selects the source profile; `Lapicque`,
`SCLapicqueLIF`, and `SCLapicqueLIFNeuron` select the retained SC profile.
Python `Population("LapicqueNeuron", ...)` follows the same canonical source
route. Existing calls that pass SC-only parameters such as `tau`, `resistance`,
`v_rest`, or `v_reset` remain on the compatibility profile.

## Lapicque 1907 source profile

With source voltage $V$, series resistance $R$, polarization resistance
$\rho$, capacitance $K$, and polarization $v$, the maintained source equation is

$$K\frac{dv}{dt}=\frac{V-v}{R}-\frac{v}{\rho}.$$

For a constant pulse over one timestep,

$$
v_{n+1}=v_\infty+(v_n-v_\infty)e^{-\Delta t/\beta},\qquad
v_\infty=\frac{V\rho}{R+\rho},\qquad
\beta=\frac{KR\rho}{R+\rho}.
$$

The first candidate with $v_{n+1}\geq v_\mathrm{threshold}$ emits one event and
latches `excited=True`. Polarization continues evolving; it is not reset.
Calling `reset()` explicitly re-arms a new experiment.

Lapicque's strength-duration relation follows directly:

$$
V(t)=\frac{\alpha}{1-e^{-t/\beta}},\qquad
\alpha=v_\mathrm{threshold}\frac{R+\rho}{\rho}.
$$

The maintained defaults $K=1.1$, $R=10$, $\rho=1$, $\Delta t=0.01$ ms, and
$v_\mathrm{threshold}=1$ give $\beta=1$ ms and $\alpha=11$. They are a
normalized reproducibility point, not claimed experimental constants from the
paper. Input to `step()` is source voltage for this profile.

## Preserved SC profile

`SCLapicqueLIFNeuron` retains the historical exact-flow hard-reset recurrence

$$
\tau\frac{dv}{dt}=-(v-v_\mathrm{rest})+RI,
$$

with constant-current exact flow and reset $v\to v_\mathrm{reset}$ at the
threshold. This profile supports repetitive events, but that reset convention
is not attributed to the complete 1907 experiment. See
[SC exact-flow hard-reset LIF](sc_lapicque_lif.md).

## Execution contract

`simulate_complete(n_steps, drive, backend=...)` returns aligned post-step
`float64` polarization/voltage and `uint8` event arrays. Python, Rust/PyO3,
Julia, Go, and Mojo accept the complete profile and parameter packet. Every
batch validates fully before caller-visible state commits; Go and Mojo also
validate before writing either C-ABI output buffer.

The independent source receipt uses $V=22$ for 2,000 steps. It records one
event at zero-based index 69, preserves the complete polarization and event
digests, and separately re-derives five strength-duration points. The
100,000-step controlled benchmark reports the same complete event vector in all
five runtimes; maximum measured state difference is `1.222e-15`.

## Hardware boundary

The source-specialized `sc_lapicque_1907` core implements the normalized exact
flow in Q32.32. Co-simulation preserves the complete event vector at source
voltages 5.5, 11, 12, and 22; the two suprathreshold cases emit at indices 248
and 69, and maximum state error stays below `7e-8`. Yosys coarse synthesis
reports 11,511 cells. A depth-20 SymbiYosys/Z3 job proves reset hygiene,
permanent excitation latch, and absence of repeated events after latching.

The old `sc_lapicque` Q16.16 path remains the separate SC compatibility core.
Timing, PPA, target-device, board, physical-silicon, and universal fixed-point
equivalence evidence remain open; the source profile therefore stays at the
honest H2 boundary.

## Evidence

| Surface | Durable evidence |
| --- | --- |
| Primary-source identity and schemas | `tests/test_model_lapicque_source_contract.py` |
| Independent oracle and receipt | `tests/test_reference_lapicque_source_receipt.py`; `src/sc_neurocore/neurons/reference_receipts/lapicque_1907.json` |
| Five-runtime complete parity | `tests/test_lapicque_backend_parity.py`; `tests/test_lapicque_engine_binding.py` |
| C-ABI failure atomicity | `tests/test_lapicque_backend_c_abi.py` |
| Source and SC co-simulation | `tests/test_cosim_lapicque.py` |
| Synthesis and formal | `hdl/reports/yosys_lapicque_1907_q3232_2026-08-30.json`; `hdl/formal/catalogue/sc_lapicque_1907.sby` |
| Controlled measurement | `benchmarks/results/bench_lapicque.json`; `tests/test_bench_lapicque.py` |
