# Wang-Buzsaki source-fidelity boundary

`WangBuzsakiNeuron` implements the three-state fast-spiking interneuron of
Wang and Buzsaki (1996), DOI
`10.1523/JNEUROSCI.16-20-06402.1996`. Sodium activation is instantaneous;
the `h` and `n` gates and membrane voltage advance in sequential
Gauss-Seidel order. One public step is 0.5 ms and contains 50 Euler substeps
at `dt = 0.01 ms`. A spike is a rising macro-boundary crossing of -20 mV;
the source model does not reset the voltage.

The committed receipt replays 512 macro steps under the repeating current
sequence `[0, 2, 5, 10, 10, 5, 2, 0]`. It records 45 events, final state
`[-47.41033815690242, 0.45409693747433205, 0.19931880843583621]`, and
SHA-256 `4a53c138c52518b13f6277dd1176190e952abb5094a14abaa0b12c9cc154013d`
over little-endian `(v, h, n, event)` rows.

The Python, production Rust, Go, Julia, and Mojo kernels execute the same
source recurrence. Their 20,000-step measured packet requires exact event
count agreement and a final-state maximum absolute error no greater than
`1e-8`; timings are loaded-host regression data, not production-speed claims.

The committed `sc_wang_buzsaki` hardware is the current equation compiler's
Q16.16 lowering. Co-simulation covers 20 source macro steps at `I = 10`; the
fixed-point event count may differ from binary64 by at most one event. Yosys
synthesis and a depth-4 public-port reset-safety BMC are separate gates. No
claim is made for universal real-number equivalence, timing closure, PPA,
device execution, network-level ING/PING reproduction, or silicon.
