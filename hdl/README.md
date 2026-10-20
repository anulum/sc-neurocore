# SC-NeuroCore HDL Modules

Synthesisable Verilog for the SC-NeuroCore neuromorphic datapath.
All modules target Yosys 0.63 + sv2v 0.0.13.

## RTL Modules

| Module | Description | Testbench | Formal Proof |
|--------|-------------|:---------:|:------------:|
| `sc_lif_neuron.v` | Fixed-point LIF neuron (16-bit state) | tb | sby |
| `sc_bitstream_encoder.v` | LFSR-based stochastic bitstream encoder | -- | sby |
| `sc_bitstream_synapse.v` | Bitwise AND synapse with popcount | -- | sby |
| `sc_dotproduct_to_current.v` | Popcount accumulator to synaptic current | tb | sby |
| `sc_firing_rate_bank.v` | Multi-neuron firing-rate counter bank | tb | sby |
| `sc_dense_layer_core.v` | Single dense-layer compute core | tb | sby |
| `sc_axil_cfg.v` | AXI-Lite configuration register file | tb | sby |
| `sc_dense_matrix_layer.v` | Matrix-level dense layer (multi-core) | tb | -- |
| `sc_dense_layer_top.v` | Top-level dense layer wrapper | -- | -- |
| `sc_neurocore_top.v` | Full chip top-level integration | tb | -- |

**7/10** modules have SymbiYosys formal proofs (`hdl/formal/*.sby`).
**7/10** modules have simulation testbenches (`hdl/tb_*.v`).

Modules without formal proofs (`sc_dense_matrix_layer`, `sc_dense_layer_top`,
`sc_neurocore_top`) are integration wrappers verified by simulation only.

`sc_dotproduct_to_current` requires a positive `N_INPUTS` and maps each
population count to `y_min + trunc((y_max-y_min)*count/N_INPUTS)`. Division
truncates towards zero. Both endpoint orders and the complete signed
`DATA_WIDTH` range are supported; the binary point is preserved. The RTL
regression compares repository and packaged RTL with exact integer results for widths 1–64 and fan-ins 1–33,
including extreme endpoints and every reachable count. The formal harness
checks the full signed 16-bit domain for three inputs without an endpoint-order
assumption. Run `pytest tests/test_dotproduct_current_rtl.py -q` with Icarus
Verilog installed, or use `tb_sc_dotproduct_to_current.v` for a compact bench.

`sc_dense_matrix_layer` shares input bitstreams and assigns a distinct weight
row to each neuron. `PROBABILITY_WIDTH` controls packed encoder thresholds;
`DATA_WIDTH` and `FRACTION` control signed current and LIF words. The default
probability width equals the state width for existing direct RTL consumers.
IR emission selects 16-bit thresholds after converting the declared probability
format and supplies both encoder seed bases. See [compute graph hardware words](../docs/api/sc-ir.md#dense-hardware-words).
The native `test_ir_dense_hardware` regression compiles repository and packaged
copies with Icarus and checks currents, encoder bits, membrane state and spikes
cycle by cycle; `tests/test_ir_dense_rtl.py` repeats the installed Python-to-RTL
boundary against the bit-true Python neurons. These are simulations, not board
measurements or routed timing results.
