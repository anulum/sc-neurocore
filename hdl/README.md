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
