# Requirements Traceability Matrix

Maps SC-NeuroCore requirements to tests and formal verification properties.

## Notation

- **REQ**: Requirement ID
- **TEST**: Test file::function or formal property
- **PROOF**: SymbiYosys property (bounded model check)

## Functional Requirements

| REQ | Description | Test(s) | Formal |
|-----|-------------|---------|--------|
| REQ-LIF-01 | LIF spike when v ≥ threshold | `test_neuron.py::test_lif_spikes_above_threshold` | `syb_lif_spike_iff_above.sby` |
| REQ-LIF-02 | LIF refractory suppresses spikes for N steps | `test_neuron.py::test_lif_refractory_gap` | `syb_lif_refractory_nospike.sby` |
| REQ-LIF-03 | LIF reset sets v = v_reset | `test_neuron.py::test_lif_reset_voltage` | `syb_lif_reset_value.sby` |
| REQ-ENC-01 | LFSR period = 2^16 - 1 | `test_encoder.py::test_lfsr_period_65535` | `syb_lfsr_maximal.sby` |
| REQ-ENC-02 | Bitstream probability within ±2% of target | `test_encoder.py::test_encoding_accuracy` | — |
| REQ-SYN-01 | STDP weight bounded [w_min, w_max] | `test_synapse.py::test_stdp_bounds` | `syb_stdp_bounds.sby` |
| REQ-SYN-02 | Pre→post potentiates, post→pre depresses | `test_synapse.py::test_stdp_direction` | — |
| REQ-DEN-01 | Dense layer output ∈ [0, 1] | `test_layers.py::test_dense_output_range` | — |
| REQ-DEN-02 | Sparse layer matches dense within 5% | `test_layers.py::test_sparse_dense_parity` | — |
| REQ-IR-01 | IR round-trips through print/parse | `test_ir.py::test_round_trip` | — |
| REQ-IR-02 | SV emit produces valid Verilog (Verilator lint) | `test_ir.py::test_sv_lint` | — |
| REQ-IR-03 | Verifier rejects disconnected graphs | `test_ir.py::test_verify_rejects_disconnected` | — |
| REQ-BIT-01 | Rust and Python popcount agree | `test_engine_parity.py::test_popcount_parity` | — |
| REQ-BIT-02 | SIMD popcount matches portable | `test_simd.py::test_avx2_matches_portable` | — |
| REQ-TRN-01 | Surrogate grad reduces loss over 5 epochs | `test_training.py::test_loss_decreases` | — |
| REQ-TRN-02 | Exported weights ∈ [0, 1] | `test_training.py::test_export_range` | — |
| REQ-HW-01 | Co-sim: Python == Verilog spike trains | `test_cosim.py::test_bitexact_10k_steps` | — |

## Neuron Model Requirements

| REQ | Description | Test(s) |
|-----|-------------|---------|
| REQ-NM-01 | All 111 Rust neuron models have step() + reset() | `cargo test --lib neurons` (373 total Rust tests) |
| REQ-NM-02 | HH fires with suprathreshold input | `neurons::biophysical::hodgkin_huxley::tests::hh_fires` |
| REQ-NM-03 | AdEx adaptation reduces rate | `neuron::tests::adex_adaptation_reduces_rate` |
| REQ-NM-04 | Loihi CUBA integer arithmetic only | `neurons::hardware::tests::loihi_cuba_fires` |
| REQ-NM-05 | Poisson firing rate matches λ within 10% | `neurons::special::tests::poisson_fires` |

## Coverage Summary

| Category | Requirements | Tests | Formal Properties |
|----------|-------------|-------|-------------------|
| LIF neuron | 3 | 6 | 3 |
| Encoder | 2 | 4 | 1 |
| Synapse | 2 | 4 | 1 |
| Dense layer | 2 | 5 | 0 |
| IR compiler | 3 | 8 | 0 |
| Bitstream | 2 | 4 | 0 |
| Training | 2 | 3 | 0 |
| Co-simulation | 1 | 2 | 0 |
| Neuron models | 5 | 206 | 0 |
| **Total** | **22** | **242** | **5** |
