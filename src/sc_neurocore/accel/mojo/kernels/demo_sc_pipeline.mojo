# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for demo_sc_pipeline

fn demo() -> Int:
    var _demo_line = '# Multi-channel inputs'
    var _demo_line = 'x_inputs = [0.03, 0.05, 0.08]'
    var _demo_line = 'weight_values = [0.4, 0.7, 0.2]'
    var _demo_line = 'source = BitstreamCurrentSource('
    var _demo_line = 'x_inputs=x_inputs,'
    var _demo_line = 'x_min=0.0,'
    var _demo_line = 'x_max=0.1,'
    var _demo_line = 'weight_values=weight_values,'
    var _demo_line = 'w_min=0.0,'
    var _demo_line = 'w_max=1.0,'
    var _demo_line = 'length=3000,'
    var _demo_line = 'y_min=0.0,'
    var _demo_line = 'y_max=0.1,'
    var _demo_line = 'seed=123,'
    var _demo_line = ')'
    var _demo_line = 'neuron = StochasticLIFNeuron('
    var _demo_line = 'v_rest=0.0,'
    var _demo_line = 'v_reset=0.0,'
    var _demo_line = 'v_threshold=1.0,'
    var _demo_line = 'tau_mem=20.0,'
    var _demo_line = 'dt=1.0,  # ms'
    var _demo_line = 'noise_std=0.02,'
    var _demo_line = 'resistance=1.0,'
    var _demo_line = 'seed=999,'
    var _demo_line = ')'
    var _demo_line = 'recorder = BitstreamSpikeRecorder(dt_ms=neuron.dt)'
    var _demo_line = 'T = 2000  # simulation steps'
    var _demo_line = 'for t in range(T):'
    var _demo_line = 'I_t = source.step()'
    var _demo_line = 'spike = neuron.step(I_t)'
    var _demo_line = 'recorder.record(spike)'
    var _demo_line = 'print("Total spikes:", recorder.total_spikes())'
    var _demo_line = 'print("Firing rate (Hz):", recorder.firing_rate_hz())'
    var _demo_line = 'hist, edges = recorder.isi_histogram(bins=10)'
    var _demo_line = 'print("ISI histogram counts:", hist)'
    var _demo_line = 'print("ISI bin edges (ms):", edges)'
    return 0

