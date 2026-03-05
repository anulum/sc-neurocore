# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from ..sources.bitstream_current_source import BitstreamCurrentSource
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder


def demo():  # type: ignore
    # Multi-channel inputs
    x_inputs = [0.03, 0.05, 0.08]
    weight_values = [0.4, 0.7, 0.2]

    source = BitstreamCurrentSource(
        x_inputs=x_inputs,
        x_min=0.0,
        x_max=0.1,
        weight_values=weight_values,
        w_min=0.0,
        w_max=1.0,
        length=3000,
        y_min=0.0,
        y_max=0.1,
        seed=123,
    )

    neuron = StochasticLIFNeuron(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,  # ms
        noise_std=0.02,
        resistance=1.0,
        seed=999,
    )

    recorder = BitstreamSpikeRecorder(dt_ms=neuron.dt)

    T = 2000  # simulation steps
    for t in range(T):
        I_t = source.step()
        spike = neuron.step(I_t)
        recorder.record(spike)

    print("Total spikes:", recorder.total_spikes())
    print("Firing rate (Hz):", recorder.firing_rate_hz())
    hist, edges = recorder.isi_histogram(bins=10)
    print("ISI histogram counts:", hist)
    print("ISI bin edges (ms):", edges)


if __name__ == "__main__":
    demo()  # type: ignore
