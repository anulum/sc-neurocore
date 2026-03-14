# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer


def demo() -> None:
    # Three input channels with different scalar values
    x_inputs = [0.02, 0.05, 0.08]
    weight_values = [0.3, 0.6, 0.9]

    layer = SCDenseLayer(
        n_neurons=5,
        x_inputs=x_inputs,
        weight_values=weight_values,
        x_min=0.0,
        x_max=0.1,
        w_min=0.0,
        w_max=1.0,
        length=4096,
        y_min=0.0,
        y_max=0.1,
        dt_ms=1.0,
        neuron_params=dict(
            v_rest=0.0,
            v_reset=0.0,
            v_threshold=1.0,
            tau_mem=25.0,
            noise_std=0.03,
            resistance=1.0,
        ),
        base_seed=42,
    )

    T = 3000
    layer.reset()
    layer.run(T)

    spikes = layer.get_spike_trains()
    summary = layer.summary()

    print("Spike matrix shape:", spikes.shape)
    print("Layer summary:")
    for s in summary["stats"]:
        print(
            f"  Neuron {s['neuron']}: "
            f"total_spikes={s['total_spikes']}, "
            f"firing_rate_hz={s['firing_rate_hz']:.2f}"
        )
    print("Average firing rate (Hz):", summary["avg_firing_rate_hz"])


if __name__ == "__main__":
    demo()
