# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drive a StochasticLIFNeuron with a bitstream-encoded

from __future__ import annotations
from typing import Any
from typing import Tuple
import numpy as np

from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..utils.bitstreams import (
    BitstreamEncoder,
    bitstream_to_probability,
)


def run_bitstream_driven_lif(
    x_input: float,
    x_min: float = 0.0,
    x_max: float = 0.1,
    length: int = 1024,
    neuron_params: dict[str, Any] | None = None,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, float]:
    """
    Drive a StochasticLIFNeuron with a bitstream-encoded input current.

    Steps:
    1. Encode scalar input current x_input in [x_min, x_max] as a unipolar
       bitstream of length `length`.
    2. At each time step t, set:
       I_t = I_high if bitstream[t] == 1 else I_low
       or more simply, treat the bit directly as a scaled current.
    3. Run neuron for `length` steps, collect spike bitstream.
    4. Estimate:
       - input probability p_in from the input bitstream
       - firing probability p_fire from the spike bitstream

    Returns
    -------
    input_bits : np.ndarray
        Input bitstream (0/1).
    spike_bits : np.ndarray
        Output spike bitstream (0/1).
    p_in : float
        Estimated input probability.
    p_fire : float
        Estimated firing probability.
    """
    if neuron_params is None:
        neuron_params = {}

    encoder = BitstreamEncoder(x_min=x_min, x_max=x_max, length=length, seed=123)
    input_bits = encoder.encode(x_input)

    neuron = StochasticLIFNeuron(**neuron_params)

    spike_bits = np.zeros(length, dtype=np.uint8)
    for t in range(length):
        # Simple mapping: 1 -> x_max, 0 -> x_min
        # You can choose something more nuanced if desired.
        I_t = encoder.decode(np.array([input_bits[t]], dtype=np.uint8))
        spike_bits[t] = neuron.step(I_t)

    # Decode probabilities
    p_in = bitstream_to_probability(input_bits)
    p_fire = bitstream_to_probability(spike_bits)

    return input_bits, spike_bits, p_in, p_fire


def demo() -> None:
    neuron_params = dict(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,
        noise_std=0.05,
        resistance=1.0,
        seed=42,
    )

    x_min = 0.0
    x_max = 0.1
    x_input = 0.06  # some current in physical space
    length = 2000

    input_bits, spike_bits, p_in, p_fire = run_bitstream_driven_lif(
        x_input=x_input,
        x_min=x_min,
        x_max=x_max,
        length=length,
        neuron_params=neuron_params,
    )

    print(f"Input scalar x_input = {x_input}")
    print(f"Estimated input p_in = {p_in:.3f}")
    print(f"Estimated firing p_fire = {p_fire:.3f}")
    print(f"Total spikes: {spike_bits.sum()} / {length}")


if __name__ == "__main__":
    demo()
