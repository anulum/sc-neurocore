# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from typing import Any
import numpy as np
from dataclasses import dataclass


@dataclass
class BCIDecoder:
    """
    Brain-Computer Interface Decoder.
    Converts continuous neural signals (e.g., EEG, LFP) into SC Bitstreams.
    """

    channels: int
    sampling_rate: int = 1000

    def normalize_signal(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Normalize signal to [0, 1] for probability encoding.
        """
        s_min = np.min(signal)
        s_max = np.max(signal)
        if s_max - s_min == 0:
            return np.zeros_like(signal)
        return (signal - s_min) / (s_max - s_min)

    def encode_to_bitstream(
        self, signal: np.ndarray[Any, Any], length: int = 256
    ) -> np.ndarray[Any, Any]:
        """
        Encodes a [Channels, Time] signal block into [Channels, Bitstream_Length].
        We assume 'signal' represents the mean firing rate/amplitude for this window.
        """
        # Take mean amplitude over the window per channel
        # signal: (Channels, Time) -> (Channels,)
        if signal.ndim > 1:
            mean_vals = np.mean(signal, axis=1)
        else:
            mean_vals = signal

        probs = self.normalize_signal(mean_vals)

        # Generate bitstreams
        # (Channels, Length)
        rands = np.random.random((self.channels, length))
        bits = (rands < probs[:, None]).astype(np.uint8)

        return bits
