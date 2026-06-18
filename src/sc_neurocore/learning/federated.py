# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy-Preserving Federated Learning using SC Bitstreams

"""Privacy-preserving federated aggregation over stochastic-computing bitstreams."""

from __future__ import annotations

from typing import Any

import numpy as np


class FederatedAggregator:
    """Privacy-preserving federated learning using SC bitstreams."""

    @staticmethod
    def aggregate_gradients(client_gradients: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Aggregate gradient bitstreams from multiple clients by majority vote.

        Parameters
        ----------
        client_gradients : list of numpy.ndarray
            Per-client bitstream arrays; all must share the same shape.

        Returns
        -------
        numpy.ndarray
            The majority-voted aggregated bitstream.
        """
        if not client_gradients:
            raise ValueError("No gradients to aggregate")

        # Stack: (Num_Clients, Gradient_Size)
        stack = np.stack(client_gradients, axis=0)

        # Sum bits at each position across clients
        # (Client1_bit_i + Client2_bit_i + ... )
        sums = np.sum(stack, axis=0)

        # Majority Vote
        # If sum > num_clients / 2, output 1
        threshold = len(client_gradients) / 2.0

        aggregated: np.ndarray[Any, Any] = (sums > threshold).astype(np.uint8)

        return aggregated

    @staticmethod
    def secure_sum_protocol(client_gradients: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Sum client bitstreams as a secure-aggregation surrogate.

        Models a secure aggregation where the server observes only the
        element-wise sum, not individual client updates, analogous to secure
        multi-party computation.
        """
        # In SC, 'Summing' bitstreams usually produces an integer result (0..N).
        # This is strictly not a bitstream anymore but a discretized value.
        stack = np.stack(client_gradients, axis=0)
        summed: np.ndarray[Any, Any] = np.sum(stack, axis=0)
        return summed
