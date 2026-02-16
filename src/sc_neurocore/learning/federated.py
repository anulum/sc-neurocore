from __future__ import annotations

import numpy as np


class FederatedAggregator:
    """
    Privacy-Preserving Federated Learning using SC Bitstreams.
    """

    @staticmethod
    def aggregate_gradients(client_gradients: list[np.ndarray]) -> np.ndarray:
        """
        Aggregates gradient bitstreams from multiple clients.

        Args:
            client_gradients: List of numpy arrays (bitstreams).
                              All must have same shape.

        Returns:
            Aggregated bitstream (Majority Vote).
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

        aggregated = (sums > threshold).astype(np.uint8)

        return aggregated

    @staticmethod
    def secure_sum_protocol(client_gradients: list[np.ndarray]) -> np.ndarray:
        """
        Simulates a secure aggregation where the server only sees the sum,
        not individual updates (like Secure Multi-Party Computation).
        """
        # In SC, 'Summing' bitstreams usually produces an integer result (0..N).
        # This is strictly not a bitstream anymore but a discretized value.
        stack = np.stack(client_gradients, axis=0)
        return np.sum(stack, axis=0)
