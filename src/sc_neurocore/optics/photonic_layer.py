import numpy as np
from dataclasses import dataclass


@dataclass
class PhotonicBitstreamLayer:
    """
    Simulates a Photonic Stochastic Computing Layer.
    Uses Phase Noise (Laser Interference) to generate bitstreams.
    """

    n_channels: int
    laser_power: float = 1.0  # SNR control

    def simulate_interference(self, length: int) -> np.ndarray:
        """
        Simulates the interference of two laser beams with phase noise.
        I = I1 + I2 + 2*sqrt(I1*I2)*cos(phi)
        """
        # Phase noise phi: Wiener process or random uniform
        phi = np.random.uniform(0, 2 * np.pi, (self.n_channels, length))

        # Normalized intensity
        intensity = 0.5 + 0.5 * np.cos(phi)

        return intensity

    def forward(self, input_probs: np.ndarray, length: int = 1024) -> np.ndarray:
        """
        Generates bitstreams where '1' occurs if interference intensity < input_prob.
        """
        input_probs = np.asarray(input_probs)
        if input_probs.shape[0] != self.n_channels:
            raise ValueError(
                f"Input shape {input_probs.shape} does not match n_channels={self.n_channels}"
            )

        # input_probs: (n_channels,)
        intensities = self.simulate_interference(length)

        # Thresholding
        bits = (intensities < input_probs[:, None]).astype(np.uint8)

        return bits
