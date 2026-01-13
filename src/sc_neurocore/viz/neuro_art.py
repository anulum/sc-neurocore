
import numpy as np
from dataclasses import dataclass

@dataclass
class NeuroArtGenerator:
    """
    Generates Art (Images) from Neural State.
    """
    resolution: int = 256
    
    def generate_visual(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Maps a 1D state vector to a 2D RGB abstract image.
        Uses the state to seed a generative pattern.
        """
        # Seed random generator with state hash to be deterministic per state
        # but chaotic
        seed = int(np.sum(np.abs(state_vector)) * 10000) % (2**32)
        rng = np.random.default_rng(seed)
        
        # Create base canvas
        img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        # 'Painters' driven by state elements
        num_painters = min(10, len(state_vector))
        
        for i in range(num_painters):
            val = state_vector[i]
            # Map value to color
            color = rng.integers(0, 255, 3)
            # Map value to position/size
            x = rng.integers(0, self.resolution)
            y = rng.integers(0, self.resolution)
            radius = int(abs(val) * 50) + 5
            
            # Draw circle (naive)
            y_grid, x_grid = np.ogrid[:self.resolution, :self.resolution]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
            img[mask] = color
            
        return img
