
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass
class DVSInputLayer:
    """
    Interface for Dynamic Vision Sensors (Event Cameras).
    Converts AER events (x, y, t, p) into SC Bitstreams.
    """
    height: int
    width: int
    decay_tau: float = 100.0 # Time constant to decay old events
    
    def __post_init__(self):
        # Surface potential representing event density
        self.surface = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_update_time = 0.0

    def process_events(self, events: List[Tuple[int, int, float, int]]) -> np.ndarray:
        """
        Integrate a batch of events.
        Events format: (x, y, timestamp_ms, polarity)
        Returns: Frame of probabilities [0, 1]
        """
        if not events:
            return self.surface
            
        current_time = events[-1][2]
        dt = current_time - self.last_update_time
        
        # Exponential decay of old activity
        # V_new = V_old * exp(-dt/tau)
        decay_factor = np.exp(-dt / self.decay_tau)
        self.surface *= decay_factor
        
        # Add new events
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Polarity is usually -1 or 1.
                # We want activity map. Let's just accumulate magnitude or positive density.
                # For simplified SC vision, we map events to "Probability of Edge".
                self.surface[y, x] += 1.0 
        
        # Clip/Sigmoid to [0, 1] for SC generation
        # Simple saturation
        output_probs = np.tanh(self.surface) # Maps 0->0, High->1
        
        self.last_update_time = current_time
        return output_probs
        
    def generate_bitstream_frame(self, length: int = 256) -> np.ndarray:
        """
        Generate a HxWxLength bitstream cube from current surface state.
        """
        probs = np.tanh(self.surface)
        # Vectorized generation
        # (H, W, Length)
        rands = np.random.random((self.height, self.width, length))
        bits = (rands < probs[:, :, None]).astype(np.uint8)
        return bits
