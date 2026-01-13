
import numpy as np
from dataclasses import dataclass

@dataclass
class ReactionDiffusionSolver:
    """
    Chemical Computing using Gray-Scott Reaction-Diffusion.
    """
    width: int
    height: int
    Da: float = 0.16
    Db: float = 0.08
    f: float = 0.060
    k: float = 0.062
    dt: float = 1.0
    
    def __post_init__(self):
        self.A = np.ones((self.height, self.width))
        self.B = np.zeros((self.height, self.width))
        
        # Seed
        r = 5
        cx, cy = self.width//2, self.height//2
        self.B[cy-r:cy+r, cx-r:cx+r] = 0.25 + 0.25*np.random.random((2*r, 2*r))

    def laplacian(self, M):
        # Finite difference Laplacian with periodic boundary
        top = np.roll(M, 1, axis=0)
        bottom = np.roll(M, -1, axis=0)
        left = np.roll(M, 1, axis=1)
        right = np.roll(M, -1, axis=1)
        return top + bottom + left + right - 4*M

    def step(self):
        La = self.laplacian(self.A)
        Lb = self.laplacian(self.B)
        
        # Reaction: A + 2B -> 3B
        reaction = self.A * (self.B ** 2)
        
        self.A += (self.Da * La - reaction + self.f * (1 - self.A)) * self.dt
        self.B += (self.Db * Lb + reaction - (self.k + self.f) * self.B) * self.dt
        
        self.A = np.clip(self.A, 0, 1)
        self.B = np.clip(self.B, 0, 1)
        
    def get_state(self):
        return self.B # Usually visualize B
