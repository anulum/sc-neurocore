
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Optional

from ..accel.vector_ops import pack_bitstream, vec_and, vec_popcount
from ..utils.bitstreams import BitstreamEncoder

@dataclass
class VectorizedSCLayer:
    """
    High-Performance SC Layer using packed bitwise operations.
    Simulates thousands of neurons efficiently on CPU.
    """
    n_inputs: int
    n_neurons: int
    length: int = 1024
    
    def __post_init__(self):
        # Weights: (n_neurons, n_inputs)
        # We store weights in packed format for fast processing
        self.weights = np.random.uniform(0.0, 1.0, (self.n_neurons, self.n_inputs))
        self.packed_weights = None
        self._refresh_packed_weights()
        
    def _refresh_packed_weights(self):
        # Generate bitstreams for all weights
        # This is expensive, done only when weights change
        encoder = BitstreamEncoder(x_min=0, x_max=1, length=self.length)
        
        # We need a 3D array: (n_neurons, n_inputs, length)
        # Then pack -> (n_neurons, n_inputs, n_packed_words)
        
        # Optimization: use vectorized RNG
        w_probs = self.weights
        # bits: (N, I, L)
        bits = (np.random.random((self.n_neurons, self.n_inputs, self.length)) < w_probs[:, :, None]).astype(np.uint8)
        
        # Pack
        # We iterate to pack.
        # Ideally, we'd have a 3D pack function.
        # For now, let's flatten, pack, reshape.
        flat = bits.reshape(-1, self.length)
        packed_flat = pack_bitstream(flat)
        self.packed_weights = packed_flat.reshape(self.n_neurons, self.n_inputs, -1)

    def forward(self, input_values: Sequence[float]) -> np.ndarray:
        """
        Compute output firing rates for the layer.
        """
        # 1. Encode Inputs -> Packed Bitstreams
        # inputs: (n_inputs,)
        # input_bits: (n_inputs, length)
        in_probs = np.array(input_values)
        input_bits = (np.random.random((self.n_inputs, self.length)) < in_probs[:, None]).astype(np.uint8)
        
        # Pack inputs
        # packed_inputs: (n_inputs, n_words)
        packed_inputs = pack_bitstream(input_bits)
        
        # 2. SC Matrix Multiplication (Dense)
        # We need to broadcast inputs to all neurons
        # packed_weights: (n_neurons, n_inputs, n_words)
        # packed_inputs:  (1,         n_inputs, n_words)
        
        # Bitwise AND = Multiplication
        # products: (n_neurons, n_inputs, n_words)
        
        # Sparsity Optimization:
        # In hardware, if input block is 0, we gate the clock.
        # In software vectorization, checking for 0 might be overhead unless sparse.
        # But for 'Energy Efficiency' demo logic:
        
        # products = vec_and(self.packed_weights, packed_inputs[None, :, :])
        
        # We can implement a "Sparse Matrix Mul" manually or just apply the logic:
        # Only compute where input is not 0.
        
        # However, numpy's bitwise_and is extremely fast. 
        # Implementing a python-level 'if' per element is SLOWER.
        # We will demonstrate the *concept* by computing a 'mask' of active inputs first.
        
        # (This is a simulation of the hardware power saving, not necessarily wall-clock speedup in Python)
        
        # products = np.zeros_like(self.packed_weights)
        # valid_mask = (packed_inputs != 0)
        # products[:, valid_mask] = vec_and(self.packed_weights[:, valid_mask], packed_inputs[None, valid_mask])
        
        # For this codebase, let's stick to the fast calculation but return a "power_stats" metric.
        
        products = vec_and(self.packed_weights, packed_inputs[None, :, :])
        
        # Calculate 'Energy Saved' (Simulated)
        # active_inputs = np.count_nonzero(packed_inputs)
        # total_inputs = packed_inputs.size
        # savings = 1.0 - (active_inputs / total_inputs)
        
        # 3. Accumulation (Integration)
        # In SC, addition is usually MUX (scaled) or OR (saturating).
        # "Accumulation" usually refers to counting spikes to get a value.
        # Here we model a "Spiking Neuron" behavior:
        # We sum the bits (current) and check threshold?
        # Or simpler: Just return the dot-product value (mac).
        
        # Count bits per neuron per input -> Sum inputs
        # But wait, popcount on 'products' gives total correlation overlap.
        
        # Count set bits across the whole word dimension
        # result per neuron per input (value)
        # This is essentially: out = sum(w * x)
        
        # We want to do this efficiently.
        # Flatten last two dims? No, we need sum over inputs and time.
        
        # Let's count bits for each (neuron, input) pair first?
        # No, optimal is:
        # Sum of popcounts across all inputs and words for each neuron.
        
        # Flatten neuron's data: (n_neurons, n_inputs * n_words)
        flat_products = products.reshape(self.n_neurons, -1)
        
        # We can implement a vectorized popcount on this 2D array
        # But our vec_popcount takes a 1D or flat array sum.
        # Let's use map or loop for now, it's still 64x faster than bit-loop.
        
        outputs = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            outputs[i] = vec_popcount(flat_products[i])
            
        # Normalize: Total bits possible = n_inputs * length
        # But this is "Sum of Currents".
        # If we want 0-1 range, we divide.
        
        return outputs / self.length # Return "total current"
