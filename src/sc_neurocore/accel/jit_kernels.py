import numpy as np
import warnings

# Try to import Numba
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator: returns the original function
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    warnings.warn("Numba not found. Using pure Python fallback. Install 'numba' for high performance.")

@jit(nopython=True)
def jit_pack_bits(bitstream: np.ndarray, packed_arr: np.ndarray):
    """
    Packs a uint8 bitstream into uint64 array.
    bitstream: (N,) uint8 {0, 1}
    packed_arr: (N//64,) uint64
    """
    n = bitstream.size
    n_packed = n // 64
    
    for i in range(n_packed):
        val = np.uint64(0)
        base = i * 64
        for j in range(64):
            if bitstream[base + j] > 0:
                val |= (np.uint64(1) << np.uint64(j))
        packed_arr[i] = val

@jit(nopython=True)
def jit_vec_mac(packed_weights: np.ndarray, packed_inputs: np.ndarray, outputs: np.ndarray):
    """
    Vectorized Multiply-Accumulate (MAC).
    Simulates: Output[i] = Sum(Weights[i] AND Inputs)
    weights: (n_neurons, n_inputs, n_words)
    inputs: (n_inputs, n_words)
    outputs: (n_neurons,)
    """
    n_neurons = packed_weights.shape[0]
    n_inputs = packed_weights.shape[1]
    n_words = packed_weights.shape[2]
    
    for i in range(n_neurons):
        total_bits = 0
        for j in range(n_inputs):
            for k in range(n_words):
                # Bitwise AND = SC Multiplication
                res = packed_weights[i, j, k] & packed_inputs[j, k]
                
                # Popcount (Hamming Weight)
                # SWAR Algorithm for 64-bit popcount (Safe for Numba nopython mode)
                x = res
                x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
                x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
                x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
                x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
                
                total_bits += x
        outputs[i] = total_bits