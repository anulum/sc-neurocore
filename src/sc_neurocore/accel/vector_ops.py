
import numpy as np

def pack_bitstream(bitstream: np.ndarray) -> np.ndarray:
    """
    Packs a uint8 bitstream (0s and 1s) into uint64 integers.
    This allows processing 64 time steps in parallel.
    
    Args:
        bitstream: Shape (N,) or (Batch, N) of uint8 {0,1}
        
    Returns:
        packed: Shape (ceil(N/64),) or (Batch, ceil(N/64)) of uint64
    """
    bitstream = np.asarray(bitstream, dtype=np.uint8)
    flat = bitstream.flatten()
    length = flat.size
    
    # Pad to multiple of 64
    pad_len = (64 - (length % 64)) % 64
    if pad_len > 0:
        flat = np.append(flat, np.zeros(pad_len, dtype=np.uint8))
        
    # Reshape to chunks of 64
    chunks = flat.reshape(-1, 64)
    
    # Pack bits to uint64
    # We multiply each bit by powers of 2 and sum
    powers = 1 << np.arange(64, dtype=np.uint64)
    packed = (chunks * powers).sum(axis=1, dtype=np.uint64)
    
    if bitstream.ndim > 1:
        return packed.reshape(bitstream.shape[0], -1)
    return packed

def unpack_bitstream(packed: np.ndarray, original_length: int) -> np.ndarray:
    """
    Unpacks uint64 array back to uint8 bitstream.
    """
    packed_flat = packed.flatten()
    
    # Extract bits
    # Shape: (num_packed, 64)
    bits = ((packed_flat[:, None] & (1 << np.arange(64, dtype=np.uint64))) > 0).astype(np.uint8)
    
    unpacked = bits.flatten()
    return unpacked[:original_length]

def vec_and(a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
    """
    Bitwise AND on packed arrays. Simulates SC Multiplication.
    """
    return np.bitwise_and(a_packed, b_packed)

def vec_popcount(packed: np.ndarray) -> int:
    """
    Count total set bits (1s) in the packed array.
    Used for integration/accumulation.
    """
    # Using numpy's ability to cast to specialized types or simple lookup?
    # Actually, Python 3.10+ int.bit_count() is fast, but for numpy arrays:
    # We can use a trick or just loop if C-extension isn't available.
    # A generic parallel popcount on uint64 in pure numpy is tricky without looping or lookup tables.
    # However, we can map to python int and sum.
    
    # For speed in pure python/numpy env without heavy deps:
    # Use binary decomposition for vectorized popcount
    x = packed.copy()
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    x = (x * 0x0101010101010101) >> 56
    return np.sum(x)
