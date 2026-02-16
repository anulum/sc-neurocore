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

    if bitstream.ndim == 1:
        # 1D case: single bitstream
        length = bitstream.size
        pad_len = (64 - (length % 64)) % 64
        if pad_len > 0:
            bitstream = np.append(bitstream, np.zeros(pad_len, dtype=np.uint8))

        chunks = bitstream.reshape(-1, 64)
        powers = 1 << np.arange(64, dtype=np.uint64)
        packed = (chunks * powers).sum(axis=1, dtype=np.uint64)
        return packed

    elif bitstream.ndim == 2:
        # 2D case: batch of bitstreams
        batch_size, length = bitstream.shape
        pad_len = (64 - (length % 64)) % 64

        if pad_len > 0:
            padding = np.zeros((batch_size, pad_len), dtype=np.uint8)
            bitstream = np.concatenate([bitstream, padding], axis=1)

        # Reshape to (batch, num_chunks, 64)
        num_chunks = bitstream.shape[1] // 64
        chunks = bitstream.reshape(batch_size, num_chunks, 64)

        powers = 1 << np.arange(64, dtype=np.uint64)
        packed = (chunks * powers).sum(axis=2, dtype=np.uint64)
        return packed

    else:
        raise ValueError(f"Expected 1D or 2D array, got {bitstream.ndim}D")


def unpack_bitstream(
    packed: np.ndarray, original_length: int, original_shape: tuple = None
) -> np.ndarray:
    """
    Unpacks uint64 array back to uint8 bitstream.

    Args:
        packed: Packed uint64 array (1D or 2D)
        original_length: Total number of bits to extract
        original_shape: Optional tuple for reshaping output (batch, length)

    Returns:
        Unpacked bitstream of shape (original_length,) or original_shape
    """
    if packed.ndim == 1:
        # 1D packed array
        bits = ((packed[:, None] & (1 << np.arange(64, dtype=np.uint64))) > 0).astype(np.uint8)
        unpacked = bits.flatten()
        return unpacked[:original_length]

    elif packed.ndim == 2:
        # 2D packed array: (batch, num_chunks)
        batch_size, num_chunks = packed.shape
        # Extract bits: (batch, num_chunks, 64)
        bits = ((packed[:, :, None] & (1 << np.arange(64, dtype=np.uint64))) > 0).astype(np.uint8)
        # Reshape to (batch, num_chunks * 64)
        unpacked = bits.reshape(batch_size, -1)

        if original_shape is not None:
            return unpacked[:, : original_shape[1]]
        else:
            # Assume original_length is per-batch
            per_batch_len = original_length // batch_size
            return unpacked[:, :per_batch_len]

    else:
        raise ValueError(f"Expected 1D or 2D packed array, got {packed.ndim}D")


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
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x * 0x0101010101010101) >> 56
    return np.sum(x)
