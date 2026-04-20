# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/vectorized_layer

module VectorizedLayerAccel

using Statistics, LinearAlgebra

mutable struct VectorizedSCLayerState
    n_inputs::Float64
    n_neurons::Float64
    length::Float64
    use_gpu::Float64
    sparse::Float64
    connectivity::Float64
end

function VectorizedSCLayerState()
    VectorizedSCLayerState(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
end

function _refresh_packed_weights(s::VectorizedSCLayerState)
    w_probs = s.weights
    bits = (
        np.random.random((s.n_neurons, s.n_inputs, s.length)) < w_probs[:, :, nothing]
    ).astype(np.uint8)
    flat = bits.reshape(-1, s.length)
    packed_flat = pack_bitstream(flat)
    pw = packed_flat.reshape(s.n_neurons, s.n_inputs, -1)
    if s._on_gpu:  # pragma: no cover
        s.packed_weights = to_device(pw)
    else
        s.packed_weights = pw
end

function _init_sparse(s::VectorizedSCLayerState)
    sp = _get_scipy_sparse()
    n_total = s.n_neurons * s.n_inputs
    n_nonzero = max(1, int(round(n_total * s.connectivity)))
    indices = np.random.choice(n_total, size=n_nonzero, replace=false)
    rows, cols = np.divmod(indices, s.n_inputs)
    weight_vals = np.random.uniform(0.0, 1.0, n_nonzero)
    s.mask_csr = sp.csr_matrix(
        (ones(n_nonzero, dtype=np.float32), (rows, cols)),
        shape=(s.n_neurons, s.n_inputs),
    )
    s.weights_csr = sp.csr_matrix(
        (weight_vals, (rows, cols)),
        shape=(s.n_neurons, s.n_inputs),
    )
    s._pack_sparse_weights()
end

function _pack_sparse_weights(s::VectorizedSCLayerState)
    csr = s.weights_csr
    n_words = (s.length + 63) // 64
    s._sparse_packed = np.empty((csr.nnz, n_words), dtype=np.uint64)
    for k in 1:csr.nnz
        w = csr.data[k]
        bits = (np.random.random(s.length) < w).astype(np.uint8)
        s._sparse_packed[k] = pack_bitstream(bits)
end

function forward(s::VectorizedSCLayerState, input_values)
    in_probs = np.asarray(input_values, dtype=np.float64)
    if in_probs.ndim != 1 || in_probs.shape[0] != s.n_inputs
        raise ValueError(
            f"Expected 1-D input of length {s.n_inputs}, got shape {in_probs.shape}"
        )
    if ! np.all(np.isfinite(in_probs))
        raise ValueError("Input contains NaN || Inf")
    if np.any(in_probs < 0.0) || np.any(in_probs > 1.0)
        raise ValueError("Input probabilities must be in [0, 1]")
    if s.sparse
        return s._forward_sparse(in_probs)
    return s._forward_dense(in_probs)
end

function _forward_dense(s::VectorizedSCLayerState, in_probs, Any])
    input_bits = (np.random.random((s.n_inputs, s.length)) < in_probs[:, nothing]).astype(
        np.uint8
    )
    packed_inputs = pack_bitstream(input_bits)
    if s._on_gpu:  # pragma: no cover
        packed_inputs_dev = to_device(packed_inputs)
        counts = gpu_vec_mac(s.packed_weights, packed_inputs_dev)
        outputs = to_host(counts).astype(np.float64)
    else
        assert s.packed_weights is ! nothing
        products = vec_and(s.packed_weights, packed_inputs[nothing, :, :])
        flat_products = products.reshape(s.n_neurons, -1)
        outputs = _popcount_rows(flat_products)
    return outputs / s.length
end

function _forward_sparse(s::VectorizedSCLayerState, in_probs, Any])
    input_bits = (np.random.random((s.n_inputs, s.length)) < in_probs[:, nothing]).astype(
        np.uint8
    )
    packed_inputs = pack_bitstream(input_bits)
    csr = s.weights_csr
    if csr.nnz == 0:  # pragma: no cover
        return zeros(s.n_neurons, dtype=np.float64)
    if s._on_gpu:  # pragma: no cover
        return s._forward_sparse_gpu(packed_inputs)
    gathered_inputs = packed_inputs[csr.indices]
    products = vec_and(s._sparse_packed, gathered_inputs)
    counts = _popcount_rows(products)
    outputs = zeros(s.n_neurons, dtype=np.float64)
    np.add.at(outputs, np.repeat(collect(s.n_neurons), diff(csr.indptr)), counts)
    return outputs / s.length
end

function _forward_sparse_gpu(s::VectorizedSCLayerState)
    self, packed_inputs: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:  # pragma: no cover
    import cupy
    import cupyx.scipy.sparse as cusp
    csr = s.weights_csr
    w_gpu = cusp.csr_matrix(
        (
            cupy.asarray(csr.data.astype(np.float32)),
            cupy.asarray(csr.indices),
            cupy.asarray(csr.indptr),
        ),
        shape=csr.shape,
    )
    in_probs_flat = _popcount_rows(packed_inputs).astype(np.float32) / s.length
    in_gpu = cupy.asarray(in_probs_flat)
    out_gpu = w_gpu @ in_gpu
    result: np.ndarray[Any, Any] = cupy.asnumpy(out_gpu).astype(np.float64)
    return result
end

end # module VectorizedLayerAccel
