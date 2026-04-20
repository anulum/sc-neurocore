# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for vectorized_layer

fn _get_scipy_sparse() -> Int:
    var __get_scipy_sparse_line = 'global _scipy_sparse'
    var __get_scipy_sparse_line = 'if _scipy_sparse is 0:'
    var __get_scipy_sparse_line = 'import scipy.sparse'
    var __get_scipy_sparse_line = '_scipy_sparse = scipy.sparse'
    return 0  # return _scipy_sparse

fn _has_scipy_sparse() -> Int:
    var __has_scipy_sparse_line = 'try:'
    var __has_scipy_sparse_line = '_get_scipy_sparse()'
    return 0  # return True
    var __has_scipy_sparse_line = 'except ImportError:  # pragma: no cover'
    return 0  # return False

fn _popcount_rows(packed: Int) -> Int:
    var __popcount_rows_line = 'x = packed.astype(uint64).copy()'
    var __popcount_rows_line = 'm1 = uint64(0x5555555555555555)'
    var __popcount_rows_line = 'm2 = uint64(0x3333333333333333)'
    var __popcount_rows_line = 'm4 = uint64(0x0F0F0F0F0F0F0F0F)'
    var __popcount_rows_line = 'h01 = uint64(0x0101010101010101)'
    var __popcount_rows_line = 'x -= (x >> uint64(1)) & m1'
    var __popcount_rows_line = 'x = (x & m2) + ((x >> uint64(2)) & m2)'
    var __popcount_rows_line = 'x = (x + (x >> uint64(4))) & m4'
    var __popcount_rows_line = 'x = (x * h01) >> uint64(56)'
    var __popcount_rows_line = 'result: ndarray[Any, Any] = x.sum(axis=1).astype(float64)'
    return 0  # return result

fn _refresh_packed_weights() -> Int:
    var __refresh_packed_weights_line = 'w_probs = weights'
    var __refresh_packed_weights_line = 'bits = ('
    var __refresh_packed_weights_line = 'random.random((n_neurons, n_inputs, length)) < w_probs[:, :,'
    var __refresh_packed_weights_line = ').astype(uint8)'
    var __refresh_packed_weights_line = 'flat = bits.reshape(-1, length)'
    var __refresh_packed_weights_line = 'packed_flat = pack_bitstream(flat)'
    var __refresh_packed_weights_line = 'pw = packed_flat.reshape(n_neurons, n_inputs, -1)'
    var __refresh_packed_weights_line = 'if _on_gpu:  # pragma: no cover'
    var __refresh_packed_weights_line = 'packed_weights = to_device(pw)'
    var __refresh_packed_weights_line = 'else:'
    var __refresh_packed_weights_line = 'packed_weights = pw'
    return 0

fn _init_sparse() -> Int:
    var __init_sparse_line = 'sp = _get_scipy_sparse()'
    var __init_sparse_line = 'n_total = n_neurons * n_inputs'
    var __init_sparse_line = 'n_nonzero = max(1, int(round(n_total * connectivity)))'
    var __init_sparse_line = 'indices = random.choice(n_total, size=n_nonzero, replace=Fal'
    var __init_sparse_line = 'rows, cols = divmod(indices, n_inputs)'
    var __init_sparse_line = 'weight_vals = random.uniform(0.0, 1.0, n_nonzero)'
    var __init_sparse_line = 'mask_csr = sp.csr_matrix('
    var __init_sparse_line = '(ones(n_nonzero, dtype=float32), (rows, cols)),'
    var __init_sparse_line = 'shape=(n_neurons, n_inputs),'
    var __init_sparse_line = ')'
    var __init_sparse_line = 'weights_csr = sp.csr_matrix('
    var __init_sparse_line = '(weight_vals, (rows, cols)),'
    var __init_sparse_line = 'shape=(n_neurons, n_inputs),'
    var __init_sparse_line = ')'
    var __init_sparse_line = '_pack_sparse_weights()'
    return 0

fn _pack_sparse_weights() -> Int:
    var __pack_sparse_weights_line = 'csr = weights_csr'
    var __pack_sparse_weights_line = 'n_words = (length + 63) // 64'
    var __pack_sparse_weights_line = '_sparse_packed = empty((csr.nnz, n_words), dtype=uint64)'
    var __pack_sparse_weights_line = 'for k in range(csr.nnz):'
    var __pack_sparse_weights_line = 'w = csr.data[k]'
    var __pack_sparse_weights_line = 'bits = (random.random(length) < w).astype(uint8)'
    var __pack_sparse_weights_line = '_sparse_packed[k] = pack_bitstream(bits)'
    return 0

fn forward(input_values: Int) -> Int:
    var _forward_line = 'in_probs = asarray(input_values, dtype=float64)'
    var _forward_line = 'if in_probs.ndim != 1 or in_probs.shape[0] != n_inputs:'
    var _forward_line = 'raise ValueError('
    var _forward_line = 'f"Expected 1-D input of length {n_inputs}, got shape {in_pro'
    var _forward_line = ')'
    var _forward_line = 'if not all(isfinite(in_probs)):'
    var _forward_line = 'raise ValueError("Input contains NaN or Inf")'
    var _forward_line = 'if any(in_probs < 0.0) or any(in_probs > 1.0):'
    var _forward_line = 'raise ValueError("Input probabilities must be in [0, 1]")'
    var _forward_line = 'if sparse:'
    return 0  # return _forward_sparse(in_probs)
    return 0  # return _forward_dense(in_probs)

fn _forward_dense(in_probs: Int) -> Int:
    var __forward_dense_line = 'input_bits = (random.random((n_inputs, length)) < in_probs[:'
    var __forward_dense_line = 'uint8'
    var __forward_dense_line = ')'
    var __forward_dense_line = 'packed_inputs = pack_bitstream(input_bits)'
    var __forward_dense_line = 'if _on_gpu:  # pragma: no cover'
    var __forward_dense_line = 'packed_inputs_dev = to_device(packed_inputs)'
    var __forward_dense_line = 'counts = gpu_vec_mac(packed_weights, packed_inputs_dev)'
    var __forward_dense_line = 'outputs = to_host(counts).astype(float64)'
    var __forward_dense_line = 'else:'
    var __forward_dense_line = 'assert packed_weights is not 0'
    var __forward_dense_line = 'products = vec_and(packed_weights, packed_inputs[0, :, :])'
    var __forward_dense_line = 'flat_products = products.reshape(n_neurons, -1)'
    var __forward_dense_line = 'outputs = _popcount_rows(flat_products)'
    return 0  # return outputs / length

fn _forward_sparse(in_probs: Int) -> Int:
    var __forward_sparse_line = 'input_bits = (random.random((n_inputs, length)) < in_probs[:'
    var __forward_sparse_line = 'uint8'
    var __forward_sparse_line = ')'
    var __forward_sparse_line = 'packed_inputs = pack_bitstream(input_bits)'
    var __forward_sparse_line = 'csr = weights_csr'
    var __forward_sparse_line = 'if csr.nnz == 0:  # pragma: no cover'
    return 0  # return zeros(n_neurons, dtype=float64)
    var __forward_sparse_line = 'if _on_gpu:  # pragma: no cover'
    return 0  # return _forward_sparse_gpu(packed_inputs)
    var __forward_sparse_line = 'gathered_inputs = packed_inputs[csr.indices]'
    var __forward_sparse_line = 'products = vec_and(_sparse_packed, gathered_inputs)'
    var __forward_sparse_line = 'counts = _popcount_rows(products)'
    var __forward_sparse_line = 'outputs = zeros(n_neurons, dtype=float64)'
    var __forward_sparse_line = 'add.at(outputs, repeat(arange(n_neurons), diff(csr.indptr)),'
    return 0  # return outputs / length

fn _forward_sparse_gpu(packed_inputs: Int) -> Int:
    var __forward_sparse_gpu_line = 'self, packed_inputs: ndarray[Any, Any]'
    var __forward_sparse_gpu_line = ') -> ndarray[Any, Any]:  # pragma: no cover'
    var __forward_sparse_gpu_line = 'import cupy'
    var __forward_sparse_gpu_line = 'import cupyx.scipy.sparse as cusp'
    var __forward_sparse_gpu_line = 'csr = weights_csr'
    var __forward_sparse_gpu_line = 'w_gpu = cusp.csr_matrix('
    var __forward_sparse_gpu_line = '('
    var __forward_sparse_gpu_line = 'cupy.asarray(csr.data.astype(float32)),'
    var __forward_sparse_gpu_line = 'cupy.asarray(csr.indices),'
    var __forward_sparse_gpu_line = 'cupy.asarray(csr.indptr),'
    var __forward_sparse_gpu_line = '),'
    var __forward_sparse_gpu_line = 'shape=csr.shape,'
    var __forward_sparse_gpu_line = ')'
    var __forward_sparse_gpu_line = 'in_probs_flat = _popcount_rows(packed_inputs).astype(float32'
    var __forward_sparse_gpu_line = 'in_gpu = cupy.asarray(in_probs_flat)'
    var __forward_sparse_gpu_line = 'out_gpu = w_gpu @ in_gpu'
    var __forward_sparse_gpu_line = 'result: ndarray[Any, Any] = cupy.asnumpy(out_gpu).astype(flo'
    return 0  # return result
