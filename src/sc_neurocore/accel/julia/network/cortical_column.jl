# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/cortical_column

module CorticalColumnAccel

using Base.Threads
using PythonCall

# Computes batched parallel CSR multi SpMV add.
# All pointer arrays are passed directly from Python as flat 1D arrays of UInt64 
# representing memory boundaries, providing zero-overhead pointer unwrapping.
function py_parallel_csr_multi_spmv_add(
    n_blocks::Int,
    n_rows::Int,
    indptr_ptrs_py,
    indices_ptrs_py,
    data_ptrs_py,
    x_ptrs_py,
    x_lens_py,
    y_ptr_py
)
    # Convert numpy pointer arrays to Julia arrays of UInt (cheap, only `n_blocks` elements)
    indptr_ptrs = pyconvert(Vector{UInt}, indptr_ptrs_py)
    indices_ptrs = pyconvert(Vector{UInt}, indices_ptrs_py)
    data_ptrs = pyconvert(Vector{UInt}, data_ptrs_py)
    x_ptrs = pyconvert(Vector{UInt}, x_ptrs_py)
    x_lens = pyconvert(Vector{Int}, x_lens_py)
    y_addr = pyconvert(UInt, y_ptr_py)

    # Wrap the output vector `y`
    y_arr = unsafe_wrap(Array, Ptr{Float64}(y_addr), n_rows; own=false)

    # Wrap all block arrays at the top level to form strongly-typed Vectors
    indptrs = Vector{Vector{Int32}}(undef, n_blocks)
    indices_arrs = Vector{Vector{Int32}}(undef, n_blocks)
    data_arrs = Vector{Vector{Float64}}(undef, n_blocks)
    x_arrs = Vector{Vector{Float64}}(undef, n_blocks)

    for b in 1:n_blocks
        # indptr always has length n_rows + 1
        indptrs[b] = unsafe_wrap(Array, Ptr{Int32}(indptr_ptrs[b]), n_rows + 1; own=false)
        
        # We find nnz from the last element of the wrapped indptr array
        nnz = indptrs[b][n_rows + 1] # 0-indexed in python, so it equals the size
        
        indices_arrs[b] = unsafe_wrap(Array, Ptr{Int32}(indices_ptrs[b]), nnz; own=false)
        data_arrs[b] = unsafe_wrap(Array, Ptr{Float64}(data_ptrs[b]), nnz; own=false)
        x_arrs[b] = unsafe_wrap(Array, Ptr{Float64}(x_ptrs[b]), x_lens[b]; own=false)
    end

    # The actual compute core
    Threads.@threads for r in 1:n_rows
        sum_val::Float64 = 0.0
        
        for b in 1:n_blocks
            indptr = indptrs[b]
            indices = indices_arrs[b]
            data = data_arrs[b]
            x = x_arrs[b]

            start_idx = indptr[r]
            end_idx = indptr[r + 1]

            for k in start_idx:(end_idx - 1)
                col = indices[k + 1]      # 0-indexed column
                sum_val += data[k + 1] * x[col + 1]
            end
        end
        
        y_arr[r] += sum_val
    end

    return nothing
end

end # module CorticalColumnAccel
