# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo acceleration for network/cortical_column

from memory import UnsafePointer
from algorithm import parallelize

alias CHUNK_SIZE = 512

@ffi.export("py_parallel_csr_multi_spmv_add_c")
fn py_parallel_csr_multi_spmv_add_c(
    n_blocks: Int32,
    n_rows: Int32,
    indptr_ptrs: UnsafePointer[UnsafePointer[Int32]],
    indices_ptrs: UnsafePointer[UnsafePointer[Int32]],
    data_ptrs: UnsafePointer[UnsafePointer[Float64]],
    x_ptrs: UnsafePointer[UnsafePointer[Float64]],
    x_lens: UnsafePointer[Int32],
    y_ptr: UnsafePointer[Float64],
):
    var blocks = int(n_blocks)
    var rows = int(n_rows)

    if blocks == 0 or rows == 0:
        return

    # Total chunks to process
    var num_chunks = (rows + CHUNK_SIZE - 1) // CHUNK_SIZE

    @parameter
    fn process_chunk(chunk_idx: Int):
        var start_row = chunk_idx * CHUNK_SIZE
        var end_row = start_row + CHUNK_SIZE
        if end_row > rows:
            end_row = rows

        for r in range(start_row, end_row):
            var row_sum: Float64 = 0.0

            for b in range(blocks):
                var indptr = indptr_ptrs[b]
                var indices = indices_ptrs[b]
                var data = data_ptrs[b]
                var x = x_ptrs[b]

                var start_idx = int(indptr[r])
                var end_idx = int(indptr[r + 1])

                for k in range(start_idx, end_idx):
                    var col = int(indices[k])
                    row_sum += data[k] * x[col]

            y_ptr[r] += row_sum

    parallelize[process_chunk](num_chunks)
