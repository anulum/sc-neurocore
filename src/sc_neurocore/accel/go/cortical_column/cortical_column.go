// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go cgo acceleration for network/cortical_column

package main

/*
#include <stdint.h>
*/
import "C"

import (
	"sync"
	"unsafe"
)

const CHUNK_SIZE = 512

//export py_parallel_csr_multi_spmv_add_c
func py_parallel_csr_multi_spmv_add_c(
	n_blocks int32,
	n_rows int32,
	indptr_ptrs **int32,
	indices_ptrs **int32,
	data_ptrs **float64,
	x_ptrs **float64,
	x_lens *int32,
	y_ptr *float64,
) {
	blocks := int(n_blocks)
	rows := int(n_rows)

	if blocks == 0 || rows == 0 {
		return
	}

	// Unpack the C pointer arrays into slice of pointers
	indptrs := unsafe.Slice(indptr_ptrs, blocks)
	indices := unsafe.Slice(indices_ptrs, blocks)
	datas := unsafe.Slice(data_ptrs, blocks)
	xs := unsafe.Slice(x_ptrs, blocks)
	xlens := unsafe.Slice(x_lens, blocks)

	y := unsafe.Slice(y_ptr, rows)

	var wg sync.WaitGroup

	// Parallel iteration over row chunks
	for chunkStart := 0; chunkStart < rows; chunkStart += CHUNK_SIZE {
		wg.Add(1)

		chunkEnd := chunkStart + CHUNK_SIZE
		if chunkEnd > rows {
			chunkEnd = rows
		}

		go func(startRow, endRow int) {
			defer wg.Done()

			// Pre-slice x-blocks so we don't do it in the inner loop
			xB := make([][]float64, blocks)
			for b := 0; b < blocks; b++ {
				xB[b] = unsafe.Slice(xs[b], int(xlens[b]))
			}

			// indptr arrays always have length rows + 1
			indptrB := make([][]int32, blocks)
			for b := 0; b < blocks; b++ {
				indptrB[b] = unsafe.Slice(indptrs[b], rows+1)
			}

			for r := startRow; r < endRow; r++ {
				var sum float64 = 0.0

				for b := 0; b < blocks; b++ {
					startIdx := int(indptrB[b][r])
					endIdx := int(indptrB[b][r+1])

					if startIdx == endIdx {
						continue // skip zero nnz row for this block
					}

					// We only strictly know the bounds up to endIdx
					indicesBlock := unsafe.Slice(indices[b], endIdx)
					dataBlock := unsafe.Slice(datas[b], endIdx)
					xBlock := xB[b]

					for k := startIdx; k < endIdx; k++ {
						col := int(indicesBlock[k])
						sum += dataBlock[k] * xBlock[col]
					}
				}
				y[r] += sum
			}
		}(chunkStart, chunkEnd)
	}

	wg.Wait()
}

func main() {}
