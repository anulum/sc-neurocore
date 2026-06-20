// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go batch mixed-precision Q8.8×Q16.16 dense MAC

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libmixed_dense.so`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `mixed_dense_forward_batch_q88_q1616_c` produces bit-identical
// outputs, overflow and underflow flags as the Rust, Julia, Mojo and Python
// references. Q8.8 weights contract Q16.16 input codes in an int64 accumulator
// (the caller keeps the contraction within int64 range); the accumulator divides
// by the Q8.8 weight scale with an arithmetic right shift (floor division) and
// saturates to the Q16.16 code range. The arithmetic is exact integer, so the
// parity tolerance is zero.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

const (
	weightFraction = 8
	i32Max         = int64(2147483647)
	i32Min         = int64(-2147483648)
)

// mixed_dense_forward_batch_q88_q1616_c — C-ABI entry point.
//
// Caller passes:
//
//	nOutputs, nInputs, nBatch — batch shape
//	weights (int16*)          — row-major nOutputs*nInputs Q8.8 weights
//	inputs (int32*)           — row-major nBatch*nInputs Q16.16 codes
//	outputs (int32*), overflow (uint8*), underflow (uint8*) — nBatch*nOutputs (written)
//
// Returns 0 on success, 1 on a non-positive shape.
//
//export mixed_dense_forward_batch_q88_q1616_c
func mixed_dense_forward_batch_q88_q1616_c(
	nOutputs, nInputs, nBatch C.int,
	weightsPtr, inputsPtr unsafe.Pointer,
	outputsPtr, overflowPtr, underflowPtr unsafe.Pointer,
) C.int {
	no := int(nOutputs)
	ni := int(nInputs)
	nb := int(nBatch)
	if no <= 0 || ni <= 0 || nb <= 0 {
		return 1
	}
	weights := unsafe.Slice((*C.int16_t)(weightsPtr), no*ni)
	inputs := unsafe.Slice((*C.int32_t)(inputsPtr), nb*ni)
	outputs := unsafe.Slice((*C.int32_t)(outputsPtr), nb*no)
	overflow := unsafe.Slice((*C.uint8_t)(overflowPtr), nb*no)
	underflow := unsafe.Slice((*C.uint8_t)(underflowPtr), nb*no)

	for b := 0; b < nb; b++ {
		inputRow := b * ni
		for o := 0; o < no; o++ {
			weightRow := o * ni
			var sum int64
			for i := 0; i < ni; i++ {
				sum += int64(weights[weightRow+i]) * int64(inputs[inputRow+i])
			}
			scaled := sum >> weightFraction
			idx := b*no + o
			if scaled > i32Max {
				outputs[idx] = C.int32_t(i32Max)
				overflow[idx] = 1
				underflow[idx] = 0
			} else if scaled < i32Min {
				outputs[idx] = C.int32_t(i32Min)
				overflow[idx] = 1
				underflow[idx] = 0
			} else {
				outputs[idx] = C.int32_t(scaled)
				overflow[idx] = 0
				if sum != 0 && scaled == 0 {
					underflow[idx] = 1
				} else {
					underflow[idx] = 0
				}
			}
		}
	}
	return 0
}

func main() {}
