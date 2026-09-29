// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go QCFS C boundary

// Package main exposes the QCFS array ABI version one. Every call validates
// the grid, threshold and every span before writing, so a refusal leaves all
// outputs unchanged. Elements are read before they are written: an output may
// be the same array as an input, but the two backward outputs must not overlap.
package main

// #include <stddef.h>
// #include <stdint.h>
import "C"

import (
	"github.com/anulum/sc-neurocore/accel/conversion"
	"unsafe"
)

// span admits one complete aligned float64 span in the signed address domain.
func span(pointer *C.double, count C.size_t) bool {
	if count == 0 {
		return true
	}
	maximum := uint64(^uint(0) >> 1)
	address, elements := uint64(uintptr(unsafe.Pointer(pointer))), uint64(count)
	return address != 0 && address%8 == 0 && elements <= maximum/8 && address <= maximum-elements*8
}

// view borrows an admitted span for the duration of one call.
func view(pointer *C.double, count C.size_t) []float64 {
	if count == 0 {
		return nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(pointer)), int(count))
}

// sc_qcfs_abi_version reports the exact QCFS array ABI version.
//
//export sc_qcfs_abi_version
func sc_qcfs_abi_version() C.uint32_t { return 1 }

// sc_qcfs_forward quantises count activations. Returns 0, or -1 for an invalid
// grid, threshold or span with every output unchanged.
//
//export sc_qcfs_forward
func sc_qcfs_forward(steps C.uint32_t, theta C.double, x *C.double, count C.size_t, output *C.double) C.int32_t {
	state := conversion.QCFSActivation{Steps: uint32(steps), Theta: float64(theta)}
	if !state.Valid() || !span(x, count) || !span(output, count) {
		return -1
	}
	values, results := view(x, count), view(output, count)
	for index := range values {
		result, _ := state.Forward(values[index])
		results[index] = result
	}
	return 0
}

// sc_qcfs_backward writes each element's input and threshold derivative.
// Returns 0, or -1 for an invalid grid, threshold or span with outputs unchanged.
//
//export sc_qcfs_backward
func sc_qcfs_backward(steps C.uint32_t, theta C.double, x, upstream *C.double, count C.size_t, inputGradient, thresholdGradient *C.double) C.int32_t {
	state := conversion.QCFSActivation{Steps: uint32(steps), Theta: float64(theta)}
	if !state.Valid() {
		return -1
	}
	for _, pointer := range []*C.double{x, upstream, inputGradient, thresholdGradient} {
		if !span(pointer, count) {
			return -1
		}
	}
	values, gradients := view(x, count), view(upstream, count)
	inputs, thresholds := view(inputGradient, count), view(thresholdGradient, count)
	for index := range values {
		input, threshold, _ := state.Backward(values[index], gradients[index])
		inputs[index], thresholds[index] = input, threshold
	}
	return 0
}

func main() {}
