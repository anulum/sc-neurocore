// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — N-MNIST C ABI

// Package main exposes the Go N-MNIST decoder to Python without Go-owned pointers.
package main

/*
#include <stdint.h>
#include <stddef.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services/loaders"
)

// nmnist_decode_c writes four doubles per event into a caller-owned destination.
// Returns zero on success or -1 for invalid lengths, pointers or capacity.
// Empty input accepts null pointers; no destination bytes change on refusal.
// Nonempty pointers must identify live buffers of the declared size; overlap is refused.
//
//export nmnist_decode_c
func nmnist_decode_c(raw *C.uint8_t, byteCount C.size_t, output *C.double, valueCount C.size_t) C.int {
	maximum := uint64(^uint(0) >> 1)
	if uint64(byteCount) > maximum || uint64(valueCount) > maximum/8 {
		return -1
	}
	n := int(byteCount)
	m := int(valueCount)
	if n%5 != 0 || m%4 != 0 || m/4 != n/5 {
		return -1
	}
	if n == 0 {
		return 0
	}
	if raw == nil || output == nil {
		return -1
	}
	inputStart := uintptr(unsafe.Pointer(raw))
	outputStart := uintptr(unsafe.Pointer(output))
	if outputStart%unsafe.Alignof(float64(0)) != 0 {
		return -1
	}
	inputEnd := inputStart + uintptr(n)
	outputEnd := outputStart + uintptr(m)*8
	if inputEnd < inputStart || outputEnd < outputStart ||
		(inputStart < outputEnd && outputStart < inputEnd) {
		return -1
	}
	input := unsafe.Slice((*byte)(unsafe.Pointer(raw)), n)
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), m)
	if err := loaders.DecodeNMNISTInto(input, destination); err != nil {
		return -1
	}
	return 0
}

func main() {}
