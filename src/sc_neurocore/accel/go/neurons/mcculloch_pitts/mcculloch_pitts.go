// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the source-faithful McCulloch-Pitts rule

// Package main exposes services.EvaluateMcCullochPittsBatch as a C-shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export mcculloch_pitts_evaluate_c
func mcculloch_pitts_evaluate_c(
	theta C.int64_t,
	excitatoryCounts *C.int64_t,
	inhibitoryFlags *C.uint8_t,
	nRows C.int64_t,
	output *C.uint8_t,
) C.int64_t {
	if nRows < 0 {
		return -1
	}
	n := int(nRows)
	if C.int64_t(n) != nRows {
		return -1
	}
	if n > 0 && (excitatoryCounts == nil || inhibitoryFlags == nil || output == nil) {
		return -1
	}

	counts := make([]int64, n)
	flags := make([]uint8, n)
	if n > 0 {
		foreignCounts := unsafe.Slice((*int64)(unsafe.Pointer(excitatoryCounts)), n)
		foreignFlags := unsafe.Slice((*uint8)(unsafe.Pointer(inhibitoryFlags)), n)
		copy(counts, foreignCounts)
		copy(flags, foreignFlags)
	}
	events, eventCount, err := services.EvaluateMcCullochPittsBatch(
		int64(theta),
		counts,
		flags,
	)
	if err != nil {
		return -1
	}
	if n > 0 {
		destination := unsafe.Slice((*uint8)(unsafe.Pointer(output)), n)
		copy(destination, events)
	}
	return C.int64_t(eventCount)
}

func main() {}
