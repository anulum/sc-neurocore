// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for source-faithful Aihara dynamics

// Package main exports the checked Aihara Float64 batch through a C shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func logistic(value, epsilon float64) float64 {
	argument := value / epsilon
	if argument >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-argument))
	}
	exponential := math.Exp(argument)
	return exponential / (1.0 + exponential)
}

// distinctBuffers checks active byte ranges without dereferencing caller memory.
// Final outputs always have one element; empty trace/input ranges are inactive.
func distinctBuffers(steps int, pointers [7]unsafe.Pointer) bool {
	for left := 0; left < len(pointers); left++ {
		leftSize := uintptr(8)
		if left < 4 {
			leftSize *= uintptr(steps)
		}
		if leftSize == 0 {
			continue
		}
		for right := left + 1; right < len(pointers); right++ {
			rightSize := uintptr(8)
			if right < 4 {
				rightSize *= uintptr(steps)
			}
			a, b := uintptr(pointers[left]), uintptr(pointers[right])
			if (a <= b && b-a < leftSize) || (b < a && a-b < rightSize) {
				return false
			}
		}
	}
	return true
}

// aihara_map_simulate_c writes a complete source-map receipt on success (0).
// The caller owns seven pairwise-disjoint Float64 ranges: four n-element
// input/trace ranges and three scalar outputs. Empty traces may be nil.
// Status 1 rejects invalid/overlapping buffers, 2 configuration, 3 input,
// and 4 numerical overflow. Any rejection leaves all caller storage unchanged.
//
//export aihara_map_simulate_c
func aihara_map_simulate_c(
	n C.int32_t,
	yInit, k, alpha, bias, epsilon C.double,
	currentPtr, yOutPtr, xOutPtr, spikesOutPtr unsafe.Pointer,
	yFinal, xFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || yFinal == nil || xFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 && (currentPtr == nil || yOutPtr == nil || xOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	if !distinctBuffers(steps, [7]unsafe.Pointer{
		currentPtr, yOutPtr, xOutPtr, spikesOutPtr,
		unsafe.Pointer(yFinal), unsafe.Pointer(xFinal), unsafe.Pointer(spikeCount),
	}) {
		return 1
	}
	values := [5]float64{float64(yInit), float64(k), float64(alpha), float64(bias), float64(epsilon)}
	for _, value := range values {
		if !finite(value) {
			return 2
		}
	}
	if values[1] < 0.0 || values[1] >= 1.0 || values[2] <= 0.0 || values[4] <= 0.0 {
		return 2
	}
	y := values[0]
	if steps == 0 {
		*yFinal = C.double(y)
		*xFinal = C.double(logistic(y, values[4]))
		*spikeCount = 0.0
		return 0
	}
	current := unsafe.Slice((*C.double)(currentPtr), steps)
	for _, value := range current {
		if !finite(float64(value)) {
			return 3
		}
	}
	yTrace := make([]float64, steps)
	xTrace := make([]float64, steps)
	spikes := make([]float64, steps)
	count := 0
	for index, drive := range current {
		nextY := values[1]*y - values[2]*logistic(y, values[4]) + values[3] + float64(drive)
		if !finite(nextY) {
			return 4
		}
		y = nextY
		x := logistic(y, values[4])
		event := 0.0
		if x >= 0.5 {
			event = 1.0
			count++
		}
		yTrace[index], xTrace[index], spikes[index] = y, x, event
	}
	copy(unsafe.Slice((*float64)(yOutPtr), steps), yTrace)
	copy(unsafe.Slice((*float64)(xOutPtr), steps), xTrace)
	copy(unsafe.Slice((*float64)(spikesOutPtr), steps), spikes)
	*yFinal = C.double(y)
	*xFinal = C.double(logistic(y, values[4]))
	*spikeCount = C.double(count)
	return 0
}

func main() {}
