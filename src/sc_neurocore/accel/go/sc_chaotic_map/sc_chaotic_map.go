// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the SC two-state chaotic map

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

func logistic(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	exponential := math.Exp(value)
	return exponential / (1.0 + exponential)
}

//export sc_chaotic_map_simulate_c
func sc_chaotic_map_simulate_c(
	n C.int32_t,
	xInit, yInit, kF, kS, alpha, delta, threshold C.double,
	currentPtr, xOutPtr, yOutPtr, spikesOutPtr unsafe.Pointer,
	xFinal, yFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || xFinal == nil || yFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 && (currentPtr == nil || xOutPtr == nil || yOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	values := [7]float64{float64(xInit), float64(yInit), float64(kF), float64(kS), float64(alpha), float64(delta), float64(threshold)}
	for _, value := range values {
		if !finite(value) {
			return 2
		}
	}
	if values[2] < 0.0 || values[5] < 0.0 {
		return 2
	}
	x, y := values[0], values[1]
	if steps == 0 {
		*xFinal, *yFinal, *spikeCount = C.double(x), C.double(y), 0.0
		return 0
	}
	current := unsafe.Slice((*C.double)(currentPtr), steps)
	for _, value := range current {
		if !finite(float64(value)) {
			return 3
		}
	}
	xTrace, yTrace := make([]float64, steps), make([]float64, steps)
	spikes := make([]float64, steps)
	count := 0
	for index, drive := range current {
		xPrevious := x
		xNext := values[2]*x*logistic(x+values[4]) - y + float64(drive)
		yNext := values[3]*y + values[5]*x
		if !finite(xNext) || !finite(yNext) {
			return 4
		}
		x = math.Min(10.0, math.Max(-10.0, xNext))
		y = math.Min(10.0, math.Max(-10.0, yNext))
		event := 0.0
		if xPrevious < values[6] && x >= values[6] {
			event = 1.0
			count++
		}
		xTrace[index], yTrace[index], spikes[index] = x, y, event
	}
	copy(unsafe.Slice((*float64)(xOutPtr), steps), xTrace)
	copy(unsafe.Slice((*float64)(yOutPtr), steps), yTrace)
	copy(unsafe.Slice((*float64)(spikesOutPtr), steps), spikes)
	*xFinal, *yFinal, *spikeCount = C.double(x), C.double(y), C.double(count)
	return 0
}

func main() {}
