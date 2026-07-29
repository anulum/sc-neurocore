// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for source-faithful Nagumo–Sato dynamics

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

func heaviside(value float64) float64 {
	if value >= 0.0 {
		return 1.0
	}
	return 0.0
}

//export nagumo_sato_map_simulate_c
func nagumo_sato_map_simulate_c(
	n C.int32_t,
	yInit, k, alpha, bias C.double,
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
	values := [4]float64{float64(yInit), float64(k), float64(alpha), float64(bias)}
	for _, value := range values {
		if !finite(value) {
			return 2
		}
	}
	if values[1] < 0.0 || values[1] >= 1.0 || values[2] <= 0.0 {
		return 2
	}
	y := values[0]
	if steps == 0 {
		*yFinal, *xFinal, *spikeCount = C.double(y), C.double(heaviside(y)), 0.0
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
		nextY := values[1]*y - values[2]*heaviside(y) + values[3] + float64(drive)
		if !finite(nextY) {
			return 4
		}
		y = nextY
		event := heaviside(y)
		yTrace[index], xTrace[index], spikes[index] = y, event, event
		count += int(event)
	}
	copy(unsafe.Slice((*float64)(yOutPtr), steps), yTrace)
	copy(unsafe.Slice((*float64)(xOutPtr), steps), xTrace)
	copy(unsafe.Slice((*float64)(spikesOutPtr), steps), spikes)
	*yFinal, *xFinal, *spikeCount = C.double(y), C.double(heaviside(y)), C.double(count)
	return 0
}

func main() {}
