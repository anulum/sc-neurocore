// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the retained SC adaptive-threshold map

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
func bounded(value, lower, upper float64) bool {
	return finite(value) && value >= lower && value <= upper
}
func sigmoid(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	exponential := math.Exp(value)
	return exponential / (1.0 + exponential)
}

//export sc_adaptive_threshold_map_simulate_c
func sc_adaptive_threshold_map_simulate_c(
	n C.int32_t,
	xInit, thetaInit, k, beta, gamma, thetaSpike, xThreshold C.double,
	currentPtr, xOutPtr, thetaOutPtr, spikesOutPtr unsafe.Pointer,
	xFinal, thetaFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || xFinal == nil || thetaFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 && (currentPtr == nil || xOutPtr == nil || thetaOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	values := [7]float64{float64(xInit), float64(thetaInit), float64(k), float64(beta), float64(gamma), float64(thetaSpike), float64(xThreshold)}
	if !bounded(values[0], -5.0, 5.0) || !bounded(values[1], -5.0, 5.0) || !bounded(values[2], 0.0, 5.0) || !bounded(values[3], 0.0, 1.0) || !bounded(values[4], 0.0, 2.0) || !bounded(values[5], 0.0, 2.0) || !bounded(values[6], 0.0, 2.0) {
		return 2
	}
	x, theta := values[0], values[1]
	if steps == 0 {
		*xFinal, *thetaFinal, *spikeCount = C.double(x), C.double(theta), 0.0
		return 0
	}
	current := unsafe.Slice((*C.double)(currentPtr), steps)
	for _, value := range current {
		if !finite(float64(value)) {
			return 3
		}
	}
	xTrace, thetaTrace, spikes := make([]float64, steps), make([]float64, steps), make([]float64, steps)
	count := 0
	for index, drive := range current {
		previousX := x
		nextX := -x + values[2]*sigmoid((x-theta)*4.0) + float64(drive)
		fired := 0.0
		if x >= values[5] {
			fired = 1.0
		}
		nextTheta := values[3]*theta + values[4]*fired
		if !finite(nextX) || !finite(nextTheta) {
			return 4
		}
		x, theta = math.Max(-5.0, math.Min(5.0, nextX)), math.Max(-5.0, math.Min(5.0, nextTheta))
		event := 0.0
		if x >= values[6] && previousX < values[6] {
			event = 1.0
			count++
		}
		xTrace[index], thetaTrace[index], spikes[index] = x, theta, event
	}
	copy(unsafe.Slice((*float64)(xOutPtr), steps), xTrace)
	copy(unsafe.Slice((*float64)(thetaOutPtr), steps), thetaTrace)
	copy(unsafe.Slice((*float64)(spikesOutPtr), steps), spikes)
	*xFinal, *thetaFinal, *spikeCount = C.double(x), C.double(theta), C.double(count)
	return 0
}

func main() {}
