// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared mirror for Izhikevich resonate-and-fire

package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validConfiguration(values [6]float64) bool {
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return values[3] > 0.0 && values[4] > 0.0 && values[5] > 0.0
}

type memoryRegion struct {
	start uintptr
	size  uintptr
}

func makeRegion(pointer unsafe.Pointer, elements int) (memoryRegion, bool) {
	if pointer == nil || elements <= 0 {
		return memoryRegion{}, false
	}
	elementBytes := unsafe.Sizeof(C.double(0))
	count := uintptr(elements)
	if count > ^uintptr(0)/elementBytes {
		return memoryRegion{}, false
	}
	return memoryRegion{start: uintptr(pointer), size: count * elementBytes}, true
}

func regionsOverlap(a, b memoryRegion) bool {
	if a.start <= b.start {
		return b.start-a.start < a.size
	}
	return a.start-b.start < b.size
}

func regionsAreDistinct(regions []memoryRegion) bool {
	for left := 0; left < len(regions); left++ {
		for right := left + 1; right < len(regions); right++ {
			if regionsOverlap(regions[left], regions[right]) {
				return false
			}
		}
	}
	return true
}

func exactFlow(x, y, current, b, omega, dt float64) (float64, float64, bool) {
	denominator := b*b + omega*omega
	xSS := -b * current / denominator
	ySS := omega * current / denominator
	decay := math.Exp(b * dt)
	angle := omega * dt
	cosAngle := math.Cos(angle)
	sinAngle := math.Sin(angle)
	if !finite(denominator) || denominator <= 0.0 ||
		!finite(xSS) || !finite(ySS) || !finite(decay) ||
		!finite(angle) || !finite(cosAngle) || !finite(sinAngle) {
		return 0.0, 0.0, false
	}
	dx := x - xSS
	dy := y - ySS
	nextX := xSS + decay*(dx*cosAngle-dy*sinAngle)
	nextY := ySS + decay*(dx*sinAngle+dy*cosAngle)
	return nextX, nextY, finite(nextX) && finite(nextY)
}

//export resonate_and_fire_simulate_c
func resonate_and_fire_simulate_c(
	n C.int32_t,
	xInit, yInit, b, omega, threshold, dt C.double,
	currentPtr, xOutPtr, yOutPtr, spikesOutPtr unsafe.Pointer,
	xFinal, yFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || xFinal == nil || yFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 &&
		(currentPtr == nil || xOutPtr == nil || yOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	xFinalRegion, xFinalOK := makeRegion(unsafe.Pointer(xFinal), 1)
	yFinalRegion, yFinalOK := makeRegion(unsafe.Pointer(yFinal), 1)
	countRegion, countOK := makeRegion(unsafe.Pointer(spikeCount), 1)
	if !xFinalOK || !yFinalOK || !countOK {
		return 1
	}
	regions := []memoryRegion{xFinalRegion, yFinalRegion, countRegion}
	if steps > 0 {
		currentRegion, currentOK := makeRegion(currentPtr, steps)
		xOutRegion, xOutOK := makeRegion(xOutPtr, steps)
		yOutRegion, yOutOK := makeRegion(yOutPtr, steps)
		spikesRegion, spikesOK := makeRegion(spikesOutPtr, steps)
		if !currentOK || !xOutOK || !yOutOK || !spikesOK {
			return 1
		}
		regions = append(
			regions,
			currentRegion,
			xOutRegion,
			yOutRegion,
			spikesRegion,
		)
	}
	if !regionsAreDistinct(regions) {
		return 1
	}

	configuration := [6]float64{
		float64(xInit),
		float64(yInit),
		float64(b),
		float64(omega),
		float64(threshold),
		float64(dt),
	}
	if !validConfiguration(configuration) {
		return 2
	}
	if steps == 0 {
		*xFinal, *yFinal, *spikeCount = xInit, yInit, 0.0
		return 0
	}
	current := unsafe.Slice((*C.double)(currentPtr), steps)
	for _, value := range current {
		if !finite(float64(value)) {
			return 3
		}
	}

	xTrace := make([]float64, steps)
	yTrace := make([]float64, steps)
	spikeTrace := make([]float64, steps)
	x, y := configuration[0], configuration[1]
	bF, omegaF, thresholdF, dtF := configuration[2], configuration[3], configuration[4], configuration[5]
	count := 0
	for index := 0; index < steps; index++ {
		nextX, nextY, ok := exactFlow(x, y, float64(current[index]), bF, omegaF, dtF)
		if !ok {
			return 4
		}
		spike := 0.0
		if y < thresholdF && nextY >= thresholdF {
			x, y = 0.0, thresholdF
			spike = 1.0
			count++
		} else {
			x, y = nextX, nextY
		}
		xTrace[index], yTrace[index], spikeTrace[index] = x, y, spike
	}

	xOut := unsafe.Slice((*C.double)(xOutPtr), steps)
	yOut := unsafe.Slice((*C.double)(yOutPtr), steps)
	spikesOut := unsafe.Slice((*C.double)(spikesOutPtr), steps)
	for index := 0; index < steps; index++ {
		xOut[index] = C.double(xTrace[index])
		yOut[index] = C.double(yTrace[index])
		spikesOut[index] = C.double(spikeTrace[index])
	}
	*xFinal = C.double(x)
	*yFinal = C.double(y)
	*spikeCount = C.double(count)
	return 0
}

func main() {}
