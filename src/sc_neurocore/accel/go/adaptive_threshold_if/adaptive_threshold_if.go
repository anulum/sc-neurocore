// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared mirror for composite reduced adaptive-threshold IF

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

func validConfiguration(values [9]float64) bool {
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return values[5] >= 0.0 &&
		values[6] > 0.0 &&
		values[7] > 0.0 &&
		values[8] > 0.0 &&
		values[4] > values[2] &&
		values[4] > values[3]
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

func exactRelaxation(state, steadyState, tau, dt float64) (float64, bool) {
	candidate := steadyState + (state-steadyState)*math.Exp(-dt/tau)
	return candidate, finite(candidate)
}

//export adaptive_threshold_if_simulate_c
func adaptive_threshold_if_simulate_c(
	n C.int32_t,
	vInit, thetaInit, vRest, vReset, thetaRest, deltaTheta, tauM, tauTheta, dt C.double,
	currentPtr, vOutPtr, thetaOutPtr, spikesOutPtr unsafe.Pointer,
	vFinal, thetaFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || vFinal == nil || thetaFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 &&
		(currentPtr == nil || vOutPtr == nil || thetaOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	vFinalRegion, vFinalOK := makeRegion(unsafe.Pointer(vFinal), 1)
	thetaFinalRegion, thetaFinalOK := makeRegion(unsafe.Pointer(thetaFinal), 1)
	countRegion, countOK := makeRegion(unsafe.Pointer(spikeCount), 1)
	if !vFinalOK || !thetaFinalOK || !countOK {
		return 1
	}
	regions := []memoryRegion{vFinalRegion, thetaFinalRegion, countRegion}
	if steps > 0 {
		currentRegion, currentOK := makeRegion(currentPtr, steps)
		vOutRegion, vOutOK := makeRegion(vOutPtr, steps)
		thetaOutRegion, thetaOutOK := makeRegion(thetaOutPtr, steps)
		spikesRegion, spikesOK := makeRegion(spikesOutPtr, steps)
		if !currentOK || !vOutOK || !thetaOutOK || !spikesOK {
			return 1
		}
		regions = append(
			regions,
			currentRegion,
			vOutRegion,
			thetaOutRegion,
			spikesRegion,
		)
	}
	if !regionsAreDistinct(regions) {
		return 1
	}

	configuration := [9]float64{
		float64(vInit),
		float64(thetaInit),
		float64(vRest),
		float64(vReset),
		float64(thetaRest),
		float64(deltaTheta),
		float64(tauM),
		float64(tauTheta),
		float64(dt),
	}
	if !validConfiguration(configuration) {
		return 2
	}
	if steps == 0 {
		*vFinal, *thetaFinal, *spikeCount = vInit, thetaInit, 0.0
		return 0
	}
	current := unsafe.Slice((*C.double)(currentPtr), steps)
	for _, value := range current {
		if !finite(float64(value)) {
			return 3
		}
	}

	vTrace := make([]float64, steps)
	thetaTrace := make([]float64, steps)
	spikeTrace := make([]float64, steps)
	v, theta := configuration[0], configuration[1]
	vRestF, vResetF := configuration[2], configuration[3]
	thetaRestF, deltaThetaF := configuration[4], configuration[5]
	tauMF, tauThetaF, dtF := configuration[6], configuration[7], configuration[8]
	count := 0
	for index := 0; index < steps; index++ {
		nextV, ok := exactRelaxation(v, vRestF+float64(current[index]), tauMF, dtF)
		if !ok {
			return 4
		}
		nextTheta, ok := exactRelaxation(theta, thetaRestF, tauThetaF, dtF)
		if !ok {
			return 4
		}
		spike := 0.0
		if nextV >= nextTheta {
			spikeTheta := nextTheta + deltaThetaF
			if !finite(spikeTheta) {
				return 4
			}
			v, theta = vResetF, spikeTheta
			spike = 1.0
			count++
		} else {
			v, theta = nextV, nextTheta
		}
		vTrace[index], thetaTrace[index], spikeTrace[index] = v, theta, spike
	}

	vOut := unsafe.Slice((*C.double)(vOutPtr), steps)
	thetaOut := unsafe.Slice((*C.double)(thetaOutPtr), steps)
	spikesOut := unsafe.Slice((*C.double)(spikesOutPtr), steps)
	for index := 0; index < steps; index++ {
		vOut[index] = C.double(vTrace[index])
		thetaOut[index] = C.double(thetaTrace[index])
		spikesOut[index] = C.double(spikeTrace[index])
	}
	*vFinal = C.double(v)
	*thetaFinal = C.double(theta)
	*spikeCount = C.double(count)
	return 0
}

func main() {}
