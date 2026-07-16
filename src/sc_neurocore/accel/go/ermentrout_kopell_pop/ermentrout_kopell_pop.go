// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared batch mirror for the MPR mean field

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

func validConfiguration(values [7]float64) bool {
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return values[0] >= 0.0 && values[2] > 0.0 && values[3] >= 0.0 && values[6] > 0.0
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

//export ermentrout_kopell_pop_simulate_c
func ermentrout_kopell_pop_simulate_c(
	n C.int32_t,
	rInit, vInit, tau, delta, etaBar, coupling, dt C.double,
	extInputPtr, rOutPtr, vOutPtr unsafe.Pointer,
	rFinal, vFinal *C.double,
) C.int32_t {
	if n < 0 || rFinal == nil || vFinal == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 && (extInputPtr == nil || rOutPtr == nil || vOutPtr == nil) {
		return 1
	}
	rFinalRegion, rFinalOK := makeRegion(unsafe.Pointer(rFinal), 1)
	vFinalRegion, vFinalOK := makeRegion(unsafe.Pointer(vFinal), 1)
	if !rFinalOK || !vFinalOK {
		return 1
	}
	regions := []memoryRegion{rFinalRegion, vFinalRegion}
	if steps > 0 {
		extInputRegion, extInputOK := makeRegion(extInputPtr, steps)
		rOutRegion, rOutOK := makeRegion(rOutPtr, steps)
		vOutRegion, vOutOK := makeRegion(vOutPtr, steps)
		if !extInputOK || !rOutOK || !vOutOK {
			return 1
		}
		regions = append(regions, extInputRegion, rOutRegion, vOutRegion)
	}
	if !regionsAreDistinct(regions) {
		return 1
	}
	configuration := [7]float64{
		float64(rInit), float64(vInit), float64(tau), float64(delta),
		float64(etaBar), float64(coupling), float64(dt),
	}
	if !validConfiguration(configuration) {
		return 2
	}
	if steps == 0 {
		*rFinal, *vFinal = rInit, vInit
		return 0
	}
	drive := unsafe.Slice((*C.double)(extInputPtr), steps)
	for _, value := range drive {
		if !finite(float64(value)) {
			return 3
		}
	}

	rTrace := make([]float64, steps)
	vTrace := make([]float64, steps)
	r, v := configuration[0], configuration[1]
	tauF, deltaF := configuration[2], configuration[3]
	etaF, couplingF, dtF := configuration[4], configuration[5], configuration[6]
	for index := 0; index < steps; index++ {
		scaledRate := math.Pi * tauF * r
		dr := deltaF/(math.Pi*tauF*tauF) + 2.0*r*v/tauF
		dv := (v*v + etaF + float64(drive[index]) + couplingF*tauF*r - scaledRate*scaledRate) / tauF
		nextR := r + dtF*dr
		nextV := v + dtF*dv
		if !finite(nextR) || !finite(nextV) || nextR < 0.0 {
			return 4
		}
		r, v = nextR, nextV
		rTrace[index], vTrace[index] = r, v
	}

	rOut := unsafe.Slice((*C.double)(rOutPtr), steps)
	vOut := unsafe.Slice((*C.double)(vOutPtr), steps)
	for index := 0; index < steps; index++ {
		rOut[index], vOut[index] = C.double(rTrace[index]), C.double(vTrace[index])
	}
	*rFinal, *vFinal = C.double(r), C.double(v)
	return 0
}

func main() {}
