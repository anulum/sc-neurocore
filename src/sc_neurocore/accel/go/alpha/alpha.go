// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared mirror for dual alpha-synapse LIF

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

func validConfiguration(values [11]float64) bool {
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return values[7] > 0.0 &&
		values[8] > 0.0 &&
		values[9] > 0.0 &&
		values[10] > 0.0 &&
		values[6] > values[5]
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

func filterCandidates(riseState, currentState, drive, tau, dt float64) (float64, float64, bool) {
	steadyState := tau * drive
	riseDelta := riseState - steadyState
	currentDelta := currentState - steadyState
	decay := math.Exp(-dt / tau)
	riseNext := steadyState + riseDelta*decay
	currentNext := steadyState + decay*(currentDelta+riseDelta*dt/tau)
	return riseNext, currentNext, finite(riseNext) && finite(currentNext)
}

func driveContribution(currentDelta, riseDelta, tauDrive, tauV, dt float64) (float64, bool) {
	rateV := 1.0 / tauV
	rateDrive := 1.0 / tauDrive
	decayV := math.Exp(-dt / tauV)
	decayDrive := math.Exp(-dt / tauDrive)
	var contribution float64
	if math.Abs(rateV-rateDrive) <= 1.0e-14 {
		contribution = rateV * decayV * (currentDelta*dt + riseDelta*dt*dt/(2.0*tauDrive))
	} else {
		rateDelta := rateV - rateDrive
		firstOrder := currentDelta * (decayDrive - decayV) / rateDelta
		secondOrder := riseDelta / tauDrive *
			(decayDrive*(rateDelta*dt-1.0) + decayV) /
			(rateDelta * rateDelta)
		contribution = rateV * (firstOrder + secondOrder)
	}
	return contribution, finite(contribution)
}

//export alpha_simulate_c
func alpha_simulate_c(
	n C.int32_t,
	vInit, aExcInit, iExcInit, aInhInit, iInhInit, vRest, vThreshold, tauV, tauExc, tauInh, dt C.double,
	excCurrentPtr, inhCurrentPtr, vOutPtr, aExcOutPtr, iExcOutPtr, aInhOutPtr, iInhOutPtr, spikesOutPtr unsafe.Pointer,
	vFinal, aExcFinal, iExcFinal, aInhFinal, iInhFinal, spikeCount *C.double,
) C.int32_t {
	if n < 0 || vFinal == nil || aExcFinal == nil || iExcFinal == nil ||
		aInhFinal == nil || iInhFinal == nil || spikeCount == nil {
		return 1
	}
	steps := int(n)
	if steps > 0 &&
		(excCurrentPtr == nil || inhCurrentPtr == nil || vOutPtr == nil || aExcOutPtr == nil ||
			iExcOutPtr == nil || aInhOutPtr == nil || iInhOutPtr == nil || spikesOutPtr == nil) {
		return 1
	}
	finalRegions := []memoryRegion{}
	for _, pointer := range []unsafe.Pointer{
		unsafe.Pointer(vFinal),
		unsafe.Pointer(aExcFinal),
		unsafe.Pointer(iExcFinal),
		unsafe.Pointer(aInhFinal),
		unsafe.Pointer(iInhFinal),
		unsafe.Pointer(spikeCount),
	} {
		region, ok := makeRegion(pointer, 1)
		if !ok {
			return 1
		}
		finalRegions = append(finalRegions, region)
	}
	regions := finalRegions
	if steps > 0 {
		for _, pair := range [][2]int{
			{int(uintptr(excCurrentPtr)), steps},
			{int(uintptr(inhCurrentPtr)), steps},
			{int(uintptr(vOutPtr)), steps},
			{int(uintptr(aExcOutPtr)), steps},
			{int(uintptr(iExcOutPtr)), steps},
			{int(uintptr(aInhOutPtr)), steps},
			{int(uintptr(iInhOutPtr)), steps},
			{int(uintptr(spikesOutPtr)), steps},
		} {
			region, ok := makeRegion(unsafe.Pointer(uintptr(pair[0])), pair[1])
			if !ok {
				return 1
			}
			regions = append(regions, region)
		}
	}
	if !regionsAreDistinct(regions) {
		return 1
	}

	configuration := [11]float64{
		float64(vInit),
		float64(aExcInit),
		float64(iExcInit),
		float64(aInhInit),
		float64(iInhInit),
		float64(vRest),
		float64(vThreshold),
		float64(tauV),
		float64(tauExc),
		float64(tauInh),
		float64(dt),
	}
	if !validConfiguration(configuration) {
		return 2
	}
	if steps == 0 {
		*vFinal, *aExcFinal, *iExcFinal, *aInhFinal, *iInhFinal, *spikeCount =
			vInit, aExcInit, iExcInit, aInhInit, iInhInit, 0.0
		return 0
	}
	excCurrent := unsafe.Slice((*C.double)(excCurrentPtr), steps)
	inhCurrent := unsafe.Slice((*C.double)(inhCurrentPtr), steps)
	for index := 0; index < steps; index++ {
		if !finite(float64(excCurrent[index])) || !finite(float64(inhCurrent[index])) {
			return 3
		}
	}

	vTrace := make([]float64, steps)
	aExcTrace := make([]float64, steps)
	iExcTrace := make([]float64, steps)
	aInhTrace := make([]float64, steps)
	iInhTrace := make([]float64, steps)
	spikeTrace := make([]float64, steps)
	v, aExc, iExc, aInh, iInh := configuration[0], configuration[1], configuration[2], configuration[3], configuration[4]
	vRestF, vThresholdF := configuration[5], configuration[6]
	tauVF, tauExcF, tauInhF, dtF := configuration[7], configuration[8], configuration[9], configuration[10]
	count := 0
	for index := 0; index < steps; index++ {
		aExcNext, iExcNext, ok := filterCandidates(aExc, iExc, float64(excCurrent[index]), tauExcF, dtF)
		if !ok {
			return 4
		}
		aInhNext, iInhNext, ok := filterCandidates(aInh, iInh, float64(inhCurrent[index]), tauInhF, dtF)
		if !ok {
			return 4
		}
		excSteady := tauExcF * float64(excCurrent[index])
		inhSteady := tauInhF * float64(inhCurrent[index])
		vSteady := vRestF + excSteady - inhSteady
		decayV := math.Exp(-dtF / tauVF)
		excContribution, ok := driveContribution(iExc-excSteady, aExc-excSteady, tauExcF, tauVF, dtF)
		if !ok {
			return 4
		}
		inhContribution, ok := driveContribution(iInh-inhSteady, aInh-inhSteady, tauInhF, tauVF, dtF)
		if !ok {
			return 4
		}
		vNext := vSteady + (v-vSteady)*decayV + excContribution - inhContribution
		if !finite(vNext) {
			return 4
		}
		spike := 0.0
		if vNext >= vThresholdF {
			v = vRestF
			spike = 1.0
			count++
		} else {
			v = vNext
		}
		aExc, iExc = aExcNext, iExcNext
		aInh, iInh = aInhNext, iInhNext
		vTrace[index], aExcTrace[index], iExcTrace[index] = v, aExc, iExc
		aInhTrace[index], iInhTrace[index], spikeTrace[index] = aInh, iInh, spike
	}

	vOut := unsafe.Slice((*C.double)(vOutPtr), steps)
	aExcOut := unsafe.Slice((*C.double)(aExcOutPtr), steps)
	iExcOut := unsafe.Slice((*C.double)(iExcOutPtr), steps)
	aInhOut := unsafe.Slice((*C.double)(aInhOutPtr), steps)
	iInhOut := unsafe.Slice((*C.double)(iInhOutPtr), steps)
	spikesOut := unsafe.Slice((*C.double)(spikesOutPtr), steps)
	for index := 0; index < steps; index++ {
		vOut[index] = C.double(vTrace[index])
		aExcOut[index] = C.double(aExcTrace[index])
		iExcOut[index] = C.double(iExcTrace[index])
		aInhOut[index] = C.double(aInhTrace[index])
		iInhOut[index] = C.double(iInhTrace[index])
		spikesOut[index] = C.double(spikeTrace[index])
	}
	*vFinal = C.double(v)
	*aExcFinal = C.double(aExc)
	*iExcFinal = C.double(iExc)
	*aInhFinal = C.double(aInh)
	*iInhFinal = C.double(iInh)
	*spikeCount = C.double(count)
	return 0
}

func main() {}
