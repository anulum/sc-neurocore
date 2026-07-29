// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for Brunel-Wang midpoint-RK2 batches

// Package main exports the complete configured cell contract.
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

// brunel_wang_simulate_c returns 0 only after writing a complete valid batch.
// Build: go build -buildmode=c-shared -o libbrunel_wang.so brunel_wang.go
//
//export brunel_wang_simulate_c
func brunel_wang_simulate_c(
	stepsC C.int,
	vC, refC, vRestC, vResetC, vThresholdC, tauMC, tauRefC C.double,
	gAmpaExtC, gAmpaRecC, gNmdaC, gGabaC C.double,
	vAmpaC, vNmdaC, vGabaC, cMC, mgConcC, dtC C.double,
	extPtr, ampaPtr, nmdaPtr, gabaPtr unsafe.Pointer,
	voltagesPtr, refractoryPtr, eventsPtr unsafe.Pointer,
	vFinalPtr, refFinalPtr unsafe.Pointer,
) C.int {
	steps := int(stepsC)
	if steps < 0 {
		return 1
	}
	values := []float64{float64(vC), float64(refC), float64(vRestC), float64(vResetC),
		float64(vThresholdC), float64(tauMC), float64(tauRefC), float64(gAmpaExtC),
		float64(gAmpaRecC), float64(gNmdaC), float64(gGabaC), float64(vAmpaC),
		float64(vNmdaC), float64(vGabaC), float64(cMC), float64(mgConcC), float64(dtC)}
	for _, value := range values {
		if !finite(value) {
			return 2
		}
	}
	if values[1] < 0 || values[5] <= 0 || values[6] <= 0 || values[7] < 0 ||
		values[8] < 0 || values[9] < 0 || values[10] < 0 || values[14] <= 0 ||
		values[15] < 0 || values[16] <= 0 {
		return 2
	}
	ext := unsafe.Slice((*C.double)(extPtr), steps)
	ampa := unsafe.Slice((*C.double)(ampaPtr), steps)
	nmda := unsafe.Slice((*C.double)(nmdaPtr), steps)
	gaba := unsafe.Slice((*C.double)(gabaPtr), steps)
	for _, gates := range [][]C.double{ext, ampa, nmda, gaba} {
		for _, gate := range gates {
			if !finite(float64(gate)) || gate < 0 {
				return 3
			}
		}
	}
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	refractory := unsafe.Slice((*C.double)(refractoryPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	v, ref := values[0], values[1]
	vRest, vReset, vThreshold, tauM, tauRef := values[2], values[3], values[4], values[5], values[6]
	gAmpaExt, gAmpaRec, gNmda, gGaba := values[7], values[8], values[9], values[10]
	vAmpa, vNmda, vGaba, cM, mgConc, dt := values[11], values[12], values[13], values[14], values[15], values[16]
	derivative := func(voltage, extGate, ampaGate, nmdaGate, gabaGate float64) float64 {
		block := 1 / (1 + mgConc/3.57*math.Exp(-0.062*voltage))
		iAmpa := -gAmpaExt*(voltage-vAmpa)*extGate - gAmpaRec*(voltage-vAmpa)*ampaGate
		iNmda := -gNmda * block * (voltage - vNmda) * nmdaGate
		iGaba := -gGaba * (voltage - vGaba) * gabaGate
		return -(voltage-vRest)/tauM + (iAmpa+iNmda+iGaba)/cM
	}
	for index := 0; index < steps; index++ {
		event := int64(0)
		if ref > 0 {
			v, ref = vReset, math.Max(0, ref-dt)
		} else {
			k1 := derivative(v, float64(ext[index]), float64(ampa[index]), float64(nmda[index]), float64(gaba[index]))
			midpoint := v + 0.5*dt*k1
			k2 := derivative(midpoint, float64(ext[index]), float64(ampa[index]), float64(nmda[index]), float64(gaba[index]))
			candidate := v + dt*k2
			if !finite(k1) || !finite(midpoint) || !finite(k2) || !finite(candidate) {
				return 4
			}
			v = candidate
			if candidate >= vThreshold {
				v, ref, event = vReset, tauRef, 1
			}
		}
		voltages[index], refractory[index], events[index] = C.double(v), C.double(ref), C.int64_t(event)
	}
	*(*C.double)(vFinalPtr), *(*C.double)(refFinalPtr) = C.double(v), C.double(ref)
	return 0
}

func main() {}
