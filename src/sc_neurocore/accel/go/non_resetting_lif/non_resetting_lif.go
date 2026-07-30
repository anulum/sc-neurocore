// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Package main exports the complete MAT(1) trace contract.
package main

/* #include <stdint.h> */
import "C"

import (
	"github.com/anulum/sc-neurocore/accel/services"
	"unsafe"
)

// non_resetting_lif_simulate_c returns zero only after writing a complete valid batch.
// Build: go build -buildmode=c-shared -o libnon_resetting_lif.so non_resetting_lif.go
//
//export non_resetting_lif_simulate_c
func non_resetting_lif_simulate_c(stepsC C.int, vC, thetaC, refractoryC, omegaC, tauMC, tauThetaC, alphaC, resistanceC, refractoryPeriodC, dtC C.double, currentsPtr, voltagesPtr, thetaPtr, refractoryOutPtr, eventsPtr, vFinalPtr, thetaFinalPtr, refractoryFinalPtr unsafe.Pointer) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || voltagesPtr == nil || thetaPtr == nil || refractoryOutPtr == nil || eventsPtr == nil || vFinalPtr == nil || thetaFinalPtr == nil || refractoryFinalPtr == nil {
		return 1
	}
	state := services.NonResettingLIFNeuronState{V: float64(vC), Theta: float64(thetaC), RefractoryRemaining: float64(refractoryC), Omega: float64(omegaC), TauM: float64(tauMC), TauTheta: float64(tauThetaC), Alpha: float64(alphaC), Resistance: float64(resistanceC), RefractoryPeriod: float64(refractoryPeriodC), Dt: float64(dtC)}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	thresholds := unsafe.Slice((*C.double)(thetaPtr), steps)
	refractory := unsafe.Slice((*C.double)(refractoryOutPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for index := 0; index < steps; index++ {
		event, err := state.Step(float64(currents[index]))
		if err != nil {
			return 2
		}
		voltages[index] = C.double(state.V)
		thresholds[index] = C.double(state.Theta)
		refractory[index] = C.double(state.RefractoryRemaining)
		events[index] = C.int64_t(event)
	}
	*(*C.double)(vFinalPtr) = C.double(state.V)
	*(*C.double)(thetaFinalPtr) = C.double(state.Theta)
	*(*C.double)(refractoryFinalPtr) = C.double(state.RefractoryRemaining)
	return 0
}
func main() {}
