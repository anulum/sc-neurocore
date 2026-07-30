// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for source MAT* batches

// Package main exports the complete non-resetting MAT* trace contract.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

// mat_simulate_c returns zero only after writing one complete valid batch.
// Build: go build -buildmode=c-shared -o libmat.so mat.go
//
//export mat_simulate_c
func mat_simulate_c(
	stepsC C.int,
	vC, theta1C, theta2C, refractoryC, omegaC C.double,
	tauMC, tau1C, tau2C, alpha1C, alpha2C, resistanceC, refractoryPeriodC, dtC C.double,
	currentsPtr, voltagesPtr, theta1Ptr, theta2Ptr, refractoryOutPtr, eventsPtr unsafe.Pointer,
	vFinalPtr, theta1FinalPtr, theta2FinalPtr, refractoryFinalPtr unsafe.Pointer,
) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || voltagesPtr == nil || theta1Ptr == nil || theta2Ptr == nil || refractoryOutPtr == nil || eventsPtr == nil || vFinalPtr == nil || theta1FinalPtr == nil || theta2FinalPtr == nil || refractoryFinalPtr == nil {
		return 1
	}
	state := services.MATNeuronState{
		V: float64(vC), Theta1: float64(theta1C), Theta2: float64(theta2C),
		RefractoryRemaining: float64(refractoryC), Omega: float64(omegaC),
		TauM: float64(tauMC), Tau1: float64(tau1C), Tau2: float64(tau2C),
		Alpha1: float64(alpha1C), Alpha2: float64(alpha2C), Resistance: float64(resistanceC),
		RefractoryPeriod: float64(refractoryPeriodC), Dt: float64(dtC),
	}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	theta1Trace := unsafe.Slice((*C.double)(theta1Ptr), steps)
	theta2Trace := unsafe.Slice((*C.double)(theta2Ptr), steps)
	refractoryTrace := unsafe.Slice((*C.double)(refractoryOutPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for index := 0; index < steps; index++ {
		event := state.Step(float64(currents[index]))
		if event < 0 {
			return 2
		}
		voltages[index] = C.double(state.V)
		theta1Trace[index] = C.double(state.Theta1)
		theta2Trace[index] = C.double(state.Theta2)
		refractoryTrace[index] = C.double(state.RefractoryRemaining)
		events[index] = C.int64_t(event)
	}
	*(*C.double)(vFinalPtr) = C.double(state.V)
	*(*C.double)(theta1FinalPtr) = C.double(state.Theta1)
	*(*C.double)(theta2FinalPtr) = C.double(state.Theta2)
	*(*C.double)(refractoryFinalPtr) = C.double(state.RefractoryRemaining)
	return 0
}

func main() {}
