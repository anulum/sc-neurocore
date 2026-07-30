// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for SC resetting-MAT batches

// Package main exports the complete SC candidate-first RK4/reset contract.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

// sc_resetting_mat_simulate_c returns zero after one complete valid batch.
// Build: go build -buildmode=c-shared -o libsc_resetting_mat.so sc_resetting_mat.go
//
//export sc_resetting_mat_simulate_c
func sc_resetting_mat_simulate_c(
	stepsC C.int,
	vC, theta1C, theta2C, vRestC, vResetC, vThresholdBaseC C.double,
	tauMC, tau1C, tau2C, h1C, h2C, resistanceC, dtC C.double,
	currentsPtr, voltagesPtr, theta1Ptr, theta2Ptr, eventsPtr unsafe.Pointer,
	vFinalPtr, theta1FinalPtr, theta2FinalPtr unsafe.Pointer,
) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || voltagesPtr == nil || theta1Ptr == nil || theta2Ptr == nil || eventsPtr == nil || vFinalPtr == nil || theta1FinalPtr == nil || theta2FinalPtr == nil {
		return 1
	}
	state := services.SCResettingMATNeuronState{
		V: float64(vC), Theta1: float64(theta1C), Theta2: float64(theta2C),
		VRest: float64(vRestC), VReset: float64(vResetC), VThresholdBase: float64(vThresholdBaseC),
		TauM: float64(tauMC), Tau1: float64(tau1C), Tau2: float64(tau2C),
		H1: float64(h1C), H2: float64(h2C), Resistance: float64(resistanceC), Dt: float64(dtC),
	}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	theta1Trace := unsafe.Slice((*C.double)(theta1Ptr), steps)
	theta2Trace := unsafe.Slice((*C.double)(theta2Ptr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for index := 0; index < steps; index++ {
		event := state.Step(float64(currents[index]))
		if event < 0 {
			return 2
		}
		voltages[index] = C.double(state.V)
		theta1Trace[index] = C.double(state.Theta1)
		theta2Trace[index] = C.double(state.Theta2)
		events[index] = C.int64_t(event)
	}
	*(*C.double)(vFinalPtr) = C.double(state.V)
	*(*C.double)(theta1FinalPtr) = C.double(state.Theta1)
	*(*C.double)(theta2FinalPtr) = C.double(state.Theta2)
	return 0
}

func main() {}
