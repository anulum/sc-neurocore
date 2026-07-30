// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Package main exports the retained SC adaptive-LIF trace contract.
package main

/* #include <stdint.h> */
import "C"
import (
	"github.com/anulum/sc-neurocore/accel/services"
	"unsafe"
)

// sc_non_resetting_adaptive_lif_simulate_c returns zero only after a complete valid batch.
//
//export sc_non_resetting_adaptive_lif_simulate_c
func sc_non_resetting_adaptive_lif_simulate_c(stepsC C.int, vC, thetaC, vRestC, thetaRestC, deltaThetaC, tauMC, tauThetaC, rMC, dtC C.double, currentsPtr, voltagesPtr, thetaPtr, eventsPtr, vFinalPtr, thetaFinalPtr unsafe.Pointer) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || voltagesPtr == nil || thetaPtr == nil || eventsPtr == nil || vFinalPtr == nil || thetaFinalPtr == nil {
		return 1
	}
	state := services.SCNonResettingAdaptiveLIFNeuronState{V: float64(vC), Theta: float64(thetaC), VRest: float64(vRestC), ThetaRest: float64(thetaRestC), DeltaTheta: float64(deltaThetaC), TauM: float64(tauMC), TauTheta: float64(tauThetaC), RM: float64(rMC), Dt: float64(dtC)}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	thresholds := unsafe.Slice((*C.double)(thetaPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for index := 0; index < steps; index++ {
		event, err := state.Step(float64(currents[index]))
		if err != nil {
			return 2
		}
		voltages[index] = C.double(state.V)
		thresholds[index] = C.double(state.Theta)
		events[index] = C.int64_t(event)
	}
	*(*C.double)(vFinalPtr) = C.double(state.V)
	*(*C.double)(thetaFinalPtr) = C.double(state.Theta)
	return 0
}
func main() {}
