// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package main

/* #include <stdint.h> */
import "C"
import (
	"github.com/anulum/sc-neurocore/accel/services"
	"unsafe"
)

// sc_sigma_delta_accumulator_simulate_c exports the retained project trace.
//
//export sc_sigma_delta_accumulator_simulate_c
func sc_sigma_delta_accumulator_simulate_c(stepsC C.int, sigmaC, thresholdC C.double, currentsPtr, sigmaPtr, eventsPtr, sigmaFinalPtr unsafe.Pointer) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || sigmaPtr == nil || eventsPtr == nil || sigmaFinalPtr == nil {
		return 1
	}
	s := services.SCSigmaDeltaAccumulatorState{Sigma: float64(sigmaC), VThreshold: float64(thresholdC)}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	sigmas := unsafe.Slice((*C.double)(sigmaPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for i := 0; i < steps; i++ {
		event, err := s.Step(float64(currents[i]))
		if err != nil {
			return 2
		}
		sigmas[i] = C.double(s.Sigma)
		events[i] = C.int64_t(event)
	}
	*(*C.double)(sigmaFinalPtr) = C.double(s.Sigma)
	return 0
}
func main() {}
