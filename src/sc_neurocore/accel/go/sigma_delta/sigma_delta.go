// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Package main exports the complete sampled APSDM trace contract.
package main

/* #include <stdint.h> */
import "C"
import (
	"github.com/anulum/sc-neurocore/accel/services"
	"unsafe"
)

// sigma_delta_simulate_c returns zero only after writing a valid batch.
//
//export sigma_delta_simulate_c
func sigma_delta_simulate_c(stepsC C.int, sigmaC, reconstructionC, deltaC, tauC, dtC C.double, currentsPtr, sigmaPtr, reconstructionPtr, eventsPtr, sigmaFinalPtr, reconstructionFinalPtr unsafe.Pointer) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || sigmaPtr == nil || reconstructionPtr == nil || eventsPtr == nil || sigmaFinalPtr == nil || reconstructionFinalPtr == nil {
		return 1
	}
	s := services.SigmaDeltaNeuronState{Sigma: float64(sigmaC), Reconstruction: float64(reconstructionC), Delta: float64(deltaC), TauReconstruction: float64(tauC), Dt: float64(dtC)}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	sigmas := unsafe.Slice((*C.double)(sigmaPtr), steps)
	reconstructions := unsafe.Slice((*C.double)(reconstructionPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for i := 0; i < steps; i++ {
		event, err := s.Step(float64(currents[i]))
		if err != nil {
			return 2
		}
		sigmas[i] = C.double(s.Sigma)
		reconstructions[i] = C.double(s.Reconstruction)
		events[i] = C.int64_t(event)
	}
	*(*C.double)(sigmaFinalPtr) = C.double(s.Sigma)
	*(*C.double)(reconstructionFinalPtr) = C.double(s.Reconstruction)
	return 0
}
func main() {}
