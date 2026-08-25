// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package main

/*
#include <stdint.h>
typedef struct { double a; int32_t event; int32_t status; } sc_sra_out;
*/
import "C"
import "github.com/anulum/sc-neurocore/accel/services"

//export sc_sra_step
func sc_sra_step(a, fMax, beta, iHalf, tauA, deltaA, dt, current, uniform C.double) C.sc_sra_out {
	s := services.SCStochasticRateAdaptationNeuronState{A: float64(a), FMax: float64(fMax), Beta: float64(beta), IHalf: float64(iHalf), TauA: float64(tauA), DeltaA: float64(deltaA), Dt: float64(dt), Rng: float64(uniform)}
	event := s.Step(float64(current))
	status := int32(0)
	if !s.Valid() {
		status = -1
	}
	return C.sc_sra_out{a: C.double(s.A), event: C.int32_t(event), status: C.int32_t(status)}
}
func main() {}
