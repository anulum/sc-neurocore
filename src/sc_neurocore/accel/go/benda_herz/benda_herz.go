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
typedef struct { double a; double phase; int32_t event; int32_t status; } benda_herz_out;
*/
import "C"
import "github.com/anulum/sc-neurocore/accel/services"

//export benda_herz_step
func benda_herz_step(a, phase, onsetGain, rheobase, adaptationSlope, tauA, dt, current C.double) C.benda_herz_out {
	s := services.BendaHerzNeuronState{A: float64(a), Phase: float64(phase), OnsetGain: float64(onsetGain), Rheobase: float64(rheobase), AdaptationSlope: float64(adaptationSlope), TauA: float64(tauA), Dt: float64(dt)}
	event := s.Step(float64(current)); status := int32(0); if event < 0 { status = -1 }
	return C.benda_herz_out{a: C.double(s.A), phase: C.double(s.Phase), event: C.int32_t(event), status: C.int32_t(status)}
}
func main() {}
