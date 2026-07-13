// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the published DPI circuit

// Package main exposes services.DPINeuronState as a C-shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export dpi_neuron_simulate_c
func dpi_neuron_simulate_c(
	iMem C.double,
	iAHP C.double,
	refractoryTime C.double,
	iThreshold C.double,
	iReset C.double,
	iRest C.double,
	iTau C.double,
	iG C.double,
	iTauAHP C.double,
	iGA C.double,
	iSpike C.double,
	i0 C.double,
	kappa C.double,
	alpha C.double,
	tau C.double,
	tauAHP C.double,
	refractoryPeriod C.double,
	dt C.double,
	nSteps C.int64_t,
	current C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-3 {
		return -1
	}
	state := services.DPINeuronState{
		IMem:             float64(iMem),
		IAHP:             float64(iAHP),
		RefractoryTime:   float64(refractoryTime),
		IThreshold:       float64(iThreshold),
		IReset:           float64(iReset),
		IRest:            float64(iRest),
		ITau:             float64(iTau),
		IG:               float64(iG),
		ITauAHP:          float64(iTauAHP),
		IGA:              float64(iGA),
		ISpike:           float64(iSpike),
		I0:               float64(i0),
		Kappa:            float64(kappa),
		Alpha:            float64(alpha),
		Tau:              float64(tau),
		TauAHP:           float64(tauAHP),
		RefractoryPeriod: float64(refractoryPeriod),
		Dt:               float64(dt),
	}
	input := float64(current)
	if !state.Valid() || math.IsNaN(input) || math.IsInf(input, 0) {
		return -1
	}
	trace, spikes, final, err := services.SimulateDPITrace(state, n, input)
	if err != nil {
		return -1
	}
	staged := make([]float64, n+3)
	copy(staged, trace)
	staged[n] = final.IMem
	staged[n+1] = final.IAHP
	staged[n+2] = final.RefractoryTime
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+3)
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
