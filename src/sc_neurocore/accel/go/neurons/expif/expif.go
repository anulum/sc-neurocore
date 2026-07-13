// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the maintained ExpIF recurrence

// Package main exposes services.ExpIFNeuronState as a C-shared library.
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

//export expif_simulate_c
func expif_simulate_c(
	v C.double,
	vRest C.double,
	vReset C.double,
	vThreshold C.double,
	vRh C.double,
	deltaT C.double,
	tau C.double,
	dt C.double,
	refractoryPeriod C.double,
	refractoryRemaining C.double,
	nSteps C.int64_t,
	current C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-2 {
		return -1
	}
	state := &services.ExpIFNeuronState{
		V:                   float64(v),
		VRest:               float64(vRest),
		VReset:              float64(vReset),
		VThreshold:          float64(vThreshold),
		VRh:                 float64(vRh),
		DeltaT:              float64(deltaT),
		Tau:                 float64(tau),
		Dt:                  float64(dt),
		RefractoryPeriod:    float64(refractoryPeriod),
		RefractoryRemaining: float64(refractoryRemaining),
	}
	if !state.Valid() || math.IsNaN(float64(current)) || math.IsInf(float64(current), 0) {
		return -1
	}
	trace := make([]float64, n+2)
	spikes := 0
	for index := 0; index < n; index++ {
		spike, err := state.Step(float64(current))
		if err != nil {
			return -1
		}
		spikes += spike
		trace[index] = state.V
	}
	trace[n] = state.V
	trace[n+1] = state.RefractoryRemaining
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+2)
	copy(destination, trace)
	return C.int64_t(spikes)
}

func main() {}
