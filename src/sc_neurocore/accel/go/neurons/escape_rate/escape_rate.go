// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the seeded EscapeRate cell

// Package main exposes the complete services.EscapeRateNeuronState contract.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export escape_rate_simulate_c
func escape_rate_simulate_c(
	v C.double,
	vRest C.double,
	vReset C.double,
	vThreshold C.double,
	tauM C.double,
	rho0 C.double,
	deltaU C.double,
	resistance C.double,
	dt C.double,
	rngState C.uint16_t,
	nSteps C.int64_t,
	current C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	maxInt := int(^uint(0) >> 1)
	if C.int64_t(n) != nSteps || n > (maxInt-2)/2 {
		return -1
	}
	rng := uint16(rngState)
	state := services.EscapeRateNeuronState{
		V:           float64(v),
		VRest:       float64(vRest),
		VReset:      float64(vReset),
		VThreshold:  float64(vThreshold),
		TauM:        float64(tauM),
		Rho0:        float64(rho0),
		DeltaU:      float64(deltaU),
		Resistance:  float64(resistance),
		Dt:          float64(dt),
		RNGState:    rng,
		InitialSeed: rng,
	}
	trace, events, final, err := services.SimulateEscapeRateTrace(
		state,
		n,
		float64(current),
	)
	if err != nil {
		return -1
	}
	staged := make([]float64, 2*n+2)
	copy(staged, trace)
	spikes := int64(0)
	for index, event := range events {
		staged[n+index] = float64(event)
		spikes += int64(event)
	}
	staged[2*n] = final.V
	staged[2*n+1] = float64(final.RNGState)
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), len(staged))
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
