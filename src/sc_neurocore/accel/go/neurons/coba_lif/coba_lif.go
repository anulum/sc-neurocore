// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the Brette et al. COBA LIF cell

// Package main exposes services.COBALIFNeuronState as a C-shared library.
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

//export coba_lif_simulate_c
func coba_lif_simulate_c(
	v C.double,
	gE C.double,
	gI C.double,
	refractoryTime C.double,
	cM C.double,
	gL C.double,
	eL C.double,
	eE C.double,
	eI C.double,
	tauE C.double,
	tauI C.double,
	vThreshold C.double,
	vReset C.double,
	refractoryPeriod C.double,
	dt C.double,
	nSteps C.int64_t,
	current C.double,
	deltaGE C.double,
	deltaGI C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-4 {
		return -1
	}
	state := services.COBALIFNeuronState{
		V:                float64(v),
		GE:               float64(gE),
		GI:               float64(gI),
		RefractoryTime:   float64(refractoryTime),
		CM:               float64(cM),
		GL:               float64(gL),
		EL:               float64(eL),
		EE:               float64(eE),
		EI:               float64(eI),
		TauE:             float64(tauE),
		TauI:             float64(tauI),
		VThreshold:       float64(vThreshold),
		VReset:           float64(vReset),
		RefractoryPeriod: float64(refractoryPeriod),
		Dt:               float64(dt),
	}
	input := float64(current)
	dge := float64(deltaGE)
	dgi := float64(deltaGI)
	if !state.Valid() || math.IsNaN(input) || math.IsInf(input, 0) ||
		math.IsNaN(dge) || math.IsInf(dge, 0) || dge < 0.0 ||
		math.IsNaN(dgi) || math.IsInf(dgi, 0) || dgi < 0.0 {
		return -1
	}
	trace, spikes, final, err := services.SimulateCOBALIFTrace(state, n, input, dge, dgi)
	if err != nil {
		return -1
	}
	staged := make([]float64, n+4)
	copy(staged, trace)
	staged[n] = final.V
	staged[n+1] = final.GE
	staged[n+2] = final.GI
	staged[n+3] = final.RefractoryTime
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+4)
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
