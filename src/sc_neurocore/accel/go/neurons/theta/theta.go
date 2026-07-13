// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the Theta exact-flow recurrence

// Package main exposes services.ThetaNeuronState as a C-shared library.
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

//export theta_simulate_c
func theta_simulate_c(
	theta C.double,
	dt C.double,
	nSteps C.int64_t,
	current C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-1 {
		return -1
	}
	state := services.ThetaNeuronState{
		Theta: float64(theta),
		Dt:    float64(dt),
	}
	input := float64(current)
	if !state.Valid() || math.IsNaN(input) || math.IsInf(input, 0) {
		return -1
	}
	trace, spikes, finalTheta, err := services.SimulateThetaTrace(state, n, input)
	if err != nil {
		return -1
	}
	staged := make([]float64, n+1)
	copy(staged, trace)
	staged[n] = finalTheta
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+1)
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
