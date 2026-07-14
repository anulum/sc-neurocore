// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for configurable threshold-linear rate

// Package main exports the maintained threshold-linear service as a C-shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export threshold_linear_rate_simulate_c
func threshold_linear_rate_simulate_c(
	r C.double,
	theta C.double,
	gain C.double,
	nSteps C.int64_t,
	current C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	maxInt := int(^uint(0) >> 1)
	if C.int64_t(n) != nSteps || n > maxInt-1 {
		return -1
	}
	initial := services.ThresholdLinearRateNeuronState{
		R: float64(r), Theta: float64(theta), Gain: float64(gain),
	}
	trace, final, err := services.SimulateThresholdLinearRateTrace(initial, n, float64(current))
	if err != nil {
		return -1
	}
	staged := make([]float64, n+1)
	copy(staged, trace)
	staged[n] = final.R
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+1)
	copy(destination, staged)
	return 0
}

func main() {}
