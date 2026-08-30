// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the Quadratic IF exact-flow recurrence

// Package main exposes services.QuadraticIFNeuronState as a C-shared library.
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

//export quadratic_if_simulate_c
func quadratic_if_simulate_c(
	v C.double,
	vReset C.double,
	vPeak C.double,
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
	state := services.QuadraticIFNeuronState{
		V:      float64(v),
		VReset: float64(vReset),
		VPeak:  float64(vPeak),
		Dt:     float64(dt),
	}
	input := float64(current)
	if !state.Valid() || math.IsNaN(input) || math.IsInf(input, 0) {
		return -1
	}
	trace, spikes, finalV, err := services.SimulateQuadraticIFTrace(state, n, input)
	if err != nil {
		return -1
	}
	staged := make([]float64, n+1)
	copy(staged, trace)
	staged[n] = finalV
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+1)
	copy(destination, staged)
	return C.int64_t(spikes)
}

//export quadratic_if_simulate_complete_c
func quadratic_if_simulate_complete_c(
	v C.double,
	vReset C.double,
	vPeak C.double,
	dt C.double,
	sourceProfile C.int64_t,
	nSteps C.int64_t,
	current C.double,
	voltage *C.double,
	events *C.uint8_t,
) C.int64_t {
	if nSteps < 0 || voltage == nil || (nSteps > 0 && events == nil) || (sourceProfile != 0 && sourceProfile != 1) {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-1 {
		return -1
	}
	state := services.QuadraticIFNeuronState{
		V: float64(v), VReset: float64(vReset), VPeak: float64(vPeak), Dt: float64(dt), SourceProfile: sourceProfile == 1,
	}
	trace, eventTrace, finalV, err := services.SimulateQuadraticIFComplete(state, n, float64(current))
	if err != nil {
		return -1
	}
	stagedVoltage := make([]float64, n+1)
	copy(stagedVoltage, trace)
	stagedVoltage[n] = finalV
	copy(unsafe.Slice((*float64)(unsafe.Pointer(voltage)), n+1), stagedVoltage)
	if n > 0 {
		copy(unsafe.Slice((*uint8)(unsafe.Pointer(events)), n), eventTrace)
	}
	count := 0
	for _, event := range eventTrace {
		count += int(event)
	}
	return C.int64_t(count)
}

func main() {}
