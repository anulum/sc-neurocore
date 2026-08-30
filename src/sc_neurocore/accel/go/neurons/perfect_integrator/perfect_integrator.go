// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the Perfect Integrator recurrence

// Package main exposes services.PerfectIntegratorNeuronState as a C-shared library.
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

//export perfect_integrator_simulate_c
func perfect_integrator_simulate_c(
	v C.double,
	cM C.double,
	vThreshold C.double,
	vReset C.double,
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
	state := &services.PerfectIntegratorNeuronState{
		V:          float64(v),
		CM:         float64(cM),
		VThreshold: float64(vThreshold),
		VReset:     float64(vReset),
		Dt:         float64(dt),
	}
	if !state.Valid() || math.IsNaN(float64(current)) || math.IsInf(float64(current), 0) {
		return -1
	}
	trace := make([]float64, n+1)
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
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+1)
	copy(destination, trace)
	return C.int64_t(spikes)
}

//export perfect_integrator_simulate_complete_c
func perfect_integrator_simulate_complete_c(
	v C.double,
	cM C.double,
	vThreshold C.double,
	vReset C.double,
	dt C.double,
	sourceProfile C.int64_t,
	nSteps C.int64_t,
	current C.double,
	voltageOutput *C.double,
	eventOutput *C.uint8_t,
) C.int64_t {
	if nSteps < 0 || voltageOutput == nil || (nSteps > 0 && eventOutput == nil) ||
		(sourceProfile != 0 && sourceProfile != 1) {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-1 {
		return -1
	}
	state := services.PerfectIntegratorNeuronState{
		V:             float64(v),
		CM:            float64(cM),
		VThreshold:    float64(vThreshold),
		VReset:        float64(vReset),
		Dt:            float64(dt),
		SourceProfile: sourceProfile == 1,
	}
	trace, events, finalV, err := services.SimulatePerfectIntegratorComplete(
		state, n, float64(current),
	)
	if err != nil {
		return -1
	}
	voltage := make([]float64, n+1)
	copy(voltage, trace)
	voltage[n] = finalV
	copy(unsafe.Slice((*float64)(unsafe.Pointer(voltageOutput)), n+1), voltage)
	if n > 0 {
		copy(unsafe.Slice((*uint8)(unsafe.Pointer(eventOutput)), n), events)
	}
	count := 0
	for _, event := range events {
		count += int(event)
	}
	return C.int64_t(count)
}

func main() {}
