// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the Lapicque exact-flow recurrence

// Package main exposes services.LapicqueNeuronState as a C-shared library.
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

//export lapicque_simulate_c
func lapicque_simulate_c(
	v C.double,
	vRest C.double,
	vReset C.double,
	vThreshold C.double,
	tau C.double,
	resistance C.double,
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
	state := &services.LapicqueNeuronState{
		V:          float64(v),
		VRest:      float64(vRest),
		VReset:     float64(vReset),
		VThreshold: float64(vThreshold),
		Tau:        float64(tau),
		Resistance: float64(resistance),
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

//export lapicque_simulate_complete_c
func lapicque_simulate_complete_c(
	v C.double,
	vRest C.double,
	vReset C.double,
	vThreshold C.double,
	tau C.double,
	resistance C.double,
	dt C.double,
	capacitance C.double,
	seriesResistance C.double,
	polarizationResistance C.double,
	excited C.int64_t,
	sourceProfile C.int64_t,
	nSteps C.int64_t,
	drive C.double,
	outputVoltage *C.double,
	outputEvents *C.uint8_t,
) C.int64_t {
	if nSteps < 0 || outputVoltage == nil || (nSteps > 0 && outputEvents == nil) ||
		(excited != 0 && excited != 1) || (sourceProfile != 0 && sourceProfile != 1) {
		return -1
	}
	n := int(nSteps)
	if C.int64_t(n) != nSteps || n > int(^uint(0)>>1)-1 {
		return -1
	}
	state := services.LapicqueNeuronState{
		V:                      float64(v),
		VRest:                  float64(vRest),
		VReset:                 float64(vReset),
		VThreshold:             float64(vThreshold),
		Tau:                    float64(tau),
		Resistance:             float64(resistance),
		Dt:                     float64(dt),
		Capacitance:            float64(capacitance),
		SeriesResistance:       float64(seriesResistance),
		PolarizationResistance: float64(polarizationResistance),
		Excited:                excited == 1,
		SourceProfile:          sourceProfile == 1,
	}
	trace, events, finalState, err := services.SimulateLapicqueComplete(
		state, n, float64(drive),
	)
	if err != nil {
		return -1
	}
	voltageDestination := unsafe.Slice((*float64)(unsafe.Pointer(outputVoltage)), n+1)
	copy(voltageDestination, trace)
	voltageDestination[n] = finalState.V
	if n > 0 {
		eventDestination := unsafe.Slice((*uint8)(unsafe.Pointer(outputEvents)), n)
		copy(eventDestination, events)
	}
	eventCount := int64(0)
	for _, event := range events {
		eventCount += int64(event)
	}
	return C.int64_t(eventCount)
}

func main() {}
