// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go retained resetting Wilson-HR project recurrence

// Package main exposes the retained SC resetting Wilson-HR recurrence through
// a C ABI. It preserves the historical unit-capacitance RK4 dynamics and hard
// voltage reset under a distinct, non-source-attributed model identity.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func allFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// sc_resetting_wilson_hr_simulate_c advances n constant-current RK4 steps.
// The caller owns a buffer of n+2 doubles: the voltage trace followed by final
// voltage and recovery state. A return value of -1 rejects the entire batch;
// the caller must not commit the output buffer or dynamic state.
//
//export sc_resetting_wilson_hr_simulate_c
func sc_resetting_wilson_hr_simulate_c(
	v0, r0, tauR, vPeak, dt C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePtr == nil {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	v := float64(v0)
	r := float64(r0)
	tr := float64(tauR)
	threshold := float64(vPeak)
	step := float64(dt)
	input := float64(current)
	if !allFinite(v, r, tr, threshold, step, input) || tr <= 0 || step <= 0 {
		return -1
	}
	derivatives := func(voltage, recovery float64) (float64, float64) {
		polynomial := -(17.81 + 47.71*voltage + 32.63*voltage*voltage) * (voltage - 0.55)
		recoveryCurrent := -26.0 * recovery * (voltage + 0.92)
		return polynomial + recoveryCurrent + input, (-recovery + 1.35*voltage + 1.03) / tr
	}
	var events int64
	for index := 0; index < n; index++ {
		dv1, dr1 := derivatives(v, r)
		dv2, dr2 := derivatives(v+0.5*step*dv1, r+0.5*step*dr1)
		dv3, dr3 := derivatives(v+0.5*step*dv2, r+0.5*step*dr2)
		dv4, dr4 := derivatives(v+step*dv3, r+step*dr3)
		if !allFinite(dv1, dr1, dv2, dr2, dv3, dr3, dv4, dr4) {
			return -1
		}
		nextV := v + step*(dv1+2.0*dv2+2.0*dv3+dv4)/6.0
		nextR := r + step*(dr1+2.0*dr2+2.0*dr3+dr4)/6.0
		if !allFinite(nextV, nextR) {
			return -1
		}
		if nextV >= threshold {
			nextV = -0.7
			events++
		}
		v, r = nextV, nextR
		trace[index] = v
	}
	trace[n] = v
	trace[n+1] = r
	return C.longlong(events)
}

func main() {}
