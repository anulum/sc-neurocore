// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the Chialvo map service

// Package main exposes the checked Chialvo recurrence as a C shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

// chialvo_map_simulate_c runs the source recurrence under a constant additive
// perturbation. The caller supplies n+2 doubles: the x trace followed by final
// x and final y. A negative return value reports rejected input or state.
//
//export chialvo_map_simulate_c
func chialvo_map_simulate_c(
	x0, y0, a, b, c, k, xThreshold C.double,
	nSteps C.int,
	current C.double,
	tracePointer *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePointer == nil {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePointer)), n+2)
	state := &services.ChialvoMapNeuronState{
		X:          float64(x0),
		Y:          float64(y0),
		A:          float64(a),
		B:          float64(b),
		C:          float64(c),
		K:          float64(k),
		XThreshold: float64(xThreshold),
	}
	xTrace, spikes, err := state.Simulate(n, float64(current))
	if err != nil {
		trace[n] = state.X
		trace[n+1] = state.Y
		return -1
	}
	copy(trace, xTrace)
	trace[n] = state.X
	trace[n+1] = state.Y
	return C.longlong(spikes)
}

func main() {}
