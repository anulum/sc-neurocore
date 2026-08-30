// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the maintained AdEx service recurrence

// Package main exposes the real services.AdExNeuronState recurrence as a
// C-shared library. Build with:
//
//	go build -buildmode=c-shared -o libadex.so adex.go
//
// The caller supplies nSteps+2 doubles: the post-reset voltage trace followed
// by final (v, w). A negative return rejects the contract without writing the
// caller's output buffer.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export adex_simulate_c
func adex_simulate_c(
	v0, w0, vRest, vReset, vThreshold, vRh C.double,
	deltaT, tau, tauW, a, b, cM, dt C.double,
	nSteps C.int64_t, current C.double,
	outputPtr *C.double,
) C.int64_t {
	n := int64(nSteps)
	if n < 0 || outputPtr == nil {
		return -1
	}
	state := services.AdExNeuronState{
		V:          float64(v0),
		W:          float64(w0),
		VRest:      float64(vRest),
		VReset:     float64(vReset),
		VThreshold: float64(vThreshold),
		VRh:        float64(vRh),
		DeltaT:     float64(deltaT),
		Tau:        float64(tau),
		TauW:       float64(tauW),
		A:          float64(a),
		B:          float64(b),
		CM:         float64(cM),
		Dt:         float64(dt),
	}
	if !state.Valid() {
		return -1
	}
	trace, _, _, spikes, err := state.SimulateComplete(int(n), float64(current))
	if err != nil {
		return -1
	}
	output := unsafe.Slice((*float64)(unsafe.Pointer(outputPtr)), int(n)+2)
	copy(output, trace)
	output[n] = state.V
	output[n+1] = state.W
	return C.int64_t(spikes)
}

//export adex_simulate_complete_c
func adex_simulate_complete_c(
	v0, w0, vRest, vReset, vThreshold, vRh C.double,
	deltaT, tau, tauW, a, b, cM, dt C.double,
	nSteps C.int64_t, current C.double,
	vPtr, wPtr *C.double,
	eventPtr *C.uint8_t,
) C.int64_t {
	n := int64(nSteps)
	if n < 0 || vPtr == nil || wPtr == nil || (n > 0 && eventPtr == nil) {
		return -1
	}
	state := services.AdExNeuronState{
		V: float64(v0), W: float64(w0), VRest: float64(vRest),
		VReset: float64(vReset), VThreshold: float64(vThreshold), VRh: float64(vRh),
		DeltaT: float64(deltaT), Tau: float64(tau), TauW: float64(tauW),
		A: float64(a), B: float64(b), CM: float64(cM), Dt: float64(dt),
	}
	vTrace, wTrace, events, spikes, err := state.SimulateComplete(int(n), float64(current))
	if err != nil {
		return -1
	}
	vOutput := unsafe.Slice((*float64)(unsafe.Pointer(vPtr)), int(n)+1)
	wOutput := unsafe.Slice((*float64)(unsafe.Pointer(wPtr)), int(n)+1)
	copy(vOutput, vTrace)
	copy(wOutput, wTrace)
	if n > 0 {
		eventOutput := unsafe.Slice((*uint8)(unsafe.Pointer(eventPtr)), int(n))
		copy(eventOutput, events)
	}
	vOutput[n] = state.V
	wOutput[n] = state.W
	return C.int64_t(spikes)
}

func main() {}
