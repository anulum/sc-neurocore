// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Ermentrout-Kopell theta map (parity with ermentrout_kopell_map_neuron.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libermentrout.so ermentrout_kopell_map.go`)
// that the Python dispatcher loads via ctypes.
//
// Parity contract: `ermentrout_kopell_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron.ErmentroutKopellMapNeuron.simulate`.
// The only transcendental is `cos`, and the theta neuron is a non-chaotic phase
// oscillator, so Go's `math.Cos` (pure-Go, may differ from the reference libm by
// a ULP) does not amplify: the trace stays within a small ULP band and the spike
// counts match. The wrap uses `math.Mod` adjusted to the floored remainder,
// matching Python's `theta % (2*pi)`.
//
// Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233-253.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// foldTwoPi returns the floored remainder of v modulo 2*pi, i.e. v in [0, 2*pi).
func foldTwoPi(v, twoPi float64) float64 {
	r := math.Mod(v, twoPi)
	if r < 0.0 {
		r += twoPi
	}
	return r
}

// ermentrout_kopell_map_simulate_c runs n steps of the theta map under a
// constant input. The caller allocates a trace buffer of length n+1: indices
// [0, n) receive the theta trace (wrapped to [0, 2*pi)), index n the final
// theta. Returns the upward-crossing spike count.
//
//export ermentrout_kopell_map_simulate_c
func ermentrout_kopell_map_simulate_c(
	theta0, dt, gain, thetaThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+1)
	theta := float64(theta0)
	d := float64(dt)
	thr := float64(thetaThreshold)
	inp := float64(gain) * float64(current)
	twoPi := 2.0 * math.Pi
	var spikes int64
	for t := 0; t < n; t++ {
		thetaPrev := theta
		cosTheta := math.Cos(theta)
		dTheta := (1.0 - cosTheta) + (1.0+cosTheta)*inp
		thetaNext := theta + d*dTheta
		if thetaNext >= thr && thetaPrev < thr {
			spikes++
		}
		theta = foldTwoPi(thetaNext, twoPi)
		trace[t] = theta
	}
	if n > 0 {
		trace[n] = theta
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
