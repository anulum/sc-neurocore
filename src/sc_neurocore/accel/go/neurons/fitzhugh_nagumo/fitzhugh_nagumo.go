// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go FitzHugh-Nagumo RK4 simulator (parity with fitzhugh_nagumo.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libfhn.so fitzhugh_nagumo.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `fitzhugh_nagumo_simulate_c` reproduces
// `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.simulate`
// bit-for-bit. The RK4 right-hand side is exact arithmetic (the cube is written
// `v*v*v`, additions and multiplications, no transcendental functions; Go does
// not contract floating-point multiply-adds), so an identical operation order
// yields an identical trace, spike count and final state.
//
// Reference: FitzHugh, R. (1961). Biophys. J. 1:445-466.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

func rhs(v, w, a, b, eps, cur float64) (float64, float64) {
	dv := v - v*v*v/3.0 - w + cur
	dw := eps * (v + a - b*w)
	return dv, dw
}

// fitzhugh_nagumo_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+2: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final w. Returns the upward-crossing spike
// count.
//
//export fitzhugh_nagumo_simulate_c
func fitzhugh_nagumo_simulate_c(
	v0, w0, a, b, epsilon, dt, vThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	v := float64(v0)
	w := float64(w0)
	aa := float64(a)
	bb := float64(b)
	eps := float64(epsilon)
	d := float64(dt)
	thr := float64(vThreshold)
	cur := float64(current)
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		k1v, k1w := rhs(v, w, aa, bb, eps, cur)
		k2v, k2w := rhs(v+0.5*d*k1v, w+0.5*d*k1w, aa, bb, eps, cur)
		k3v, k3w := rhs(v+0.5*d*k2v, w+0.5*d*k2w, aa, bb, eps, cur)
		k4v, k4w := rhs(v+d*k3v, w+d*k3w, aa, bb, eps, cur)
		v = v + d*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
		w = w + d*(k1w+2.0*k2w+2.0*k3w+k4w)/6.0
		trace[t] = v
		if v >= thr && vPrev < thr {
			spikes++
		}
	}
	if n > 0 {
		trace[n] = v
		trace[n+1] = w
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
