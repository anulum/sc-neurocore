// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go FitzHugh-Rinzel RK4 simulator (parity with fitzhugh_rinzel.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libfhr.so fitzhugh_rinzel.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `fitzhugh_rinzel_simulate_c` reproduces
// `sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron.simulate`
// bit-for-bit. The RK4 right-hand side is exact arithmetic (the cube is written
// `v*v*v`, additions and multiplications, no transcendental functions; Go does
// not contract floating-point multiply-adds), so an identical operation order
// yields an identical trace, spike count and final state.
//
// Reference: FitzHugh, R. (1976); Rinzel, J. (1987).
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

func deriv(v, w, y, a, b, c, d, delta, mu, cur float64) (float64, float64, float64) {
	dv := v - v*v*v/3.0 - w + y + cur
	dw := delta * (a + v - b*w)
	dy := mu * (c - v - d*y)
	return dv, dw, dy
}

// fitzhugh_rinzel_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+3: indices [0, n) receive the v trace,
// index n the final v, n+1 the final w, n+2 the final y. Returns the
// upward-crossing spike count.
//
//export fitzhugh_rinzel_simulate_c
func fitzhugh_rinzel_simulate_c(
	v0, w0, y0, a, b, c, d, delta, mu, dt, vThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePtr == nil {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+3)
	v := float64(v0)
	w := float64(w0)
	y := float64(y0)
	aa := float64(a)
	bb := float64(b)
	cc := float64(c)
	dd := float64(d)
	del := float64(delta)
	muu := float64(mu)
	dtt := float64(dt)
	thr := float64(vThreshold)
	cur := float64(current)
	if !allFinite(v, w, y, aa, bb, cc, dd, del, muu, dtt, thr, cur) || bb <= 0 || dd <= 0 || del <= 0 || muu <= 0 || dtt <= 0 {
		return -1
	}
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		k1v, k1w, k1y := deriv(v, w, y, aa, bb, cc, dd, del, muu, cur)
		k2v, k2w, k2y := deriv(v+0.5*dtt*k1v, w+0.5*dtt*k1w, y+0.5*dtt*k1y, aa, bb, cc, dd, del, muu, cur)
		k3v, k3w, k3y := deriv(v+0.5*dtt*k2v, w+0.5*dtt*k2w, y+0.5*dtt*k2y, aa, bb, cc, dd, del, muu, cur)
		k4v, k4w, k4y := deriv(v+dtt*k3v, w+dtt*k3w, y+dtt*k3y, aa, bb, cc, dd, del, muu, cur)
		if !allFinite(k1v, k1w, k1y, k2v, k2w, k2y, k3v, k3w, k3y, k4v, k4w, k4y) {
			return -1
		}
		nextV := v + dtt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
		nextW := w + dtt*(k1w+2.0*k2w+2.0*k3w+k4w)/6.0
		nextY := y + dtt*(k1y+2.0*k2y+2.0*k3y+k4y)/6.0
		if !allFinite(nextV, nextW, nextY) {
			return -1
		}
		v, w, y = nextV, nextW, nextY
		trace[t] = v
		if v >= thr && vPrev < thr {
			spikes++
		}
	}
	if n > 0 {
		trace[n] = v
		trace[n+1] = w
		trace[n+2] = y
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
