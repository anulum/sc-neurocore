// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Pernarowski 1994 beta-cell burster (parity with pernarowski.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libpernarowski.so pernarowski.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `pernarowski_simulate_c` reproduces
// `sc_neurocore.neurons.models.pernarowski.PernarowskiNeuron.simulate`
// bit-for-bit. The right-hand side is exact polynomial arithmetic (the cubic uses
// v*v*v, matching the engine v.powi(3)), so an identical RK4 operation order
// yields an identical v trace, upward-crossing spike count, and final state.
//
// Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814-832.
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

// pernarowski_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+3: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final w, index n+2 the final z. Returns the
// upward-crossing spike count.
//
//export pernarowski_simulate_c
func pernarowski_simulate_c(
	v0, w0, z0, alpha, beta, eps1, eps2, gamma, dt, vThreshold C.double,
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
	z := float64(z0)
	pa := float64(alpha)
	pb := float64(beta)
	pe1 := float64(eps1)
	pe2 := float64(eps2)
	pg := float64(gamma)
	pdt := float64(dt)
	thr := float64(vThreshold)
	cur := float64(current)
	if !allFinite(v, w, z, pa, pb, pe1, pe2, pg, pdt, thr, cur) || pe1 <= 0 || pe2 <= 0 || pg <= 0 || pdt <= 0 {
		return -1
	}
	deriv := func(vv, ww, zz float64) (float64, float64, float64) {
		dv := vv - vv*vv*vv/3.0 - ww - zz + cur
		dw := pe1 * (vv - pg*ww + pa)
		dz := pe2 * (pb*(vv+0.7) - zz)
		return dv, dw, dz
	}
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		dv1, dw1, dz1 := deriv(v, w, z)
		dv2, dw2, dz2 := deriv(v+0.5*pdt*dv1, w+0.5*pdt*dw1, z+0.5*pdt*dz1)
		dv3, dw3, dz3 := deriv(v+0.5*pdt*dv2, w+0.5*pdt*dw2, z+0.5*pdt*dz2)
		dv4, dw4, dz4 := deriv(v+pdt*dv3, w+pdt*dw3, z+pdt*dz3)
		if !allFinite(dv1, dw1, dz1, dv2, dw2, dz2, dv3, dw3, dz3, dv4, dw4, dz4) {
			return -1
		}
		nextV := v + pdt*(dv1+2.0*dv2+2.0*dv3+dv4)/6.0
		nextW := w + pdt*(dw1+2.0*dw2+2.0*dw3+dw4)/6.0
		nextZ := z + pdt*(dz1+2.0*dz2+2.0*dz3+dz4)/6.0
		if !allFinite(nextV, nextW, nextZ) {
			return -1
		}
		v, w, z = nextV, nextW, nextZ
		if v >= thr && vPrev < thr {
			spikes++
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = w
	trace[n+2] = z
	return C.longlong(spikes)
}

func main() {} // required for c-shared
