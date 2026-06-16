// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Terman-Wang 1995 LEGION relaxation oscillator (parity with terman_wang.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libtermanwang.so terman_wang.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `terman_wang_simulate_c` reproduces
// `sc_neurocore.neurons.models.terman_wang.TermanWangOscillator.simulate`. The
// cubic is exact (v*v*v, matching the engine v.powi(3)); the tanh gating uses
// Go's math.Tanh, so the trace is within a per-step ULP band of the NumPy
// reference (the 2D relaxation oscillator is non-chaotic, so it does not amplify)
// with identical spike counts.
//
// Reference: Terman, D. & Wang, D.L. (1995). Physica D 81:148-176.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// terman_wang_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+2: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final w. Returns the upward-crossing spike
// count.
//
//export terman_wang_simulate_c
func terman_wang_simulate_c(
	v0, w0, alpha, beta, eps, rho, dt, vPeak C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	v := float64(v0)
	w := float64(w0)
	pa := float64(alpha)
	pb := float64(beta)
	pe := float64(eps)
	prho := float64(rho)
	pdt := float64(dt)
	thr := float64(vPeak)
	cur := float64(current)
	deriv := func(vv, ww float64) (float64, float64) {
		f := 3.0*vv - vv*vv*vv + 2.0
		g := pa * (1.0 + math.Tanh(vv/pb))
		return f - ww + cur + prho, pe * (g - ww)
	}
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		dv1, dw1 := deriv(v, w)
		dv2, dw2 := deriv(v+0.5*pdt*dv1, w+0.5*pdt*dw1)
		dv3, dw3 := deriv(v+0.5*pdt*dv2, w+0.5*pdt*dw2)
		dv4, dw4 := deriv(v+pdt*dv3, w+pdt*dw3)
		v = v + pdt*(dv1+2.0*dv2+2.0*dv3+dv4)/6.0
		w = w + pdt*(dw1+2.0*dw2+2.0*dw3+dw4)/6.0
		trace[t] = v
		if v >= thr && vPrev < thr {
			spikes++
		}
	}
	trace[n] = v
	trace[n+1] = w
	return C.longlong(spikes)
}

func main() {} // required for c-shared
