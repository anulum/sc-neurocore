// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go SC triangular project recurrence

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libsc_triangular_mckean.so mckean.go`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `sc_triangular_mckean_simulate_c` reproduces
// `sc_neurocore.neurons.models.sc_triangular_mckean.SCTriangularMcKeanNeuron.simulate` bit-for-bit. The
// piecewise-linear right-hand side is exact arithmetic (additions, multiplications
// and branch selection), so an identical RK4 operation order yields an identical
// v trace, upward-crossing spike count, and final state.
//
// Provenance: SC project recurrence; no external paper attribution.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

// sc_triangular_mckean_simulate_c runs n RK4 steps under a constant input. The caller allocates
// a trace buffer of length n+2: indices [0, n) receive the v trace, index n the
// final v, index n+1 the final w. Returns the upward-crossing spike count.
//
//export sc_triangular_mckean_simulate_c
func sc_triangular_mckean_simulate_c(
	v0, w0, a, eps, gamma, dt, vPeak C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	v := float64(v0)
	w := float64(w0)
	pa := float64(a)
	pe := float64(eps)
	pg := float64(gamma)
	pdt := float64(dt)
	thr := float64(vPeak)
	cur := float64(current)
	halfA := pa / 2.0
	mid := (1.0 + pa) / 2.0
	fv := func(x float64) float64 {
		if x < halfA {
			return -x
		}
		if x < mid {
			return x - pa
		}
		return 1.0 - x
	}
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		dv1 := fv(v) - w + cur
		dw1 := pe * (v - pg*w)
		v2 := v + 0.5*pdt*dv1
		w2 := w + 0.5*pdt*dw1
		dv2 := fv(v2) - w2 + cur
		dw2 := pe * (v2 - pg*w2)
		v3 := v + 0.5*pdt*dv2
		w3 := w + 0.5*pdt*dw2
		dv3 := fv(v3) - w3 + cur
		dw3 := pe * (v3 - pg*w3)
		v4 := v + pdt*dv3
		w4 := w + pdt*dw3
		dv4 := fv(v4) - w4 + cur
		dw4 := pe * (v4 - pg*w4)
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
