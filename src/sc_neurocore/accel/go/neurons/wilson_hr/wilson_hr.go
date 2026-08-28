// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Wilson 1999 polynomial cortical model (parity with wilson_hr.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libwilsonhr.so wilson_hr.go`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `wilson_hr_simulate_c` reproduces
// `sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron.simulate` bit-for-bit.
// The right-hand side is exact polynomial arithmetic, so an identical RK4
// operation order yields an identical continuous v trace, upward-crossing
// count, and final state.
//
// Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.
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

// wilson_hr_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+2: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final r. Returns the spike count.
//
//export wilson_hr_simulate_c
func wilson_hr_simulate_c(
	v0, r0, capacitance, tauR, vPeak, dt C.double,
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
	cap := float64(capacitance)
	tr := float64(tauR)
	thr := float64(vPeak)
	pdt := float64(dt)
	cur := float64(current)
	if !allFinite(v, r, cap, tr, thr, pdt, cur) || cap <= 0 || tr <= 0 || pdt <= 0 {
		return -1
	}
	deriv := func(vv, rr float64) (float64, float64) {
		poly := -(17.81 + 47.71*vv + 32.63*vv*vv) * (vv - 0.55)
		syn := -26.0 * rr * (vv + 0.92)
		return (poly + syn + cur) / cap, (-rr + 1.35*vv + 1.03) / tr
	}
	var spikes int64
	for t := 0; t < n; t++ {
		vPrev := v
		dv1, dr1 := deriv(v, r)
		dv2, dr2 := deriv(v+0.5*pdt*dv1, r+0.5*pdt*dr1)
		dv3, dr3 := deriv(v+0.5*pdt*dv2, r+0.5*pdt*dr2)
		dv4, dr4 := deriv(v+pdt*dv3, r+pdt*dr3)
		if !allFinite(dv1, dr1, dv2, dr2, dv3, dr3, dv4, dr4) {
			return -1
		}
		nextV := v + pdt*(dv1+2.0*dv2+2.0*dv3+dv4)/6.0
		nextR := r + pdt*(dr1+2.0*dr2+2.0*dr3+dr4)/6.0
		if !allFinite(nextV, nextR) {
			return -1
		}
		v, r = nextV, nextR
		if v >= thr && vPrev < thr {
			spikes++
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = r
	return C.longlong(spikes)
}

func main() {} // required for c-shared
