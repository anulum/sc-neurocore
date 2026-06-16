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
// operation order yields an identical v trace (already hard-reset to -0.7 on
// spiking steps), spike count, and final state.
//
// Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

// wilson_hr_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+2: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final r. Returns the spike count.
//
//export wilson_hr_simulate_c
func wilson_hr_simulate_c(
	v0, r0, tauR, vPeak, dt C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	v := float64(v0)
	r := float64(r0)
	tr := float64(tauR)
	thr := float64(vPeak)
	pdt := float64(dt)
	cur := float64(current)
	deriv := func(vv, rr float64) (float64, float64) {
		poly := -(17.81 + 47.71*vv + 32.63*vv*vv) * (vv - 0.55)
		syn := -26.0 * rr * (vv + 0.92)
		return poly + syn + cur, (-rr + 1.35*vv + 1.03) / tr
	}
	var spikes int64
	for t := 0; t < n; t++ {
		dv1, dr1 := deriv(v, r)
		dv2, dr2 := deriv(v+0.5*pdt*dv1, r+0.5*pdt*dr1)
		dv3, dr3 := deriv(v+0.5*pdt*dv2, r+0.5*pdt*dr2)
		dv4, dr4 := deriv(v+pdt*dv3, r+pdt*dr3)
		v = v + pdt*(dv1+2.0*dv2+2.0*dv3+dv4)/6.0
		r = r + pdt*(dr1+2.0*dr2+2.0*dr3+dr4)/6.0
		if v >= thr {
			v = -0.7
			spikes++
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = r
	return C.longlong(spikes)
}

func main() {} // required for c-shared
