// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Izhikevich 2007 RK4 simulator (parity with izhikevich2007.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libizh2007.so izhikevich2007.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `izhikevich2007_simulate_c` reproduces
// `sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron.simulate`
// bit-for-bit. The NeuroML right-hand side `k (v-vr)(v-vt)/C` is exact
// arithmetic (products, a sum and a division, no transcendental functions; Go
// does not contract floating-point multiply-adds), so an identical operation
// order yields an identical trace, spike count and final state.
//
// Reference: Izhikevich, E.M. (2007), Dynamical Systems in Neuroscience.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func rhs(v, u, cap, k, vr, vt, a, b, cur float64) (float64, float64) {
	dv := (k*(v-vr)*(v-vt) - u + cur) / cap
	du := a * (b*(v-vr) - u)
	return dv, du
}

// izhikevich2007_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+2: indices [0, n) receive the v trace,
// index n the final v, index n+1 the final u. Returns the spike count.
//
//export izhikevich2007_simulate_c
func izhikevich2007_simulate_c(
	v0, u0, cap, k, vr, vt, vpeak, a, b, c, d, dt C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	v := float64(v0)
	u := float64(u0)
	cp := float64(cap)
	kk := float64(k)
	vrf := float64(vr)
	vtf := float64(vt)
	vpk := float64(vpeak)
	aa := float64(a)
	bb := float64(b)
	cc := float64(c)
	dd := float64(d)
	dtt := float64(dt)
	cur := float64(current)
	if n < 0 || tracePtr == nil || cp <= 0.0 || dtt <= 0.0 ||
		!allFinite(v, u, cp, kk, vrf, vtf, vpk, aa, bb, cc, dd, dtt, cur) {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	dt6 := dtt / 6.0
	var spikes int64
	for t := 0; t < n; t++ {
		k1v, k1u := rhs(v, u, cp, kk, vrf, vtf, aa, bb, cur)
		k2v, k2u := rhs(v+0.5*dtt*k1v, u+0.5*dtt*k1u, cp, kk, vrf, vtf, aa, bb, cur)
		k3v, k3u := rhs(v+0.5*dtt*k2v, u+0.5*dtt*k2u, cp, kk, vrf, vtf, aa, bb, cur)
		k4v, k4u := rhs(v+dtt*k3v, u+dtt*k3u, cp, kk, vrf, vtf, aa, bb, cur)
		v = v + dt6*(k1v+2.0*k2v+2.0*k3v+k4v)
		u = u + dt6*(k1u+2.0*k2u+2.0*k3u+k4u)
		if v >= vpk {
			v = cc
			u = u + dd
			spikes++
		}
		if !allFinite(v, u) {
			return -2
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = u
	return C.longlong(spikes)
}

func allFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func main() {} // required for c-shared
