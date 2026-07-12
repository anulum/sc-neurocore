// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Ibarz-Tanaka 2007 four-branch map

// Package main exposes the source recurrence through a C shared-library ABI.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// ibarz_tanaka_map_simulate_c runs Ibarz et al. (2007), Eqs. 2-3.
// The caller provides n+2 Float64 slots: the v trace followed by final v and u.
// A negative return reports invalid input before any output buffer mutation.
//
//export ibarz_tanaka_map_simulate_c
func ibarz_tanaka_map_simulate_c(
	v0, u0, alpha, mu, sigma C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	v := float64(v0)
	u := float64(u0)
	a := float64(alpha)
	m := float64(mu)
	sig := float64(sigma)
	cur := float64(current)
	if n < 0 || tracePtr == nil || !finite(v, u, a, m, sig, cur) || a <= 0.0 || m <= 0.0 {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	var events int64
	for step := 0; step < n; step++ {
		lower := -1.0 - a/2.0
		upper := 1.0 + cur + u
		var vNext float64
		switch {
		case v < lower:
			vNext = -(a*a)/4.0 - a + cur + u
		case v <= 0.0:
			vNext = a*v + (v+1.0)*(v+1.0) + cur + u
		case v < upper:
			vNext = upper
		default:
			vNext = -1.0
			events++
		}
		uNext := u - m*(v+1.0-sig)
		if !finite(vNext, uNext) {
			return -1
		}
		v, u = vNext, uNext
		trace[step] = v
	}
	trace[n] = v
	trace[n+1] = u
	return C.longlong(events)
}

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func main() {}
