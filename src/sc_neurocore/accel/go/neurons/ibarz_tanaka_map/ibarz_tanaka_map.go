// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Ibarz analysis profile of the Shilnikov-Rulkov map

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

// ibarz_tanaka_map_simulate_c runs the Shilnikov-Rulkov map as profiled in
// Ibarz et al. (2007), Eqs. 2-3.
// The caller provides n+2 Float64 slots: the v trace followed by final v and u.
// A negative return reports invalid input or arithmetic before any output
// buffer mutation. A validation pass makes the batch failure-atomic.
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
	probeV := v
	probeU := u
	for step := 0; step < n; step++ {
		var ok bool
		probeV, probeU, _, ok = candidate(probeV, probeU, a, m, sig, cur)
		if !ok {
			return -2
		}
	}

	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	var events int64
	for step := 0; step < n; step++ {
		var event int64
		var ok bool
		v, u, event, ok = candidate(v, u, a, m, sig, cur)
		if !ok {
			return -2 // unreachable after the identical validation pass
		}
		events += event
		trace[step] = v
	}
	trace[n] = v
	trace[n+1] = u
	return C.longlong(events)
}

func candidate(v, u, alpha, mu, sigma, current float64) (float64, float64, int64, bool) {
	lower := -1.0 - alpha/2.0
	upper := 1.0 + current + u
	var vNext float64
	var event int64
	switch {
	case v < lower:
		vNext = -(alpha*alpha)/4.0 - alpha + current + u
	case v <= 0.0:
		vNext = alpha*v + (v+1.0)*(v+1.0) + current + u
	case v < upper:
		vNext = upper
	default:
		vNext = -1.0
		event = 1
	}
	uNext := u - mu*(v+1.0-sigma)
	return vNext, uNext, event, finite(vNext, uNext)
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
