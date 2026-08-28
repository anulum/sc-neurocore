// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained upward-crossing Rulkov-map Go backend

// Package main exposes the retained map through a C shared-library ABI.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// sc_upward_crossing_rulkov_map_simulate_c runs the historical recurrence.
// The caller provides n+2 Float64 slots. A negative return reports invalid
// input or a non-finite candidate.
//
//export sc_upward_crossing_rulkov_map_simulate_c
func sc_upward_crossing_rulkov_map_simulate_c(
	x0, y0, alpha, sigma, mu, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	x := float64(x0)
	y := float64(y0)
	a := float64(alpha)
	sig := float64(sigma)
	m := float64(mu)
	threshold := float64(xThreshold)
	drive := float64(current)
	if n < 0 || tracePtr == nil || !finite(x, y, a, sig, m, threshold, drive) || a <= 0.0 || m <= 0.0 {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	var events int64
	for step := 0; step < n; step++ {
		previousX := x
		boundary := a + y + drive
		var xNext float64
		if x <= 0.0 {
			xNext = a/(1.0-x) + y + drive
		} else if x < boundary {
			xNext = boundary
		} else {
			xNext = -1.0
		}
		yNext := y - m*(x+1.0) + m*sig
		if !finite(xNext, yNext) {
			return -1
		}
		x, y = xNext, yNext
		trace[step] = x
		if x >= threshold && previousX < threshold {
			events++
		}
	}
	trace[n] = x
	trace[n+1] = y
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
