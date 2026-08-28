// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go retained clipped rational-recovery map

// Package main exposes the count-neutral project recurrence over a C ABI.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

//export sc_clipped_rational_recovery_map_simulate_c
func sc_clipped_rational_recovery_map_simulate_c(
	x0, y0, alpha, beta, j, xThreshold, clipBound C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	if nSteps < 0 || tracePtr == nil {
		return -1
	}
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	x := float64(x0)
	y := float64(y0)
	palpha := float64(alpha)
	pbeta := float64(beta)
	pj := float64(j)
	threshold := float64(xThreshold)
	bound := float64(clipBound)
	cur := float64(current)
	values := [...]float64{x, y, palpha, pbeta, pj, threshold, bound, cur}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return -1
		}
	}
	if !(palpha > 0.0 && pbeta > 0.0 && bound > 0.0 &&
		math.Abs(x) <= bound && math.Abs(y) <= bound) {
		return -1
	}
	var events int64
	for index := 0; index < n; index++ {
		xPrevious := x
		field := palpha * x
		if x >= 0.0 {
			field = palpha * x / (1.0 + palpha*x)
		}
		xCandidate := field + y + cur + pj
		yCandidate := y - pbeta*(x+1.0)
		if math.IsNaN(xCandidate) || math.IsInf(xCandidate, 0) ||
			math.IsNaN(yCandidate) || math.IsInf(yCandidate, 0) {
			return -1
		}
		x = math.Max(-bound, math.Min(bound, xCandidate))
		y = math.Max(-bound, math.Min(bound, yCandidate))
		trace[index] = x
		if x >= threshold && xPrevious < threshold {
			events++
		}
	}
	trace[n] = x
	trace[n+1] = y
	return C.longlong(events)
}

func main() {}
