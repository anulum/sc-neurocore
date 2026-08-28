// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Cazelles four-branch map

package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

//export cazelles_map_simulate_c
func cazelles_map_simulate_c(
	x, alpha, x0, x1, x2, x3, x4 C.double,
	a1, a2, a3, a4, b1, b2, b3, b4 C.double,
	exponent, nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+1)
	xv := float64(x)
	av := float64(alpha)
	lo0, lo1, lo2 := float64(x0), float64(x1), float64(x2)
	lo3, lo4 := float64(x3), float64(x4)
	aa1, aa2, aa3, aa4 := float64(a1), float64(a2), float64(a3), float64(a4)
	bb1, bb2, bb3, bb4 := float64(b1), float64(b2), float64(b3), float64(b4)
	cur := float64(current)
	exp := int(exponent)
	if !finite(xv, av, lo0, lo1, lo2, lo3, lo4, aa1, aa2, aa3, aa4, bb1, bb2, bb3, bb4, cur) ||
		av < 0.0 || av >= 1.0 || (exp != 1 && exp != 2) ||
		!(lo0 < lo1 && lo1 < lo2 && lo2 < lo3 && lo3 < lo4) || xv < lo0 || xv > lo4 {
		trace[n] = xv
		return -1
	}
	var events int64
	for index := 0; index < n; index++ {
		var base float64
		if xv < lo1 {
			base = aa1 + bb1*xv
		} else if xv < lo2 {
			base = aa2 + bb2*xv
		} else if xv < lo3 {
			base = aa3 + bb3*xv
		} else {
			base = aa4 + bb4*xv
		}
		power := xv
		if exp == 2 {
			power = xv * xv
		}
		candidate := base + av*power + cur
		tolerance := 8.0 * 2.220446049250313e-16 * math.Max(1.0, math.Max(math.Abs(lo0), math.Abs(lo4)))
		if candidate < lo0 && candidate >= lo0-tolerance {
			candidate = lo0
		} else if candidate > lo4 && candidate <= lo4+tolerance {
			candidate = lo4
		}
		if !finite(candidate) || candidate < lo0 || candidate > lo4 {
			trace[n] = xv
			return -2
		}
		if xv >= lo1 && candidate < lo1 {
			events++
		}
		xv = candidate
		trace[index] = xv
	}
	trace[n] = xv
	return C.longlong(events)
}

func main() {}
