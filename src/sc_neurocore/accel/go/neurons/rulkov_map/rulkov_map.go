// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Rulkov 2001 fast/slow map (parity with rulkov_map.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o librulkov.so rulkov_map.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `rulkov_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.rulkov_map.RulkovMapNeuron.simulate`
// bit-for-bit. The fast map is exact floating-point arithmetic (one division,
// additions, multiplications), so an identical operation order yields an
// identical trace, upward-crossing spike count and final state.
//
// Reference: Rulkov, N.F. (2002). Phys. Rev. E 65:041922.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

// rulkov_map_simulate_c runs n steps of the fast/slow map under a constant
// input. The caller allocates a trace buffer of length n+2: indices [0, n)
// receive the x trace, index n the final x, index n+1 the final y. Returns the
// upward-crossing spike count.
//
//export rulkov_map_simulate_c
func rulkov_map_simulate_c(
	x0, y0, alpha, sigma, mu, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	x := float64(x0)
	y := float64(y0)
	a := float64(alpha)
	sig := float64(sigma)
	m := float64(mu)
	thr := float64(xThreshold)
	cur := float64(current)
	var spikes int64
	for t := 0; t < n; t++ {
		xPrev := x
		branchBoundary := a + y + cur
		var xNew float64
		if x <= 0.0 {
			xNew = a/(1.0-x) + y + cur
		} else if x < branchBoundary {
			xNew = branchBoundary
		} else {
			xNew = -1.0
		}
		yNew := y - m*(x+1.0) + m*sig
		x = xNew
		y = yNew
		trace[t] = x
		if x >= thr && xPrev < thr {
			spikes++
		}
	}
	if n > 0 {
		trace[n] = x
		trace[n+1] = y
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
