// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Courbage-Nekorkin-Vdovin 2007 map (parity with courage_nekorkin_map.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libcourage.so courage_nekorkin_map.go`) that
// the Python dispatcher loads via ctypes.
//
// Parity contract: `courage_nekorkin_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.courage_nekorkin_map.CourageNekorkinMapNeuron.simulate`
// bit-for-bit. The map is exact floating-point arithmetic (additions,
// multiplications, one division for the breakpoints, and a piecewise/Heaviside
// branch), so an identical operation order yields an identical trace,
// upward-crossing spike count, and final state.
//
// Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
// Chaos 17:043109 (arXiv:0712.2097), eqs. 3-5.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

// courage_nekorkin_map_simulate_c runs n steps of the map under a constant
// input. The caller allocates a trace buffer of length n+2: indices [0, n)
// receive the x trace, index n the final x, index n+1 the final y. Returns the
// upward-crossing spike count.
//
//export courage_nekorkin_map_simulate_c
func courage_nekorkin_map_simulate_c(
	x0, y0, m0, m1, a, d, j, beta, eps, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	x := float64(x0)
	y := float64(y0)
	pm0 := float64(m0)
	pm1 := float64(m1)
	pa := float64(a)
	pd := float64(d)
	pj := float64(j)
	pbeta := float64(beta)
	peps := float64(eps)
	thr := float64(xThreshold)
	cur := float64(current)
	am1 := pa * pm1
	den := pm0 + pm1
	jmin := am1 / den
	jmax := (pm0 + am1) / den
	var spikes int64
	for t := 0; t < n; t++ {
		xPrev := x
		var fx float64
		if x <= jmin {
			fx = -pm0 * x
		} else if x < jmax {
			fx = pm1 * (x - pa)
		} else {
			fx = -pm0 * (x - 1.0)
		}
		h := 0.0
		if (x - pd) >= 0.0 {
			h = 1.0
		}
		xNew := x + fx - y - pbeta*h + cur
		yNew := y + peps*(x-pj)
		x = xNew
		y = yNew
		trace[t] = x
		if x >= thr && xPrev < thr {
			spikes++
		}
	}
	trace[n] = x
	trace[n+1] = y
	return C.longlong(spikes)
}

func main() {} // required for c-shared
