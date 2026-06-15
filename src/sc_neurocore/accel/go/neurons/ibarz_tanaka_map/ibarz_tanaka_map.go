// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Ibarz-Tanaka piecewise-linear map (parity with ibarz_tanaka_map.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libibarz.so ibarz_tanaka_map.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `ibarz_tanaka_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron.simulate`
// bit-for-bit. The map is exact floating-point arithmetic (one division,
// additions, multiplications), so an identical operation order yields an
// identical trace, spike count and final state.
//
// Reference: Ibarz, B., Casado, J.M. & Sanjuán, M.A.F. (2011).
// Phys. Rep. 501:1-74.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

// ibarz_tanaka_map_simulate_c runs n steps of the piecewise-linear map under a
// constant input. The caller allocates a trace buffer of length n+2: indices
// [0, n) receive the x trace (already reset on spiking steps), index n the
// final x, index n+1 the final y. Returns the spike count.
//
//export ibarz_tanaka_map_simulate_c
func ibarz_tanaka_map_simulate_c(
	x0, y0, alpha, beta, mu, sigma, xThreshold, xReset C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	x := float64(x0)
	y := float64(y0)
	a := float64(alpha)
	b := float64(beta)
	m := float64(mu)
	sig := float64(sigma)
	thr := float64(xThreshold)
	reset := float64(xReset)
	cur := float64(current)
	var spikes int64
	for t := 0; t < n; t++ {
		var f float64
		if x <= 0.0 {
			f = a / (1.0 - x)
		} else {
			f = a + b*x
		}
		xNew := f + y + cur
		yNew := y - m*(x+1.0) + m*sig
		y = yNew
		if xNew >= thr {
			x = reset
			spikes++
		} else {
			x = xNew
		}
		trace[t] = x
	}
	if n > 0 {
		trace[n] = x
		trace[n+1] = y
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
