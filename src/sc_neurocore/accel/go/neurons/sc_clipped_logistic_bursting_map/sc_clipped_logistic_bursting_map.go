// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go retained clipped-logistic 2001 bursting map (parity with sc_clipped_logistic_bursting_map.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `sc_clipped_logistic_bursting_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.sc_clipped_logistic_bursting_map.retained clipped-logisticMapNeuron.simulate`
// bit-for-bit. The map is exact floating-point arithmetic (a*x*(1-x),
// additions, a clamp), so an identical operation order yields an identical
// trace, spike count, and final state.
//
// Project-defined recurrence retained without whole-model attribution.
package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

func clampUnit(v float64) float64 {
	if v < -2.0 {
		return -2.0
	}
	if v > 2.0 {
		return 2.0
	}
	return v
}

// sc_clipped_logistic_bursting_map_simulate_c runs n steps of the map under a constant input.
// The caller allocates a trace buffer of length n+2: indices [0, n) receive
// the x trace, index n the final x, index n+1 the final y. Returns the spike
// count.
//
//export sc_clipped_logistic_bursting_map_simulate_c
func sc_clipped_logistic_bursting_map_simulate_c(
	x0, y0, a, epsilon, sigma, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+2)
	x := float64(x0)
	y := float64(y0)
	aa := float64(a)
	eps := float64(epsilon)
	sig := float64(sigma)
	thr := float64(xThreshold)
	cur := float64(current)
	var spikes int64
	for t := 0; t < n; t++ {
		f := aa * x * (1.0 - x)
		xNew := f - y + cur
		yNew := y + eps*(x-sig)
		x = clampUnit(xNew)
		y = yNew
		trace[t] = x
		if x >= thr {
			spikes++
		}
	}
	trace[n] = x
	trace[n+1] = y
	return C.longlong(spikes)
}

func main() {} // required for c-shared
