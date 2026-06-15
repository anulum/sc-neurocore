// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Medvedev 2005 1D spiking map (parity with medvedev_map.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libmedvedev.so medvedev_map.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `medvedev_map_simulate_c` reproduces
// `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron.simulate`
// bit-for-bit. The map is exact floating-point arithmetic (a multiply, an add,
// and a fold into [0, 1)). The fold is `math.Mod` adjusted to a non-negative
// remainder — i.e. the Euclidean remainder — which equals Python's `x % 1.0`
// and Rust's `rem_euclid(1.0)` bit-for-bit (Go's bare `math.Mod` is truncated
// and must be adjusted for negative inputs).
//
// Reference: Medvedev, G.S. (2005). Physica D 202:37-59.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// foldUnit returns the Euclidean remainder of v modulo 1.0, i.e. v in [0, 1).
func foldUnit(v float64) float64 {
	r := math.Mod(v, 1.0)
	if r < 0.0 {
		r += 1.0
	}
	return r
}

// medvedev_map_simulate_c runs n steps of the 1D map under a constant input.
// The caller allocates a trace buffer of length n+1: indices [0, n) receive the
// x trace (folded into [0, 1)), index n the final x. Returns the upward-crossing
// spike count.
//
//export medvedev_map_simulate_c
func medvedev_map_simulate_c(
	x0, alpha, beta, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+1)
	x := float64(x0)
	a := float64(alpha)
	b := float64(beta)
	thr := float64(xThreshold)
	cur := float64(current)
	var spikes int64
	for t := 0; t < n; t++ {
		xPrev := x
		if x < b {
			x = a*x + cur
		} else {
			x = a*(1.0-x) + cur
		}
		x = foldUnit(x)
		trace[t] = x
		if x >= thr && xPrev < thr {
			spikes++
		}
	}
	if n > 0 {
		trace[n] = x
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
