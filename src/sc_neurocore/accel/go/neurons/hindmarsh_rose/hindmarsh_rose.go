// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Hindmarsh-Rose RK4 simulator (parity with hindmarsh_rose.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libhr.so hindmarsh_rose.go`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `hindmarsh_rose_simulate_c` reproduces
// `sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron.simulate`
// bit-for-bit. The RK4 right-hand side is exact arithmetic (the square and cube
// are `x*x` and `(x*x)*x`; Go does not contract floating-point multiply-adds),
// so an identical operation order yields an identical trace, spike count and
// final state even though the bursting dynamics are chaotic.
//
// Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87-102.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func allFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func deriv(x, y, z, b, r, s, xRest, cur float64) (float64, float64, float64) {
	x2 := x * x
	x3 := x2 * x
	dx := y - x3 + b*x2 - z + cur
	dy := 1.0 - 5.0*x2 - y
	dz := r * (s*(x-xRest) - z)
	return dx, dy, dz
}

// hindmarsh_rose_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+3: indices [0, n) receive the x trace,
// index n the final x, n+1 the final y, n+2 the final z. Returns the
// upward-crossing spike count.
//
//export hindmarsh_rose_simulate_c
func hindmarsh_rose_simulate_c(
	x0, y0, z0, b, r, s, xRest, dt, xThreshold C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePtr == nil {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+3)
	x := float64(x0)
	y := float64(y0)
	z := float64(z0)
	bb := float64(b)
	rr := float64(r)
	ss := float64(s)
	xRestF := float64(xRest)
	d := float64(dt)
	thr := float64(xThreshold)
	cur := float64(current)
	if !allFinite(x, y, z, bb, rr, ss, xRestF, d, thr, cur) || rr <= 0 || ss <= 0 || d <= 0 {
		return -1
	}
	dt6 := d / 6.0
	var spikes int64
	for t := 0; t < n; t++ {
		xPrev := x
		k1x, k1y, k1z := deriv(x, y, z, bb, rr, ss, xRestF, cur)
		k2x, k2y, k2z := deriv(x+0.5*d*k1x, y+0.5*d*k1y, z+0.5*d*k1z, bb, rr, ss, xRestF, cur)
		k3x, k3y, k3z := deriv(x+0.5*d*k2x, y+0.5*d*k2y, z+0.5*d*k2z, bb, rr, ss, xRestF, cur)
		k4x, k4y, k4z := deriv(x+d*k3x, y+d*k3y, z+d*k3z, bb, rr, ss, xRestF, cur)
		if !allFinite(k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z) {
			return -1
		}
		nextX := x + dt6*(k1x+2.0*k2x+2.0*k3x+k4x)
		nextY := y + dt6*(k1y+2.0*k2y+2.0*k3y+k4y)
		nextZ := z + dt6*(k1z+2.0*k2z+2.0*k3z+k4z)
		if !allFinite(nextX, nextY, nextZ) {
			return -1
		}
		x, y, z = nextX, nextY, nextZ
		trace[t] = x
		if x >= thr && xPrev < thr {
			spikes++
		}
	}
	if n > 0 {
		trace[n] = x
		trace[n+1] = y
		trace[n+2] = z
	}
	return C.longlong(spikes)
}

func main() {} // required for c-shared
