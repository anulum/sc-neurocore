// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go N-step simulator for the Wilson-Cowan 1972 E/I model

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libwilson_cowan.so`) that the
// Python dispatcher loads via ctypes.
//
// Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// sigmoid — published Wilson-Cowan 1972 two-term form:
//
//	S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
//
// The subtracted baseline makes S(0) = 0 exactly.
func sigmoid(a, theta, x float64) float64 {
	baseline := 1.0 / (1.0 + math.Exp(a*theta))
	return 1.0/(1.0+math.Exp(-a*(x-theta))) - baseline
}

// wilson_cowan_simulate_c — C-ABI entry point.
//
//export wilson_cowan_simulate_c
func wilson_cowan_simulate_c(
	n C.int,
	eInit, iInit C.double,
	wEE, wEI, wIE, wII C.double,
	tauE, tauI C.double,
	a, theta, dt C.double,
	extPtr unsafe.Pointer,
	eOut, iOut unsafe.Pointer,
	eFinalOut, iFinalOut *C.double,
) C.int {
	N := int(n)
	ext := unsafe.Slice((*C.double)(extPtr), N)
	eo := unsafe.Slice((*C.double)(eOut), N)
	io := unsafe.Slice((*C.double)(iOut), N)

	e := float64(eInit)
	i := float64(iInit)
	af := float64(a)
	θ := float64(theta)

	for t := 0; t < N; t++ {
		sE := sigmoid(af, θ, float64(wEE)*e-float64(wEI)*i+float64(ext[t]))
		sI := sigmoid(af, θ, float64(wIE)*e-float64(wII)*i)
		e += (-e + sE) / float64(tauE) * float64(dt)
		i += (-i + sI) / float64(tauI) * float64(dt)
		eo[t] = C.double(e)
		io[t] = C.double(i)
	}
	*eFinalOut = C.double(e)
	*iFinalOut = C.double(i)
	return 0
}

func main() {}
