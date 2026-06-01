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

func wilsonCowanDerivatives(e, i, ext, wEE, wEI, wIE, wII, tauE, tauI, a, theta float64) (float64, float64) {
	sE := sigmoid(a, theta, wEE*e-wEI*i+ext)
	sI := sigmoid(a, theta, wIE*e-wII*i)
	return (-e + sE) / tauE, (-i + sI) / tauI
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
	δt := float64(dt)
	wee := float64(wEE)
	wei := float64(wEI)
	wie := float64(wIE)
	wii := float64(wII)
	τe := float64(tauE)
	τi := float64(tauI)

	for t := 0; t < N; t++ {
		drive := float64(ext[t])
		k1E, k1I := wilsonCowanDerivatives(e, i, drive, wee, wei, wie, wii, τe, τi, af, θ)
		k2E, k2I := wilsonCowanDerivatives(e+0.5*δt*k1E, i+0.5*δt*k1I, drive, wee, wei, wie, wii, τe, τi, af, θ)
		k3E, k3I := wilsonCowanDerivatives(e+0.5*δt*k2E, i+0.5*δt*k2I, drive, wee, wei, wie, wii, τe, τi, af, θ)
		k4E, k4I := wilsonCowanDerivatives(e+δt*k3E, i+δt*k3I, drive, wee, wei, wie, wii, τe, τi, af, θ)
		e += δt * (k1E + 2.0*k2E + 2.0*k3E + k4E) / 6.0
		i += δt * (k1I + 2.0*k2I + 2.0*k3I + k4I) / 6.0
		eo[t] = C.double(e)
		io[t] = C.double(i)
	}
	*eFinalOut = C.double(e)
	*iFinalOut = C.double(i)
	return 0
}

func main() {}
