// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the Medvedev 2005 first-return map

// Package main exposes a checked C-ABI shared library for the calibrated
// slow-calcium first-return recurrence in models/medvedev_map.py.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

type firstReturnParameters struct {
	beta0              float64
	betaHC             float64
	betaSN             float64
	delta              float64
	decayT0            float64
	alphaT0            float64
	f0                 float64
	f1                 float64
	homoclinicExponent float64
	d                  float64
	inputGain          float64
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func (p firstReturnParameters) valid() bool {
	return finite(p.beta0) && finite(p.betaHC) && finite(p.betaSN) &&
		finite(p.delta) && finite(p.decayT0) && finite(p.alphaT0) &&
		finite(p.f0) && finite(p.f1) && finite(p.homoclinicExponent) &&
		finite(p.d) && finite(p.inputGain) &&
		0.0 < p.beta0 && p.beta0 < p.betaSN && p.betaSN < p.betaHC &&
		p.betaHC < p.delta && 0.0 < p.decayT0 && p.decayT0 < 1.0 &&
		0.0 < p.alphaT0 && p.alphaT0 < 1.0 && 0.0 <= p.f1 &&
		p.f1 < p.f0 && p.homoclinicExponent > 0.0 && p.d > 0.0 &&
		p.inputGain >= 0.0
}

func (p firstReturnParameters) u0() float64 {
	return p.beta0 / (p.delta - p.beta0)
}

func (p firstReturnParameters) uHC() float64 {
	return p.betaHC / (p.delta - p.betaHC)
}

func (p firstReturnParameters) uSN() float64 {
	return p.betaSN / (p.delta - p.betaSN)
}

func (p firstReturnParameters) candidate(u, current float64) (float64, bool) {
	var next float64
	if u <= p.u0() {
		next = p.decayT0*u + (1.0-p.decayT0)*p.f0 + p.inputGain*current
	} else if u <= p.uHC() {
		u1 := (1.0-p.alphaT0)*u + p.alphaT0*p.f0
		gap := p.betaHC - p.delta*u1/(1.0+u1)
		innerReturn := p.f1
		if gap > 0.0 {
			logArgument := p.d * gap
			if !finite(logArgument) || logArgument <= 0.0 {
				return 0.0, false
			}
			scale := math.Exp(p.homoclinicExponent * math.Log(logArgument))
			innerReturn = scale*(u1-p.f1) + p.f1
		}
		next = innerReturn + p.inputGain*current
	} else {
		next = p.uSN()
	}
	return next, finite(next)
}

// medvedev_map_simulate_c runs n checked first-return iterations. The caller
// allocates n+1 Float64 slots: [0,n) receives the u trace and [n] the final u.
// A negative result rejects malformed input or a non-finite candidate.
//
//export medvedev_map_simulate_c
func medvedev_map_simulate_c(
	u0, beta0, betaHC, betaSN, delta, decayT0, alphaT0, f0, f1,
	homoclinicExponent, d, inputGain C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePtr == nil {
		return -1
	}
	u := float64(u0)
	cur := float64(current)
	parameters := firstReturnParameters{
		beta0:              float64(beta0),
		betaHC:             float64(betaHC),
		betaSN:             float64(betaSN),
		delta:              float64(delta),
		decayT0:            float64(decayT0),
		alphaT0:            float64(alphaT0),
		f0:                 float64(f0),
		f1:                 float64(f1),
		homoclinicExponent: float64(homoclinicExponent),
		d:                  float64(d),
		inputGain:          float64(inputGain),
	}
	if !finite(u) || !finite(cur) || !parameters.valid() {
		return -1
	}

	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+1)
	var events int64
	for index := 0; index < n; index++ {
		if u <= parameters.uHC() {
			events++
		}
		candidate, ok := parameters.candidate(u, cur)
		if !ok {
			return -1
		}
		u = candidate
		trace[index] = u
	}
	trace[n] = u
	return C.longlong(events)
}

func main() {}
