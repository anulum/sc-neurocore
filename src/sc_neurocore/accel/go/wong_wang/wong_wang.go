// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go batch mirror for Wong-Wang 2006

// Package main exports the publication-faithful Euler/OU recurrence through a
// C ABI.  The caller supplies two standard-normal samples per macro-step.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

const (
	phiA = 270.0
	phiB = 108.0
	phiD = 0.154
)

func phi(iSyn float64) (float64, bool) {
	if math.IsNaN(iSyn) || math.IsInf(iSyn, 0) {
		return 0.0, false
	}
	x := phiA*iSyn - phiB
	scaled := -phiD * x
	response := 0.0
	if scaled > 700.0 {
		response = 0.0
	} else if math.Abs(x) < 1.0e-7 {
		response = 1.0 / phiD
	} else {
		response = x / -math.Expm1(scaled)
	}
	if math.IsNaN(response) || math.IsInf(response, 0) {
		return 0.0, false
	}
	return math.Max(0.0, response), true
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteGate(value float64) bool {
	return finite(value) && value >= 0.0 && value <= 1.0
}

func validConfiguration(
	s1, s2, noise1, noise2, tauS, tauAMPA, gamma, jN, jCross, i0, sigma, dt float64,
) bool {
	return finiteGate(s1) && finiteGate(s2) &&
		finite(noise1) && finite(noise2) &&
		finite(tauS) && tauS > 0.0 &&
		finite(tauAMPA) && tauAMPA > 0.0 &&
		finite(gamma) && gamma > 0.0 &&
		finite(jN) && jN >= 0.0 &&
		finite(jCross) && jCross >= 0.0 &&
		finite(i0) && finite(sigma) && sigma >= 0.0 &&
		finite(dt) && dt > 0.0
}

// wong_wang_simulate_c advances the reduced circuit and returns zero on
// success.  Non-zero codes report invalid length, configuration, input, or
// candidate state respectively.
//
//export wong_wang_simulate_c
func wong_wang_simulate_c(
	n C.int,
	s1Init, s2Init, noise1Init, noise2Init C.double,
	tauS, tauAMPA, gamma, jN, jCross, i0, sigma, dt C.double,
	stim1Ptr, stim2Ptr, xiPtr unsafe.Pointer,
	s1Out, s2Out, noise1Out, noise2Out, r1Out, r2Out unsafe.Pointer,
	s1FinalOut, s2FinalOut, noise1FinalOut, noise2FinalOut *C.double,
) C.int {
	if n < 0 {
		return 1
	}
	steps := int(n)
	s1 := float64(s1Init)
	s2 := float64(s2Init)
	noise1 := float64(noise1Init)
	noise2 := float64(noise2Init)
	tau := float64(tauS)
	tauNoise := float64(tauAMPA)
	gm := float64(gamma)
	jn := float64(jN)
	jx := float64(jCross)
	i0f := float64(i0)
	sg := float64(sigma)
	dtf := float64(dt)
	if !validConfiguration(s1, s2, noise1, noise2, tau, tauNoise, gm, jn, jx, i0f, sg, dtf) {
		return 2
	}

	stim1 := unsafe.Slice((*C.double)(stim1Ptr), steps)
	stim2 := unsafe.Slice((*C.double)(stim2Ptr), steps)
	xi := unsafe.Slice((*C.double)(xiPtr), 2*steps)
	s1Trace := unsafe.Slice((*C.double)(s1Out), steps)
	s2Trace := unsafe.Slice((*C.double)(s2Out), steps)
	noise1Trace := unsafe.Slice((*C.double)(noise1Out), steps)
	noise2Trace := unsafe.Slice((*C.double)(noise2Out), steps)
	r1Trace := unsafe.Slice((*C.double)(r1Out), steps)
	r2Trace := unsafe.Slice((*C.double)(r2Out), steps)

	noiseScale := math.Sqrt(dtf/tauNoise) * sg
	noiseDecay := dtf / tauNoise
	for step := 0; step < steps; step++ {
		drive1 := float64(stim1[step])
		drive2 := float64(stim2[step])
		xi1 := float64(xi[2*step])
		xi2 := float64(xi[2*step+1])
		if !finite(drive1) || !finite(drive2) || !finite(xi1) || !finite(xi2) {
			return 3
		}
		rate1, ok1 := phi(jn*s1 - jx*s2 + i0f + drive1 + noise1)
		rate2, ok2 := phi(jn*s2 - jx*s1 + i0f + drive2 + noise2)
		if !ok1 || !ok2 {
			return 4
		}
		nextS1 := s1 + dtf*(-s1/tau+(1.0-s1)*gm*rate1)
		nextS2 := s2 + dtf*(-s2/tau+(1.0-s2)*gm*rate2)
		nextNoise1 := noise1 - noiseDecay*noise1 + noiseScale*xi1
		nextNoise2 := noise2 - noiseDecay*noise2 + noiseScale*xi2
		if !finiteGate(nextS1) || !finiteGate(nextS2) || !finite(nextNoise1) || !finite(nextNoise2) {
			return 5
		}
		s1, s2, noise1, noise2 = nextS1, nextS2, nextNoise1, nextNoise2
		s1Trace[step] = C.double(s1)
		s2Trace[step] = C.double(s2)
		noise1Trace[step] = C.double(noise1)
		noise2Trace[step] = C.double(noise2)
		r1Trace[step] = C.double(rate1)
		r2Trace[step] = C.double(rate2)
	}
	*s1FinalOut = C.double(s1)
	*s2FinalOut = C.double(s2)
	*noise1FinalOut = C.double(noise1)
	*noise2FinalOut = C.double(noise2)
	return 0
}

func main() {}
