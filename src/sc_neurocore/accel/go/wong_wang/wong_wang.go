// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go N-step simulator for the Wong-Wang 2006 decision unit

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libwong_wang.so`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `wong_wang_simulate_c` produces the same per-step
// `s1`, `s2`, `r1`, `r2` traces and final `(s1, s2)` as the Rust simulator
// and the Python primary when given matching xi noise (2 samples per
// step, order: [xi1_t0, xi2_t0, xi1_t1, xi2_t1, …]).
//
// Reference (match Python module):
//
//	Wong, K.-F. & Wang, X.-J. (2006). J. Neurosci. 26:1314–1328.
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

// phi reproduces the Python primary's φ(I) transfer function with the
// same singularity guard.
func phi(iSyn float64) float64 {
	x := phiA*iSyn - phiB
	if math.Abs(x) < 1e-6 {
		return 1.0 / phiD
	}
	return x / (1.0 - math.Exp(-phiD*x))
}

// clamp01 bounds into [0, 1] with branch-only ops (matches the Python
// scalar fix).
func clamp01(x float64) float64 {
	if x < 0.0 {
		return 0.0
	}
	if x > 1.0 {
		return 1.0
	}
	return x
}

func derivativeS1(
	s1, s2, stim1, xi1, tau, gm, jn, jx, i0f, sg float64,
) float64 {
	r1 := phi(jn*s1 - jx*s2 + i0f + stim1 + sg*xi1)
	return -s1/tau + (1.0-s1)*gm*r1
}

func derivativeS2(
	s1, s2, stim2, xi2, tau, gm, jn, jx, i0f, sg float64,
) float64 {
	r2 := phi(jn*s2 - jx*s1 + i0f + stim2 + sg*xi2)
	return -s2/tau + (1.0-s2)*gm*r2
}

// wong_wang_simulate_c — C-ABI entry point.
//
// Caller passes:
//
//	n       — number of steps (length of stim / trace arrays)
//	s1Init  — initial s1
//	s2Init  — initial s2
//	tauS, gamma, jN, jCross, i0, sigma, dt — model parameters
//	stim1Ptr, stim2Ptr — float64* length n
//	xiPtr              — float64* length 2n (xi1_t0, xi2_t0, xi1_t1, …)
//	s1Out, s2Out, r1Out, r2Out — float64* length n (written)
//	s1FinalOut, s2FinalOut     — float64* length 1 (written)
//
// Returns 0 on success.
//
//export wong_wang_simulate_c
func wong_wang_simulate_c(
	n C.int,
	s1Init, s2Init C.double,
	tauS, gamma, jN, jCross, i0, sigma, dt C.double,
	stim1Ptr, stim2Ptr, xiPtr unsafe.Pointer,
	s1Out, s2Out, r1Out, r2Out unsafe.Pointer,
	s1FinalOut, s2FinalOut *C.double,
) C.int {
	N := int(n)
	stim1 := unsafe.Slice((*C.double)(stim1Ptr), N)
	stim2 := unsafe.Slice((*C.double)(stim2Ptr), N)
	xi := unsafe.Slice((*C.double)(xiPtr), 2*N)
	s1out := unsafe.Slice((*C.double)(s1Out), N)
	s2out := unsafe.Slice((*C.double)(s2Out), N)
	r1out := unsafe.Slice((*C.double)(r1Out), N)
	r2out := unsafe.Slice((*C.double)(r2Out), N)

	s1 := float64(s1Init)
	s2 := float64(s2Init)
	tau := float64(tauS)
	gm := float64(gamma)
	jn := float64(jN)
	jx := float64(jCross)
	i0f := float64(i0)
	sg := float64(sigma)
	dtf := float64(dt)

	for t := 0; t < N; t++ {
		xi1 := float64(xi[2*t])
		xi2 := float64(xi[2*t+1])
		i1 := jn*s1 - jx*s2 + i0f + float64(stim1[t]) + sg*xi1
		i2 := jn*s2 - jx*s1 + i0f + float64(stim2[t]) + sg*xi2
		r1 := phi(i1)
		r2 := phi(i2)
		k1S1 := -s1/tau + (1.0-s1)*gm*r1
		k1S2 := -s2/tau + (1.0-s2)*gm*r2
		k2S1 := derivativeS1(s1+0.5*dtf*k1S1, s2+0.5*dtf*k1S2, float64(stim1[t]), xi1, tau, gm, jn, jx, i0f, sg)
		k2S2 := derivativeS2(s1+0.5*dtf*k1S1, s2+0.5*dtf*k1S2, float64(stim2[t]), xi2, tau, gm, jn, jx, i0f, sg)
		k3S1 := derivativeS1(s1+0.5*dtf*k2S1, s2+0.5*dtf*k2S2, float64(stim1[t]), xi1, tau, gm, jn, jx, i0f, sg)
		k3S2 := derivativeS2(s1+0.5*dtf*k2S1, s2+0.5*dtf*k2S2, float64(stim2[t]), xi2, tau, gm, jn, jx, i0f, sg)
		k4S1 := derivativeS1(s1+dtf*k3S1, s2+dtf*k3S2, float64(stim1[t]), xi1, tau, gm, jn, jx, i0f, sg)
		k4S2 := derivativeS2(s1+dtf*k3S1, s2+dtf*k3S2, float64(stim2[t]), xi2, tau, gm, jn, jx, i0f, sg)
		s1 += dtf * (k1S1 + 2.0*k2S1 + 2.0*k3S1 + k4S1) / 6.0
		s2 += dtf * (k1S2 + 2.0*k2S2 + 2.0*k3S2 + k4S2) / 6.0
		s1 = clamp01(s1)
		s2 = clamp01(s2)
		s1out[t] = C.double(s1)
		s2out[t] = C.double(s2)
		r1out[t] = C.double(r1)
		r2out[t] = C.double(r2)
	}
	*s1FinalOut = C.double(s1)
	*s2FinalOut = C.double(s2)
	return 0
}

func main() {}
