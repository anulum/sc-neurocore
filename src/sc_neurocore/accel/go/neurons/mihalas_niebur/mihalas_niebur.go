// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Mihalas-Niebur 2009 generalised IF (parity with mihalas_niebur.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libmihalasniebur.so mihalas_niebur.go`) that
// the Python dispatcher loads via ctypes.
//
// Parity contract: `mihalas_niebur_simulate_c` reproduces
// `sc_neurocore.neurons.models.mihalas_niebur.MihalasNieburNeuron.simulate`. The
// Mihalas-Niebur right-hand side is purely linear (no transcendental functions),
// so every RK4 stage is exact arithmetic and the trace, spike count and final
// (v, theta, i1, i2) state are bit-identical to the NumPy reference.
//
// Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

// mihalas_niebur_simulate_c runs n RK4 steps under a constant input. The caller
// allocates a trace buffer of length n+4: indices [0, n) receive the v trace
// (post-reset on firing steps), and indices n, n+1, n+2, n+3 receive the final
// v, theta, i1, i2. Returns the spike count.
//
//export mihalas_niebur_simulate_c
func mihalas_niebur_simulate_c(
	v0, theta0, i1_0, i2_0, vRest, vReset, thetaReset, thetaInf C.double,
	tauV, tauTheta, tau1, tau2, a, b, r1, r2, dt C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+4)
	v := float64(v0)
	theta := float64(theta0)
	i1 := float64(i1_0)
	i2 := float64(i2_0)
	pVRest := float64(vRest)
	pVReset := float64(vReset)
	pThetaReset := float64(thetaReset)
	pThetaInf := float64(thetaInf)
	pTauV := float64(tauV)
	pTauTheta := float64(tauTheta)
	pTau1 := float64(tau1)
	pTau2 := float64(tau2)
	pa := float64(a)
	pb := float64(b)
	pr1 := float64(r1)
	pr2 := float64(r2)
	pdt := float64(dt)
	cur := float64(current)
	halfDt := 0.5 * pdt
	deriv := func(vv, th, j1, j2 float64) (float64, float64, float64, float64) {
		return (-(vv - pVRest) + j1 + j2 + cur) / pTauV,
			(pThetaInf - th + pa*(vv-pVRest)) / pTauTheta,
			-j1 / pTau1,
			-j2 / pTau2
	}
	var spikes int64
	for t := 0; t < n; t++ {
		k1v, k1t, k1a, k1b := deriv(v, theta, i1, i2)
		k2v, k2t, k2a, k2b := deriv(v+halfDt*k1v, theta+halfDt*k1t, i1+halfDt*k1a, i2+halfDt*k1b)
		k3v, k3t, k3a, k3b := deriv(v+halfDt*k2v, theta+halfDt*k2t, i1+halfDt*k2a, i2+halfDt*k2b)
		k4v, k4t, k4a, k4b := deriv(v+pdt*k3v, theta+pdt*k3t, i1+pdt*k3a, i2+pdt*k3b)
		v = v + pdt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
		theta = theta + pdt*(k1t+2.0*k2t+2.0*k3t+k4t)/6.0
		i1 = i1 + pdt*(k1a+2.0*k2a+2.0*k3a+k4a)/6.0
		i2 = i2 + pdt*(k1b+2.0*k2b+2.0*k3b+k4b)/6.0
		if v >= theta {
			v = pVReset + pb*(v-pVRest)
			if pThetaReset > theta {
				theta = pThetaReset
			}
			i1 += pr1
			i2 += pr2
			spikes++
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = theta
	trace[n+2] = i1
	trace[n+3] = i2
	return C.longlong(spikes)
}

func main() {} // required for c-shared
