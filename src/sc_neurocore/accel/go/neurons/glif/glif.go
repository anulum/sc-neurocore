// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Allen GLIF5 (parity with glif.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libglif.so glif.go`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `glif_simulate_c` reproduces
// `sc_neurocore.neurons.models.glif.GLIFNeuron.simulate`. The Allen GLIF5
// right-hand side is purely linear (no transcendental functions), so every RK4
// stage is exact arithmetic and the trace, spike count and final
// (v, theta, i_asc1, i_asc2) state are bit-identical to the NumPy reference.
//
// Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

// glif_simulate_c runs n RK4 steps under a constant input. The caller allocates a
// trace buffer of length n+4: indices [0, n) receive the v trace (post-reset on
// firing steps), and indices n, n+1, n+2, n+3 receive the final v, theta,
// i_asc1, i_asc2. Returns the spike count.
//
//export glif_simulate_c
func glif_simulate_c(
	v0, theta0, thetaInf, iAsc1_0, iAsc2_0, vRest, vReset, tauM, tauTheta C.double,
	tauAsc1, tauAsc2, aTheta, deltaTheta, rAsc1, rAsc2, resistance, dt C.double,
	nSteps C.int, current C.double,
	tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+4)
	v := float64(v0)
	theta := float64(theta0)
	a1 := float64(iAsc1_0)
	a2 := float64(iAsc2_0)
	pThetaInf := float64(thetaInf)
	pVRest := float64(vRest)
	pVReset := float64(vReset)
	pTauM := float64(tauM)
	pTauTheta := float64(tauTheta)
	pTauAsc1 := float64(tauAsc1)
	pTauAsc2 := float64(tauAsc2)
	pATheta := float64(aTheta)
	pDeltaTheta := float64(deltaTheta)
	pRAsc1 := float64(rAsc1)
	pRAsc2 := float64(rAsc2)
	pResistance := float64(resistance)
	pdt := float64(dt)
	cur := float64(current)
	halfDt := 0.5 * pdt
	deriv := func(vv, th, x1, x2 float64) (float64, float64, float64, float64) {
		return (-(vv - pVRest) + pResistance*cur + x1 + x2) / pTauM,
			(pThetaInf - th + pATheta*(vv-pVRest)) / pTauTheta,
			-x1 / pTauAsc1,
			-x2 / pTauAsc2
	}
	var spikes int64
	for t := 0; t < n; t++ {
		k1v, k1t, k1a, k1b := deriv(v, theta, a1, a2)
		k2v, k2t, k2a, k2b := deriv(v+halfDt*k1v, theta+halfDt*k1t, a1+halfDt*k1a, a2+halfDt*k1b)
		k3v, k3t, k3a, k3b := deriv(v+halfDt*k2v, theta+halfDt*k2t, a1+halfDt*k2a, a2+halfDt*k2b)
		k4v, k4t, k4a, k4b := deriv(v+pdt*k3v, theta+pdt*k3t, a1+pdt*k3a, a2+pdt*k3b)
		v = v + pdt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
		theta = theta + pdt*(k1t+2.0*k2t+2.0*k3t+k4t)/6.0
		a1 = a1 + pdt*(k1a+2.0*k2a+2.0*k3a+k4a)/6.0
		a2 = a2 + pdt*(k1b+2.0*k2b+2.0*k3b+k4b)/6.0
		if v >= theta {
			v = pVReset
			theta += pDeltaTheta
			a1 += pRAsc1
			a2 += pRAsc2
			spikes++
		}
		trace[t] = v
	}
	trace[n] = v
	trace[n+1] = theta
	trace[n+2] = a1
	trace[n+3] = a2
	return C.longlong(spikes)
}

func main() {} // required for c-shared
