// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go source-faithful Mihalas-Niebur batch kernel

// Package main exports the Mihalaş-Niebur equations 2.1–2.2 through a C ABI.
//
// Build with:
//
//	go build -buildmode=c-shared -o libmihalasniebur.so mihalas_niebur.go
//
// Rates are per millisecond and current-like quantities are divided by
// capacitance. The kernel matches the Python fixed-grid RK4 specialisation,
// including the published event reset.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

//export mihalas_niebur_simulate_c
func mihalas_niebur_simulate_c(
	v0, theta0, i1Initial, i2Initial, vRest, vReset, thetaReset, thetaInf C.double,
	leakRate, thresholdVoltageCoupling, thresholdDecayRate C.double,
	currentDecayRate1, currentDecayRate2, currentRetention1, currentRetention2 C.double,
	currentJump1, currentJump2, dt C.double,
	nSteps C.int, current C.double, tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	if n < 0 || tracePtr == nil {
		return -1
	}
	values := []float64{
		float64(v0), float64(theta0), float64(i1Initial), float64(i2Initial),
		float64(vRest), float64(vReset), float64(thetaReset), float64(thetaInf),
		float64(leakRate), float64(thresholdVoltageCoupling), float64(thresholdDecayRate),
		float64(currentDecayRate1), float64(currentDecayRate2),
		float64(currentRetention1), float64(currentRetention2),
		float64(currentJump1), float64(currentJump2), float64(dt), float64(current),
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return -1
		}
	}
	if leakRate <= 0 || thresholdDecayRate <= 0 || currentDecayRate1 <= 0 ||
		currentDecayRate2 <= 0 || dt <= 0 || thetaReset <= vReset {
		return -1
	}

	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+4)
	v := float64(v0)
	theta := float64(theta0)
	i1 := float64(i1Initial)
	i2 := float64(i2Initial)
	halfDT := 0.5 * float64(dt)
	stepDT := float64(dt)
	drive := float64(current)
	derivatives := func(voltage, threshold, current1, current2 float64) [4]float64 {
		return [4]float64{
			drive + current1 + current2 - float64(leakRate)*(voltage-float64(vRest)),
			float64(thresholdVoltageCoupling)*(voltage-float64(vRest)) -
				float64(thresholdDecayRate)*(threshold-float64(thetaInf)),
			-float64(currentDecayRate1) * current1,
			-float64(currentDecayRate2) * current2,
		}
	}
	addScaled := func(state, slope [4]float64, scale float64) [4]float64 {
		return [4]float64{
			state[0] + scale*slope[0],
			state[1] + scale*slope[1],
			state[2] + scale*slope[2],
			state[3] + scale*slope[3],
		}
	}

	var events int64
	for index := 0; index < n; index++ {
		state := [4]float64{v, theta, i1, i2}
		k1 := derivatives(v, theta, i1, i2)
		s2 := addScaled(state, k1, halfDT)
		k2 := derivatives(s2[0], s2[1], s2[2], s2[3])
		s3 := addScaled(state, k2, halfDT)
		k3 := derivatives(s3[0], s3[1], s3[2], s3[3])
		s4 := addScaled(state, k3, stepDT)
		k4 := derivatives(s4[0], s4[1], s4[2], s4[3])
		v = state[0] + stepDT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
		theta = state[1] + stepDT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
		i1 = state[2] + stepDT*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
		i2 = state[3] + stepDT*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
		if v >= theta {
			i1 = float64(currentRetention1)*i1 + float64(currentJump1)
			i2 = float64(currentRetention2)*i2 + float64(currentJump2)
			v = float64(vReset)
			theta = math.Max(float64(thetaReset), theta)
			events++
		}
		if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(theta) ||
			math.IsInf(theta, 0) || math.IsNaN(i1) || math.IsInf(i1, 0) ||
			math.IsNaN(i2) || math.IsInf(i2, 0) {
			return -1
		}
		trace[index] = v
	}
	trace[n] = v
	trace[n+1] = theta
	trace[n+2] = i1
	trace[n+3] = i2
	return C.longlong(events)
}

func main() {}
