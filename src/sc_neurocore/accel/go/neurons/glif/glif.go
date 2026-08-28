// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Teeter 2018 GLIF5 source model

// Package main exposes the canonical five-state GLIF5 constant-current batch.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func convolution(decayRate, forcingRate, dt float64) float64 {
	difference := decayRate - forcingRate
	scale := math.Max(1.0, math.Max(math.Abs(decayRate), math.Abs(forcingRate)))
	if math.Abs(difference) <= 1e-12*scale {
		return dt * math.Exp(-decayRate*dt)
	}
	return (math.Exp(-forcingRate*dt) - math.Exp(-decayRate*dt)) / difference
}

// glif_simulate_c runs n source-faithful GLIF5 steps. The caller owns n+6
// doubles: the voltage trace followed by the complete final state. A negative
// return reports invalid input or a non-finite candidate.
//
//export glif_simulate_c
func glif_simulate_c(
	v0, thetaSpike0, iAsc1_0, iAsc2_0, thetaVoltage0, refractoryRemaining0 C.double,
	eL, capacitance, resistance, thetaInf, bSpike, bVoltage, aVoltage C.double,
	kAsc1, kAsc2, fV, deltaV, deltaThetaSpike, fAsc1, fAsc2 C.double,
	deltaIAsc1, deltaIAsc2, refractoryPeriod, dt C.double,
	nSteps C.int, current C.double, tracePtr *C.double,
) C.longlong {
	n := int(nSteps)
	values := []float64{
		float64(v0), float64(thetaSpike0), float64(iAsc1_0), float64(iAsc2_0),
		float64(thetaVoltage0), float64(refractoryRemaining0), float64(eL),
		float64(capacitance), float64(resistance), float64(thetaInf), float64(bSpike),
		float64(bVoltage), float64(aVoltage), float64(kAsc1), float64(kAsc2),
		float64(fV), float64(deltaV), float64(deltaThetaSpike), float64(fAsc1),
		float64(fAsc2), float64(deltaIAsc1), float64(deltaIAsc2),
		float64(refractoryPeriod), float64(dt), float64(current),
	}
	if n < 0 || tracePtr == nil || !finite(values...) || values[7] <= 0.0 ||
		values[8] <= 0.0 || values[10] <= 0.0 || values[11] <= 0.0 ||
		values[13] <= 0.0 || values[14] <= 0.0 || values[23] <= 0.0 ||
		values[5] < 0.0 || values[22] < 0.0 {
		return -1
	}
	trace := unsafe.Slice((*float64)(unsafe.Pointer(tracePtr)), n+6)
	v, thetaSpike := values[0], values[1]
	iAsc1, iAsc2 := values[2], values[3]
	thetaVoltage, refractoryRemaining := values[4], values[5]
	membraneRate := 1.0 / (values[8] * values[7])
	membraneDecay := math.Exp(-membraneRate * values[23])
	spikeDecay := math.Exp(-values[10] * values[23])
	voltageDecay := math.Exp(-values[11] * values[23])
	asc1Decay := math.Exp(-values[13] * values[23])
	asc2Decay := math.Exp(-values[14] * values[23])
	voltageConvolution := convolution(values[11], membraneRate, values[23])
	var events int64
	for index := 0; index < n; index++ {
		if refractoryRemaining > 0.0 {
			refractoryRemaining = math.Max(0.0, refractoryRemaining-values[23])
			trace[index] = v
			continue
		}
		totalCurrent := values[24] + iAsc1 + iAsc2
		equilibriumOffset := values[8] * totalCurrent
		voltageOffset := v - values[6]
		nextOffset := equilibriumOffset + (voltageOffset-equilibriumOffset)*membraneDecay
		v = values[6] + nextOffset
		thetaSpike *= spikeDecay
		iAsc1 *= asc1Decay
		iAsc2 *= asc2Decay
		forcing := equilibriumOffset*(1.0-voltageDecay)/values[11] +
			(voltageOffset-equilibriumOffset)*voltageConvolution
		thetaVoltage = thetaVoltage*voltageDecay + values[12]*forcing
		if !finite(v, thetaSpike, iAsc1, iAsc2, thetaVoltage) {
			return -1
		}
		if v > values[9]+thetaSpike+thetaVoltage {
			v = values[6] + values[15]*(v-values[6]) - values[16]
			thetaSpike += values[17]
			iAsc1 = values[18]*iAsc1 + values[20]
			iAsc2 = values[19]*iAsc2 + values[21]
			refractoryRemaining = values[22]
			events++
		}
		if !finite(v, thetaSpike, iAsc1, iAsc2, thetaVoltage) {
			return -1
		}
		trace[index] = v
	}
	trace[n], trace[n+1], trace[n+2] = v, thetaSpike, iAsc1
	trace[n+3], trace[n+4], trace[n+5] = iAsc2, thetaVoltage, refractoryRemaining
	return C.longlong(events)
}

func main() {}
