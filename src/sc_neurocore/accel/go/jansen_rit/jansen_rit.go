// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared batch mirror for Jansen–Rit

// Package main exports equation-(6) explicit-Euler dynamics through a C ABI.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validConfiguration(values [15]float64) bool {
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return values[6] > 0.0 && values[7] > 0.0 &&
		values[8] > 0.0 && values[9] > 0.0 && values[10] >= 0.0 &&
		values[11] > 0.0 && values[13] > 0.0 && values[14] > 0.0
}

func sigmoid(voltage, e0, v0, slope float64) float64 {
	exponent := slope * (v0 - voltage)
	if exponent >= 0.0 {
		expNeg := math.Exp(-exponent)
		return 2.0 * e0 * expNeg / (1.0 + expNeg)
	}
	return 2.0 * e0 / (1.0 + math.Exp(exponent))
}

// jansen_rit_simulate_c advances one complete drive batch.  Return codes mark
// invalid length, configuration, input, or candidate state respectively.
//
//export jansen_rit_simulate_c
func jansen_rit_simulate_c(
	n C.int,
	y0Init, y3Init, y1Init, y4Init, y2Init, y5Init C.double,
	aExc, bExc, aRate, bRate, c, e0, v0, slope, dt C.double,
	pExtPtr unsafe.Pointer,
	y0Out, y3Out, y1Out, y4Out, y2Out, y5Out, eegOut unsafe.Pointer,
	y0Final, y3Final, y1Final, y4Final, y2Final, y5Final *C.double,
) C.int {
	if n < 0 {
		return 1
	}
	configuration := [15]float64{
		float64(y0Init), float64(y3Init), float64(y1Init),
		float64(y4Init), float64(y2Init), float64(y5Init),
		float64(aExc), float64(bExc), float64(aRate), float64(bRate),
		float64(c), float64(e0), float64(v0), float64(slope), float64(dt),
	}
	if !validConfiguration(configuration) {
		return 2
	}
	steps := int(n)
	drive := unsafe.Slice((*C.double)(pExtPtr), steps)
	for _, value := range drive {
		if !finite(float64(value)) {
			return 3
		}
	}

	traces := make([][]float64, 7)
	for index := range traces {
		traces[index] = make([]float64, steps)
	}
	y0f, y3f, y1f := configuration[0], configuration[1], configuration[2]
	y4f, y2f, y5f := configuration[3], configuration[4], configuration[5]
	a, b := configuration[6], configuration[7]
	ar, br, c1 := configuration[8], configuration[9], configuration[10]
	e0f, v0f, rf, dtf := configuration[11], configuration[12], configuration[13], configuration[14]
	c2, c3, c4 := 0.8*c1, 0.25*c1, 0.25*c1
	for index := 0; index < steps; index++ {
		sPyramidal := sigmoid(y1f-y2f, e0f, v0f, rf)
		sExcitatory := sigmoid(c1*y0f, e0f, v0f, rf)
		sInhibitory := sigmoid(c3*y0f, e0f, v0f, rf)
		next := [6]float64{
			y0f + dtf*y3f,
			y3f + dtf*(a*ar*sPyramidal-2.0*ar*y3f-ar*ar*y0f),
			y1f + dtf*y4f,
			y4f + dtf*(a*ar*(float64(drive[index])+c2*sExcitatory)-2.0*ar*y4f-ar*ar*y1f),
			y2f + dtf*y5f,
			y5f + dtf*(b*br*c4*sInhibitory-2.0*br*y5f-br*br*y2f),
		}
		for _, value := range next {
			if !finite(value) {
				return 4
			}
		}
		y0f, y3f, y1f, y4f, y2f, y5f = next[0], next[1], next[2], next[3], next[4], next[5]
		traces[0][index], traces[1][index] = y0f, y3f
		traces[2][index], traces[3][index] = y1f, y4f
		traces[4][index], traces[5][index] = y2f, y5f
		traces[6][index] = y1f - y2f
	}

	outputPointers := [7]unsafe.Pointer{y0Out, y3Out, y1Out, y4Out, y2Out, y5Out, eegOut}
	for index, pointer := range outputPointers {
		output := unsafe.Slice((*C.double)(pointer), steps)
		for step, value := range traces[index] {
			output[step] = C.double(value)
		}
	}
	*y0Final, *y3Final, *y1Final = C.double(y0f), C.double(y3f), C.double(y1f)
	*y4Final, *y2Final, *y5Final = C.double(y4f), C.double(y2f), C.double(y5f)
	return 0
}

func main() {}
