// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-shared batch mirror for the Amari 1977 neural field

// Package main exports a complete periodic-grid Amari batch through a C ABI.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// amari_field_simulate_c advances steps rows of n-site input. Return codes
// identify invalid dimensions, configuration, state/input, kernel, or candidate.
//
//export amari_field_simulate_c
func amari_field_simulate_c(
	stepsC, nC C.int,
	tauC, aExcC, aWidthC, bInhC, bWidthC, dxC, dtC C.double,
	uInitPtr, currentsPtr, statesOutPtr, ratesOutPtr, finalOutPtr unsafe.Pointer,
) C.int {
	steps, n := int(stepsC), int(nC)
	if steps < 0 || n < 2 {
		return 1
	}
	tau, aExc, aWidth := float64(tauC), float64(aExcC), float64(aWidthC)
	bInh, bWidth, dx, dt := float64(bInhC), float64(bWidthC), float64(dxC), float64(dtC)
	parameters := []float64{tau, aExc, aWidth, bInh, bWidth, dx, dt}
	for _, value := range parameters {
		if !finite(value) {
			return 2
		}
	}
	if tau <= 0 || aExc < 0 || aWidth <= 0 || bInh < 0 || bWidth <= 0 || dx <= 0 || dt <= 0 {
		return 2
	}
	uInit := unsafe.Slice((*C.double)(uInitPtr), n)
	currents := unsafe.Slice((*C.double)(currentsPtr), steps*n)
	for _, value := range uInit {
		if !finite(float64(value)) {
			return 3
		}
	}
	for _, value := range currents {
		if !finite(float64(value)) {
			return 3
		}
	}
	kernel := make([]float64, n)
	for offset := range kernel {
		wrapped := offset
		if n-offset < wrapped {
			wrapped = n - offset
		}
		distance := float64(wrapped) * dx
		kernel[offset] = aExc*math.Exp(-aWidth*distance) - bInh*math.Exp(-bWidth*distance)
	}
	if !finite(kernel[0]) || !finite(kernel[n/2]) || kernel[0] <= 0 || kernel[n/2] >= 0 {
		return 4
	}
	u := make([]float64, n)
	for index, value := range uInit {
		u[index] = float64(value)
	}
	candidate := make([]float64, n)
	statesOut := unsafe.Slice((*C.double)(statesOutPtr), steps*n)
	ratesOut := unsafe.Slice((*C.double)(ratesOutPtr), steps)
	finalOut := unsafe.Slice((*C.double)(finalOutPtr), n)
	for step := 0; step < steps; step++ {
		for i := 0; i < n; i++ {
			convolution := 0.0
			for j, value := range u {
				if value > 0 {
					convolution += kernel[(i+n-j)%n]
				}
			}
			candidate[i] = u[i] + (-u[i]+convolution*dx+float64(currents[step*n+i]))*(dt/tau)
			if !finite(candidate[i]) {
				return 5
			}
		}
		copy(u, candidate)
		active := 0
		for i, value := range u {
			statesOut[step*n+i] = C.double(value)
			if value > 0 {
				active++
			}
		}
		ratesOut[step] = C.double(float64(active) / float64(n))
	}
	for index, value := range u {
		finalOut[index] = C.double(value)
	}
	return 0
}

func main() {}
