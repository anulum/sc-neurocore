// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 neuron integrator ports

// Package main exposes C-ABI RK4 simulators for the priority neuron
// integrator paths.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

const izhSpikeThreshold = 30.0

func izhRHS(v, u, current float64) (float64, float64) {
	dv := 0.04*v*v + 5.0*v + 140.0 - u + current
	du := 0.02 * (0.2*v - u)
	return dv, du
}

//export simulate_izhikevich_rk4_c
func simulate_izhikevich_rk4_c(
	n C.int,
	dt C.double,
	currentPtr unsafe.Pointer,
	vOutPtr unsafe.Pointer,
	uOutPtr unsafe.Pointer,
	spikesOutPtr unsafe.Pointer,
) C.int {
	N := int(n)
	currents := unsafe.Slice((*C.double)(currentPtr), N)
	vOut := unsafe.Slice((*C.double)(vOutPtr), N)
	uOut := unsafe.Slice((*C.double)(uOutPtr), N)
	spikesOut := unsafe.Slice((*C.uint64_t)(spikesOutPtr), N)

	dtf := float64(dt)
	v := -65.0
	u := 0.2 * v
	nSpikes := 0

	for idx := 0; idx < N; idx++ {
		current := float64(currents[idx])
		k1v, k1u := izhRHS(v, u, current)
		k2v, k2u := izhRHS(v+0.5*dtf*k1v, u+0.5*dtf*k1u, current)
		k3v, k3u := izhRHS(v+0.5*dtf*k2v, u+0.5*dtf*k2u, current)
		k4v, k4u := izhRHS(v+dtf*k3v, u+dtf*k3u, current)

		v += (dtf / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v)
		u += (dtf / 6.0) * (k1u + 2.0*k2u + 2.0*k3u + k4u)

		if v >= izhSpikeThreshold {
			v = -65.0
			u += 8.0
			spikesOut[nSpikes] = C.uint64_t(idx)
			nSpikes++
		}

		vOut[idx] = C.double(v)
		uOut[idx] = C.double(u)
	}
	return C.int(nSpikes)
}

func clampExpArg(x float64) float64 {
	if x < -20.0 {
		return -20.0
	}
	if x > 20.0 {
		return 20.0
	}
	return x
}

func adexRHS(v, w, current float64) (float64, float64) {
	expArg := clampExpArg((v - -55.0) / 2.0)
	expTerm := 2.0 * math.Exp(expArg)
	dv := (-(v+65.0)+expTerm)/20.0 + (-w+current)/200.0
	dw := (0.5*(v+65.0) - w) / 100.0
	return dv, dw
}

//export simulate_adex_rk4_c
func simulate_adex_rk4_c(
	n C.int,
	dt C.double,
	currentPtr unsafe.Pointer,
	vOutPtr unsafe.Pointer,
	wOutPtr unsafe.Pointer,
	spikesOutPtr unsafe.Pointer,
) C.int {
	N := int(n)
	currents := unsafe.Slice((*C.double)(currentPtr), N)
	vOut := unsafe.Slice((*C.double)(vOutPtr), N)
	wOut := unsafe.Slice((*C.double)(wOutPtr), N)
	spikesOut := unsafe.Slice((*C.uint64_t)(spikesOutPtr), N)

	dtf := float64(dt)
	v := -65.0
	w := 0.0
	nSpikes := 0

	for idx := 0; idx < N; idx++ {
		current := float64(currents[idx])
		k1v, k1w := adexRHS(v, w, current)
		k2v, k2w := adexRHS(v+0.5*dtf*k1v, w+0.5*dtf*k1w, current)
		k3v, k3w := adexRHS(v+0.5*dtf*k2v, w+0.5*dtf*k2w, current)
		k4v, k4w := adexRHS(v+dtf*k3v, w+dtf*k3w, current)

		v += (dtf / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v)
		w += (dtf / 6.0) * (k1w + 2.0*k2w + 2.0*k3w + k4w)

		if v >= -50.0 {
			v = -68.0
			w += 7.0
			spikesOut[nSpikes] = C.uint64_t(idx)
			nSpikes++
		}

		vOut[idx] = C.double(v)
		wOut[idx] = C.double(w)
	}
	return C.int(nSpikes)
}

func alphaM(v float64) float64 {
	d := v + 40.0
	if math.Abs(d) < 1e-7 {
		return 1.0
	}
	return 0.1 * d / (1.0 - math.Exp(-d/10.0))
}

func betaM(v float64) float64 {
	return 4.0 * math.Exp(-(v+65.0)/18.0)
}

func alphaH(v float64) float64 {
	return 0.07 * math.Exp(-(v+65.0)/20.0)
}

func betaH(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-(v+35.0)/10.0))
}

func alphaN(v float64) float64 {
	d := v + 55.0
	if math.Abs(d) < 1e-7 {
		return 0.1
	}
	return 0.01 * d / (1.0 - math.Exp(-d/10.0))
}

func betaN(v float64) float64 {
	return 0.125 * math.Exp(-(v+65.0)/80.0)
}

func hhRHS(state [4]float64, current float64) [4]float64 {
	v := state[0]
	m := state[1]
	h := state[2]
	n := state[3]
	dm := alphaM(v)*(1.0-m) - betaM(v)*m
	dh := alphaH(v)*(1.0-h) - betaH(v)*h
	dn := alphaN(v)*(1.0-n) - betaN(v)*n
	iNa := 120.0 * math.Pow(m, 3.0) * h * (v - 50.0)
	iK := 36.0 * math.Pow(n, 4.0) * (v + 77.0)
	iL := 0.3 * (v + 54.4)
	dv := -iNa - iK - iL + current
	return [4]float64{dv, dm, dh, dn}
}

func addScaled(state [4]float64, deriv [4]float64, scale float64) [4]float64 {
	return [4]float64{
		state[0] + scale*deriv[0],
		state[1] + scale*deriv[1],
		state[2] + scale*deriv[2],
		state[3] + scale*deriv[3],
	}
}

//export simulate_hodgkin_huxley_rk4_c
func simulate_hodgkin_huxley_rk4_c(
	n C.int,
	dt C.double,
	currentPtr unsafe.Pointer,
	vOutPtr unsafe.Pointer,
	mOutPtr unsafe.Pointer,
	hOutPtr unsafe.Pointer,
	nOutPtr unsafe.Pointer,
	spikesOutPtr unsafe.Pointer,
) C.int {
	N := int(n)
	currents := unsafe.Slice((*C.double)(currentPtr), N)
	vOut := unsafe.Slice((*C.double)(vOutPtr), N)
	mOut := unsafe.Slice((*C.double)(mOutPtr), N)
	hOut := unsafe.Slice((*C.double)(hOutPtr), N)
	nOut := unsafe.Slice((*C.double)(nOutPtr), N)
	spikesOut := unsafe.Slice((*C.uint64_t)(spikesOutPtr), N)

	dtf := float64(dt)
	substeps := int(math.Round(1.0 / dtf))
	state := [4]float64{-65.0, 0.05, 0.6, 0.32}
	nSpikes := 0

	for idx := 0; idx < N; idx++ {
		vPrev := state[0]
		current := float64(currents[idx])
		for step := 0; step < substeps; step++ {
			k1 := hhRHS(state, current)
			k2 := hhRHS(addScaled(state, k1, 0.5*dtf), current)
			k3 := hhRHS(addScaled(state, k2, 0.5*dtf), current)
			k4 := hhRHS(addScaled(state, k3, dtf), current)
			for axis := 0; axis < 4; axis++ {
				state[axis] += (dtf / 6.0) * (k1[axis] + 2.0*k2[axis] + 2.0*k3[axis] + k4[axis])
			}
		}

		if state[0] >= 0.0 && vPrev < 0.0 {
			spikesOut[nSpikes] = C.uint64_t(idx)
			nSpikes++
		}

		vOut[idx] = C.double(state[0])
		mOut[idx] = C.double(state[1])
		hOut[idx] = C.double(state[2])
		nOut[idx] = C.double(state[3])
	}
	return C.int(nSpikes)
}

func main() {}
