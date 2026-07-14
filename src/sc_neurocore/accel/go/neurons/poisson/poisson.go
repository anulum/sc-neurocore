// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C-ABI bridge for the seeded Poisson generator

// Package main exposes the complete services.PoissonNeuronState contract.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export poisson_simulate_c
func poisson_simulate_c(
	rateHz C.double,
	dtMs C.double,
	rngState C.uint16_t,
	nSteps C.int64_t,
	rateOverride C.double,
	output *C.double,
) C.int64_t {
	if nSteps < 0 || output == nil {
		return -1
	}
	n := int(nSteps)
	maxInt := int(^uint(0) >> 1)
	if C.int64_t(n) != nSteps || n > maxInt-1 {
		return -1
	}
	rng := uint16(rngState)
	if rng == 0 {
		rng = 0xACE1
	}
	state := services.PoissonNeuronState{
		RateHz:      float64(rateHz),
		DtMs:        float64(dtMs),
		RNGState:    rng,
		InitialSeed: rng,
	}
	events, final, err := services.SimulatePoissonTrace(
		state,
		n,
		float64(rateOverride),
	)
	if err != nil {
		return -1
	}
	staged := make([]float64, n+1)
	spikes := int64(0)
	for index, event := range events {
		staged[index] = float64(event)
		spikes += int64(event)
	}
	staged[n] = float64(final.RNGState)
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), len(staged))
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
