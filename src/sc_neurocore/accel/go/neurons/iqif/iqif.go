// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for the exact Wu et al. IQIF recurrence

// Package main exposes services.IntegerQIFNeuronState as a C-shared library.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

//export iqif_simulate_c
func iqif_simulate_c(
	v C.int64_t,
	vRest C.int64_t,
	vThreshold C.int64_t,
	vReset C.int64_t,
	a C.int64_t,
	b C.int64_t,
	vMax C.int64_t,
	vMin C.int64_t,
	nSteps C.int64_t,
	current C.int64_t,
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
	initial := services.IntegerQIFNeuronState{
		V:          int64(v),
		VRest:      int64(vRest),
		VThreshold: int64(vThreshold),
		VReset:     int64(vReset),
		A:          int64(a),
		B:          int64(b),
		VMax:       int64(vMax),
		VMin:       int64(vMin),
	}
	trace, spikes, final, err := services.SimulateIQIFTrace(initial, n, int64(current))
	if err != nil {
		return -1
	}
	staged := make([]float64, n+1)
	for index, value := range trace {
		staged[index] = float64(value)
	}
	staged[n] = float64(final.V)
	destination := unsafe.Slice((*float64)(unsafe.Pointer(output)), n+1)
	copy(destination, staged)
	return C.int64_t(spikes)
}

func main() {}
