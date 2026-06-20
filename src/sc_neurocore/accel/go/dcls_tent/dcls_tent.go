// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go batch DCLS-max Q8.8 tent kernel

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libdcls_tent.so`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `dcls_max_forward_batch_q88_c` produces bit-identical
// per-channel outputs, accumulators, overflow flags, active-tap counts and
// max gates as the Rust, Julia, Mojo and Python references. The kernel is
// exact integer Q8.8 arithmetic, so the parity tolerance is zero.
//
// Reference (match Python module):
//
//	Khalfaoui-Hassani, Pellegrini & Masquelier (2023), Dilated convolution
//	with learnable spacings, NeurIPS.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

const (
	fraction  = 8
	q88One    = int64(1) << fraction
	i16Max    = int64(32767)
	i16Min    = int64(-32768)
	i32Max    = int64(2147483647)
	i32Min    = int64(-2147483648)
	i16MaxQ16 = i16Max << fraction
	i16MinQ16 = i16Min << fraction
)

// tentGateQ88 mirrors the Python primary tent gate with truncating integer
// division.
func tentGateQ88(tapIndex, centreQ88, sigmaQ88 int64) int64 {
	delayQ88 := tapIndex << fraction
	distanceQ88 := delayQ88 - centreQ88
	if distanceQ88 < 0 {
		distanceQ88 = -distanceQ88
	}
	if distanceQ88 >= sigmaQ88 {
		return 0
	}
	gate := ((sigmaQ88 - distanceQ88) << fraction) / sigmaQ88
	if gate > q88One {
		return q88One
	}
	if gate < 0 {
		return 0
	}
	return gate
}

// saturateContraction reproduces the Q16.16/Q8.8 saturating output contract.
func saturateContraction(accumulator int64) (int64, int64, bool) {
	accQ16 := accumulator
	if accQ16 > i32Max {
		accQ16 = i32Max
	} else if accQ16 < i32Min {
		accQ16 = i32Min
	}
	accOverflow := accQ16 != accumulator
	if accumulator > i16MaxQ16 {
		return i16Max, accQ16, true
	}
	if accumulator < i16MinQ16 {
		return i16Min, accQ16, true
	}
	return accumulator >> fraction, accQ16, accOverflow
}

// dcls_max_forward_batch_q88_c — C-ABI entry point.
//
// Caller passes:
//
//	nChannels, nTaps          — batch shape
//	spikes (uint8*)           — length nChannels*nTaps, row-major per channel
//	weights (int16*)          — length nChannels*nTaps
//	centres, sigmas (int16*)  — length nChannels
//	outputs (int16*), accumulators (int32*), overflow (uint8*),
//	active (int64*), maxGates (int16*) — length nChannels (written)
//
// Returns 0 on success, 1 on a non-positive shape, 2 on a non-positive sigma.
//
//export dcls_max_forward_batch_q88_c
func dcls_max_forward_batch_q88_c(
	nChannels, nTaps C.int,
	spikesPtr, weightsPtr, centresPtr, sigmasPtr unsafe.Pointer,
	outputsPtr, accumulatorsPtr, overflowPtr, activePtr, maxGatesPtr unsafe.Pointer,
) C.int {
	nc := int(nChannels)
	nt := int(nTaps)
	if nc <= 0 || nt <= 0 {
		return 1
	}
	total := nc * nt
	spikes := unsafe.Slice((*C.uint8_t)(spikesPtr), total)
	weights := unsafe.Slice((*C.int16_t)(weightsPtr), total)
	centres := unsafe.Slice((*C.int16_t)(centresPtr), nc)
	sigmas := unsafe.Slice((*C.int16_t)(sigmasPtr), nc)
	outputs := unsafe.Slice((*C.int16_t)(outputsPtr), nc)
	accumulators := unsafe.Slice((*C.int32_t)(accumulatorsPtr), nc)
	overflow := unsafe.Slice((*C.uint8_t)(overflowPtr), nc)
	active := unsafe.Slice((*C.int64_t)(activePtr), nc)
	maxGates := unsafe.Slice((*C.int16_t)(maxGatesPtr), nc)

	for c := 0; c < nc; c++ {
		centre := int64(centres[c])
		sigma := int64(sigmas[c])
		if sigma <= 0 {
			return 2
		}
		base := c * nt
		var accumulator int64
		var activeCount int64
		var maxGate int64
		for t := 0; t < nt; t++ {
			if spikes[base+t] == 0 {
				continue
			}
			activeCount++
			gate := tentGateQ88(int64(t), centre, sigma)
			if gate > maxGate {
				maxGate = gate
			}
			accumulator += int64(weights[base+t]) * gate
		}
		out, acc, overflowed := saturateContraction(accumulator)
		outputs[c] = C.int16_t(out)
		accumulators[c] = C.int32_t(acc)
		if overflowed {
			overflow[c] = 1
		} else {
			overflow[c] = 0
		}
		active[c] = C.int64_t(activeCount)
		maxGates[c] = C.int16_t(maxGate)
	}
	return 0
}

func main() {}
