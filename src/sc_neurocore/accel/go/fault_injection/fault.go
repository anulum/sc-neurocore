// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go fault injection (parity with FaultInjector.inject)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libfault.so`) that the Python
// dispatcher loads via ctypes. It implements the 5 fault models from
// `sc_neurocore.fault_injection.FaultInjector.inject`:
//
//   1. BIT_FLIP        — XOR each bit with prob `ber`
//   2. STUCK_AT_0      — force each bit to 0 with prob `ber`
//   3. STUCK_AT_1      — force each bit to 1 with prob `ber`
//   4. DROPOUT         — alias for STUCK_AT_0
//   5. GAUSSIAN_NOISE  — add N(0, ber) then threshold at 0.5
//
// Each kernel operates in-place on an output buffer (mirror of the input)
// and writes the count of bits actually changed to *nAffectedOut.
//
// The RNG is `math/rand/v2 ChaCha8` (seeded per call from `seed`),
// which differs from numpy PCG64 — bitwise parity with Python is
// impossible. Statistical parity within 4σ of the Binomial(n, ber)
// mean is the honest contract, verified in the bench harness.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"math/rand/v2"
	"unsafe"
)

// wrap exposes a C pointer as a Go []byte slice without copy.
func wrapBytes(ptr *C.uint8_t, n int) []byte {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n)
}

func wrapU64(ptr *C.uint64_t, n int) []uint64 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*uint64)(unsafe.Pointer(ptr)), n)
}

// newRng returns a deterministic ChaCha8 generator seeded from a u64.
func newRng(seed uint64) *rand.Rand {
	var s [32]byte
	for i := 0; i < 8; i++ {
		s[i] = byte(seed >> (8 * i))
	}
	return rand.New(rand.NewChaCha8(s))
}

// ── Pure Go inject kernels (operate on Go slices) ──

func injectBitflip(out []byte, ber float64, seed uint64) uint64 {
	if ber <= 0.0 {
		return 0
	}
	r := newRng(seed)
	var n uint64
	for i := range out {
		if r.Float64() < ber {
			out[i] ^= 0x01
			n++
		}
	}
	return n
}

func injectStuckAt0(out []byte, ber float64, seed uint64) uint64 {
	if ber <= 0.0 {
		return 0
	}
	r := newRng(seed)
	var affected uint64
	for i := range out {
		if r.Float64() < ber {
			if out[i] != 0 {
				affected++
			}
			out[i] = 0
		}
	}
	return affected
}

func injectStuckAt1(out []byte, ber float64, seed uint64) uint64 {
	if ber <= 0.0 {
		return 0
	}
	r := newRng(seed)
	var affected uint64
	for i := range out {
		if r.Float64() < ber {
			if out[i] == 0 {
				affected++
			}
			out[i] = 1
		}
	}
	return affected
}

func injectGaussian(out []byte, ber float64, seed uint64) uint64 {
	if ber <= 0.0 {
		return 0
	}
	r := newRng(seed)
	var flipped uint64
	for i := range out {
		original := out[i]
		noisy := math.Min(math.Max(float64(original)+ber*r.NormFloat64(), 0.0), 1.0)
		var newBit byte
		if noisy > 0.5 {
			newBit = 1
		}
		if newBit != original {
			flipped++
		}
		out[i] = newBit
	}
	return flipped
}

// ── Exported C-ABI wrappers (called from Python via ctypes) ──

//export inject_bitflip_c
func inject_bitflip_c(
	bsPtr *C.uint8_t,
	n C.int64_t,
	ber C.double,
	seed C.uint64_t,
	nAffectedOut *C.uint64_t,
) {
	out := wrapBytes(bsPtr, int(n))
	*nAffectedOut = C.uint64_t(injectBitflip(out, float64(ber), uint64(seed)))
}

//export inject_stuck_at_0_c
func inject_stuck_at_0_c(
	bsPtr *C.uint8_t,
	n C.int64_t,
	ber C.double,
	seed C.uint64_t,
	nAffectedOut *C.uint64_t,
) {
	out := wrapBytes(bsPtr, int(n))
	*nAffectedOut = C.uint64_t(injectStuckAt0(out, float64(ber), uint64(seed)))
}

//export inject_stuck_at_1_c
func inject_stuck_at_1_c(
	bsPtr *C.uint8_t,
	n C.int64_t,
	ber C.double,
	seed C.uint64_t,
	nAffectedOut *C.uint64_t,
) {
	out := wrapBytes(bsPtr, int(n))
	*nAffectedOut = C.uint64_t(injectStuckAt1(out, float64(ber), uint64(seed)))
}

//export inject_dropout_c
func inject_dropout_c(
	bsPtr *C.uint8_t,
	n C.int64_t,
	ber C.double,
	seed C.uint64_t,
	nAffectedOut *C.uint64_t,
) {
	out := wrapBytes(bsPtr, int(n))
	*nAffectedOut = C.uint64_t(injectStuckAt0(out, float64(ber), uint64(seed)))
}

//export inject_gaussian_c
func inject_gaussian_c(
	bsPtr *C.uint8_t,
	n C.int64_t,
	ber C.double,
	seed C.uint64_t,
	nAffectedOut *C.uint64_t,
) {
	out := wrapBytes(bsPtr, int(n))
	*nAffectedOut = C.uint64_t(injectGaussian(out, float64(ber), uint64(seed)))
}

// suppress unused-import warning when wrapU64 is unused
var _ = wrapU64

func main() {} // required for c-shared
