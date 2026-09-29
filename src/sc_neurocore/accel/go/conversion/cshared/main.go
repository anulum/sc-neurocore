// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go pinned owned dense IF C boundary

// Package main exposes ABI-one owned dense IF replay without trace copies.
package main

// #include "abi.h"
import "C"

import (
	"github.com/anulum/sc-neurocore/accel/conversion"
	"runtime"
	"runtime/cgo"
	"unsafe"
)

type owner struct {
	result *conversion.ReplayResult
	pins   runtime.Pinner
}

// pin retains every numeric allocation referenced by C views until the final supplying owner is released.
func (o *owner) pin() {
	groups := [][][]float64{{o.result.Output}, o.result.FinalState, o.result.StateTrace, o.result.SpikeTrace}
	for _, group := range groups {
		for _, values := range group {
			if len(values) > 0 {
				o.pins.Pin(&values[0])
			}
		}
	}
}

// sc_if_abi_version reports the exact ownership/buffer ABI version.
//
//export sc_if_abi_version
func sc_if_abi_version() C.uint32_t { return 1 }

// sc_if_replay publishes one pinned result owner on success. Returns0 success,
// -1 invalid input,-2 resource refusal,-3 arithmetic overflow,-4 internal panic.
// Failure leaves the exclusive live aligned result slot unchanged. Input pointers
// denote complete live aligned arrays, unchanged during call; accessibility is
// the caller's contract and cannot be established by span checks.
//
//export sc_if_replay
func sc_if_replay(request *C.sc_if_request, result *unsafe.Pointer) (status C.int32_t) {
	status = -4
	defer func() {
		if recover() != nil {
			status = -4
		}
	}()
	if !span(unsafe.Pointer(request), 1, uint64(C.sizeof_sc_if_request), uint64(unsafe.Alignof(C.sc_if_request{}))) || !span(unsafe.Pointer(result), 1, uint64(unsafe.Sizeof(unsafe.Pointer(nil))), uint64(unsafe.Alignof(unsafe.Pointer(nil)))) {
		return -1
	}
	value, err := execute(request)
	if err != nil {
		switch err {
		case conversion.ErrInvalidInput:
			return -1
		case conversion.ErrResourceLimit:
			return -2
		case conversion.ErrOverflow:
			return -3
		}
		return -4
	}
	retained := &owner{result: value}
	retained.pin()
	handle := cgo.NewHandle(retained)
	slot := C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	*(*C.uintptr_t)(slot) = C.uintptr_t(handle)
	*result = slot
	return 0
}

// sc_if_buffer borrows a pinned numeric buffer, valid until exactly-once free.
// Kind0 output/index0,1 final,2 state trace,3 spike trace. Unknown kind/index
// returns-1 without modifying the exclusive live aligned view. Handle must be
// a live result from this library and cannot be concurrently released. Returned
// doubles may be mutated; result metadata and caller handle must stay unchanged.
//
//export sc_if_buffer
func sc_if_buffer(handle unsafe.Pointer, kind C.uint32_t, index C.size_t, view *C.sc_if_view) (status C.int32_t) {
	status = -1
	defer func() {
		if recover() != nil {
			status = -1
		}
	}()
	if !span(handle, 1, uint64(unsafe.Sizeof(C.uintptr_t(0))), uint64(unsafe.Alignof(C.uintptr_t(0)))) || !span(unsafe.Pointer(view), 1, uint64(C.sizeof_sc_if_view), uint64(unsafe.Alignof(C.sc_if_view{}))) {
		return -1
	}
	retained := cgo.Handle(*(*C.uintptr_t)(handle)).Value().(*owner)
	var values []float64
	switch kind {
	case 0:
		if index != 0 {
			return -1
		}
		values = retained.result.Output
	case 1:
		if uint64(index) >= uint64(len(retained.result.FinalState)) {
			return -1
		}
		values = retained.result.FinalState[int(index)]
	case 2:
		if uint64(index) >= uint64(len(retained.result.StateTrace)) {
			return -1
		}
		values = retained.result.StateTrace[int(index)]
	case 3:
		if uint64(index) >= uint64(len(retained.result.SpikeTrace)) {
			return -1
		}
		values = retained.result.SpikeTrace[int(index)]
	default:
		return -1
	}
	var pointer *C.double
	if len(values) > 0 {
		pointer = (*C.double)(unsafe.Pointer(&values[0]))
	}
	view.data = pointer
	view.len = C.size_t(len(values))
	return 0
}

// sc_if_free unpins buffers and releases the opaque C slot exactly once. Null is
// a no-op. Nonnull must be a live unmodified handle from this library; every view
// must have expired and no replay/view access may run concurrently with release.
//
//export sc_if_free
func sc_if_free(pointer unsafe.Pointer) {
	if pointer == nil {
		return
	}
	handle := cgo.Handle(*(*C.uintptr_t)(pointer))
	retained := handle.Value().(*owner)
	retained.pins.Unpin()
	handle.Delete()
	C.free(pointer)
}

func main() {}
