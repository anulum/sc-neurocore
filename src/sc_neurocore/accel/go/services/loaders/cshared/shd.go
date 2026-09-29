// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — N-MNIST C ABI

//go:build hdf5

package main

/*
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
typedef struct {
    double *events;
    size_t value_count;
    int64_t label;
} sc_shd_recording;
*/
import "C"

import (
	"github.com/anulum/sc-neurocore/accel/services/loaders"
	"unsafe"
)

// shd_read_c reads one HDF5 row into C-owned memory, returning 0 or -1.
// path must be a live NUL-terminated string; recording must identify a writable,
// initially zeroed struct with no outstanding allocation. index and maximumBytes
// must fit a Go int. The budget bounds event bytes, not total HDF5 memory.
// Refusal leaves the destination unchanged. No Go pointer crosses the ABI.
// Release successful results exactly once with shd_free_c before reuse.
//
//export shd_read_c
func shd_read_c(path *C.char, index C.size_t, maximumBytes C.size_t, recording *C.sc_shd_recording) C.int {
	maximum := uint64(^uint(0) >> 1)
	if path == nil || recording == nil || uint64(index) > maximum || uint64(maximumBytes) > maximum {
		return -1
	}
	if recording.events != nil || recording.value_count != 0 || recording.label != 0 {
		return -1
	}
	sample, err := loaders.ReadSHDRecording(C.GoString(path), int(index), int(maximumBytes))
	if err != nil {
		return -1
	}
	var allocation unsafe.Pointer
	if len(sample.Events) != 0 {
		allocation = C.calloc(C.size_t(len(sample.Events)), 8)
		if allocation == nil {
			return -1
		}
		copy(unsafe.Slice((*float64)(allocation), len(sample.Events)), sample.Events)
	}
	recording.events = (*C.double)(allocation)
	recording.value_count = C.size_t(len(sample.Events))
	recording.label = C.int64_t(sample.Label)
	return 0
}

// shd_free_c releases a successful result and resets its fields for reuse.
// The pointer must be null or a live struct produced by shd_read_c. Repeated
// release of the reset struct is safe; copying an owning struct is forbidden.
//
//export shd_free_c
func shd_free_c(recording *C.sc_shd_recording) {
	if recording == nil {
		return
	}
	C.free(unsafe.Pointer(recording.events))
	recording.events = nil
	recording.value_count = 0
	recording.label = 0
}
