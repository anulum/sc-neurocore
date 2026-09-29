// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Indexed SHD HDF5 recordings

//go:build hdf5

package loaders

/*
#cgo pkg-config: hdf5
#include <stdlib.h>
#include "shd_hdf5.h"
*/
import "C"

import (
	"errors"
	"strings"
	"sync"
	"unsafe"
)

var shdHDF5 sync.Mutex

// SHDRecording holds one HDF5 row as float64 x,y,polarity,t_ms events and its label.
// Auditory channels use y=0 and polarity=0. Timestamps retain float64 precision.
type SHDRecording struct {
	Events []float64
	Label  int64
}

// ReadSHDRecording reads one indexed recording, without loading the corpus.
// maximumBytes bounds the four-double event matrix; HDF5 input/reclaim buffers
// are additional transient memory. Both paired vectors and labels must have the
// same number of recordings. Numeric vectors are widened before seconds-to-ms
// conversion. Invalid files, indices, shapes or budgets refuse without fallback.
// The caller verifies dataset-manifest identity and event geometry separately.
// Calls are serialised because the system HDF5 library may lack thread safety.
func ReadSHDRecording(path string, index int, maximumBytes int) (SHDRecording, error) {
	if path == "" || strings.ContainsRune(path, '\x00') || index < 0 || maximumBytes < 0 {
		return SHDRecording{}, errors.New("invalid SHD path, index or event budget")
	}
	name := C.CString(path)
	defer C.free(unsafe.Pointer(name))
	shdHDF5.Lock()
	defer shdHDF5.Unlock()
	var sample C.sc_shd_sample
	result := C.sc_shd_read(name, C.size_t(index), C.size_t(maximumBytes), &sample)
	defer C.sc_shd_free(&sample)
	if result == -2 {
		return SHDRecording{}, errors.New("SHD recording exceeds the event budget")
	}
	if result != 0 {
		return SHDRecording{}, errors.New("invalid or unreadable indexed SHD recording")
	}
	n := int(sample.count)
	times := unsafe.Slice((*float64)(unsafe.Pointer(sample.times)), n)
	units := unsafe.Slice((*float64)(unsafe.Pointer(sample.units)), n)
	events := make([]float64, n*4)
	for i := range n {
		events[4*i] = units[i]
		events[4*i+3] = times[i] * 1000
	}
	return SHDRecording{Events: events, Label: int64(sample.label)}, nil
}
