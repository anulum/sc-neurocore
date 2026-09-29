// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go borrowed C replay request admission

package main

// #include "abi.h"
import "C"

import (
	"github.com/anulum/sc-neurocore/accel/conversion"
	"unsafe"
)

// span admits complete aligned spans in the signed host address domain; zero count borrows nothing.
func span(pointer unsafe.Pointer, count, width, alignment uint64) bool {
	if count == 0 {
		return true
	}
	maximum := uint64(^uint(0) >> 1)
	address := uint64(uintptr(pointer))
	return address != 0 && address%alignment == 0 && count <= maximum/width && address <= maximum-count*width
}

// doubles borrows aligned caller doubles without copying; zero count accepts null without making a Go null slice.
func doubles(pointer *C.double, count C.size_t) ([]float64, error) {
	if !span(unsafe.Pointer(pointer), uint64(count), 8, 8) {
		return nil, conversion.ErrInvalidInput
	}
	if count == 0 {
		return []float64{}, nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(pointer)), int(count)), nil
}

// execute admits complete coefficient storage before snapshots and delegates all dynamics to the canonical Go API.
func execute(request *C.sc_if_request) (*conversion.ReplayResult, error) {
	maximum := uint64(^uint(0) >> 1)
	if request.version != 1 || request.flags&^15 != 0 || request.layer_count == 0 || request.max_working_bytes == 0 || uint64(request.max_working_bytes) > maximum || uint64(request.steps) > maximum || uint64(request.batch) > maximum {
		return nil, conversion.ErrInvalidInput
	}
	if request.layer_count > request.max_working_bytes/16 {
		return nil, conversion.ErrResourceLimit
	}
	if !span(unsafe.Pointer(request.layers), uint64(request.layer_count), uint64(C.sizeof_sc_if_layer), uint64(unsafe.Alignof(C.sc_if_layer{}))) {
		return nil, conversion.ErrInvalidInput
	}
	specs := unsafe.Slice(request.layers, int(request.layer_count))
	coefficients := uint64(0)
	for _, spec := range specs {
		inputs, outputs := uint64(spec.inputs), uint64(spec.outputs)
		if inputs == 0 || outputs == 0 || inputs > maximum/outputs || (spec.bias_len != 0 && spec.bias_len != spec.outputs) {
			return nil, conversion.ErrInvalidInput
		}
		size := inputs*outputs + uint64(spec.bias_len)
		if size > maximum-coefficients {
			return nil, conversion.ErrResourceLimit
		}
		coefficients += size
	}
	if coefficients > uint64(request.max_working_bytes)/16 {
		return nil, conversion.ErrResourceLimit
	}
	parameters := make([]conversion.LayerParameters, len(specs))
	for index, spec := range specs {
		weights, err := doubles(spec.weights, spec.inputs*spec.outputs)
		if err != nil {
			return nil, err
		}
		var bias []float64
		if spec.bias_len != 0 {
			bias, err = doubles(spec.bias, spec.bias_len)
			if err != nil {
				return nil, err
			}
		}
		parameters[index] = conversion.LayerParameters{Inputs: int(spec.inputs), Outputs: int(spec.outputs), Weights: weights, Bias: bias, Threshold: float64(spec.threshold), InitialFraction: float64(spec.initial_fraction)}
	}
	mode := conversion.Spikes
	if request.flags&4 != 0 {
		mode = conversion.Linear
	}
	model, err := conversion.NewConvertedSNNFromParameters(parameters, mode, int(request.max_working_bytes))
	if err != nil {
		return nil, err
	}
	frames, err := doubles(request.frames, request.frames_len)
	if err != nil {
		return nil, err
	}
	options := conversion.ReplayOptions{Trace: request.flags&1 != 0, BinaryInputs: request.flags&2 != 0, MaxWorkingBytes: int(request.max_working_bytes)}
	if request.flags&8 != 0 {
		options.InitialState = make([][]float64, len(specs))
		for index, spec := range specs {
			options.InitialState[index], err = doubles(spec.initial, spec.initial_len)
			if err != nil {
				return nil, err
			}
		}
	}
	return model.Replay(frames, int(request.steps), int(request.batch), options)
}
