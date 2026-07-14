// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exact Go service for the Wu et al. 2021 IQIF recurrence

package services

import (
	"errors"
	"math"
)

const (
	iqifInt32Min = -1 << 31
	iqifInt32Max = 1<<31 - 1
)

// IntegerQIFNeuronState holds the complete signed-integer soma contract.
type IntegerQIFNeuronState struct {
	V          int64
	VRest      int64
	VThreshold int64
	VReset     int64
	A          int64
	B          int64
	VMax       int64
	VMin       int64
}

// NewIntegerQIFNeuron creates the source tutorial's default neuron.
func NewIntegerQIFNeuron() *IntegerQIFNeuronState {
	return &IntegerQIFNeuronState{
		V:          128,
		VRest:      128,
		VThreshold: 200,
		VReset:     128,
		A:          1,
		B:          1,
		VMax:       255,
		VMin:       0,
	}
}

func iqifInt32(value int64) bool {
	return iqifInt32Min <= value && value <= iqifInt32Max
}

// Valid reports whether every field satisfies the source/public contract.
func (s IntegerQIFNeuronState) Valid() bool {
	values := [...]int64{s.V, s.VRest, s.VThreshold, s.VReset, s.A, s.B, s.VMax, s.VMin}
	for _, value := range values {
		if !iqifInt32(value) {
			return false
		}
	}
	return s.A >= 0 && s.B >= 0 && s.A+s.B > 0 &&
		s.VMin < s.VRest && s.VRest < s.VThreshold && s.VThreshold < s.VMax &&
		s.VMin <= s.VReset && s.VReset <= s.VMax &&
		s.VMin <= s.V && s.V <= s.VMax
}

// BranchPoint returns trunc((B*VThreshold + A*VRest)/(A+B)).
func (s IntegerQIFNeuronState) BranchPoint() int64 {
	// Go integer division truncates toward zero, matching the pinned C++ source.
	return (s.B*s.VThreshold + s.A*s.VRest) / (s.A + s.B)
}

// Step advances the exact Q0.3 recurrence. Invalid work does not mutate state.
func (s *IntegerQIFNeuronState) Step(current int64) (int, error) {
	if !s.Valid() || !iqifInt32(current) {
		return 0, ErrIQIFInvalidContract
	}
	force := s.B * (s.V - s.VThreshold)
	if s.V < s.BranchPoint() {
		force = s.A * (s.VRest - s.V)
	}
	candidate := s.V + (force >> 3) + current
	if candidate > s.VMax {
		s.V = s.VReset
		return 1, nil
	}
	s.V = candidate
	if s.V < s.VMin {
		s.V = s.VMin
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *IntegerQIFNeuronState) Reset() {
	s.V = s.VRest
}

// SimulateIntegerQIFNeuron preserves the historical Float64 wrapper while
// rejecting every value that cannot represent the exact integer current.
func SimulateIntegerQIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	if math.IsNaN(iExt) || math.IsInf(iExt, 0) || math.Trunc(iExt) != iExt ||
		iExt < iqifInt32Min || iExt > iqifInt32Max {
		panic(ErrIQIFInvalidContract)
	}
	trace, spikes, _, err := SimulateIQIFTrace(*NewIntegerQIFNeuron(), nSteps, int64(iExt))
	if err != nil {
		panic(err)
	}
	transport := make([]float64, len(trace))
	for index, value := range trace {
		transport[index] = float64(value)
	}
	return transport, spikes
}

// SimulateIQIFTrace runs one complete contract and returns post-step state.
func SimulateIQIFTrace(
	initial IntegerQIFNeuronState,
	nSteps int,
	current int64,
) ([]int64, int, IntegerQIFNeuronState, error) {
	if nSteps < 0 || !initial.Valid() || !iqifInt32(current) {
		return nil, 0, initial, ErrIQIFInvalidContract
	}
	state := initial
	trace := make([]int64, nSteps)
	spikes := 0
	for index := 0; index < nSteps; index++ {
		spike, err := state.Step(current)
		if err != nil {
			return nil, 0, initial, err
		}
		trace[index] = state.V
		spikes += spike
	}
	return trace, spikes, state, nil
}

// ErrIQIFInvalidContract denotes an invalid integer state, ordering or input.
var ErrIQIFInvalidContract = errors.New("IQIF requires ordered signed-int32 state and non-negative Q0.3 coefficients")
