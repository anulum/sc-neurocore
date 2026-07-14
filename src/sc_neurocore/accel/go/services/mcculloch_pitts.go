// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source-faithful McCulloch-Pitts Go service

package services

import "errors"

const maxMcCullochPittsCount int64 = (1 << 31) - 1

var (
	ErrMcCullochPittsInvalidCount = errors.New("mcculloch pitts excitatory count must be in signed 32-bit non-negative range")
	ErrMcCullochPittsInvalidFlag  = errors.New("mcculloch pitts inhibitory flag must be zero or one")
	ErrMcCullochPittsInvalidState = errors.New("mcculloch pitts theta must be a positive signed 32-bit integer")
	ErrMcCullochPittsLength       = errors.New("mcculloch pitts input lengths must match")
)

// McCullochPittsNeuronState holds the fixed source threshold and no cell state.
type McCullochPittsNeuronState struct {
	Theta int64
}

// NewMcCullochPittsNeuron creates the 1943 theta-one logical neuron.
func NewMcCullochPittsNeuron() *McCullochPittsNeuronState {
	return &McCullochPittsNeuronState{Theta: 1}
}

// NewMcCullochPittsNeuronWithTheta validates a custom afferent-count threshold.
func NewMcCullochPittsNeuronWithTheta(theta int64) (*McCullochPittsNeuronState, error) {
	state := &McCullochPittsNeuronState{Theta: theta}
	if !ValidateMcCullochPitts(state) {
		return nil, ErrMcCullochPittsInvalidState
	}
	return state, nil
}

// Step evaluates one preceding-instant afferent pattern.
func (s *McCullochPittsNeuronState) Step(excitatoryCount int64, inhibitoryActive bool) (int, error) {
	if !ValidateMcCullochPitts(s) {
		return 0, ErrMcCullochPittsInvalidState
	}
	if excitatoryCount < 0 || excitatoryCount > maxMcCullochPittsCount {
		return 0, ErrMcCullochPittsInvalidCount
	}
	if inhibitoryActive {
		return 0, nil
	}
	if excitatoryCount >= s.Theta {
		return 1, nil
	}
	return 0, nil
}

// ValidateMcCullochPitts enforces the positive signed-ABI-safe threshold.
func ValidateMcCullochPitts(s *McCullochPittsNeuronState) bool {
	return s != nil && s.Theta >= 1 && s.Theta <= maxMcCullochPittsCount
}

// EvaluateMcCullochPittsBatch validates every row before producing output.
func EvaluateMcCullochPittsBatch(theta int64, counts []int64, flags []uint8) ([]uint8, int, error) {
	state, err := NewMcCullochPittsNeuronWithTheta(theta)
	if err != nil {
		return nil, 0, err
	}
	if len(counts) != len(flags) {
		return nil, 0, ErrMcCullochPittsLength
	}

	inhibited := make([]bool, len(flags))
	for index, count := range counts {
		if count < 0 || count > maxMcCullochPittsCount {
			return nil, 0, ErrMcCullochPittsInvalidCount
		}
		if flags[index] > 1 {
			return nil, 0, ErrMcCullochPittsInvalidFlag
		}
		inhibited[index] = flags[index] == 1
	}

	events := make([]uint8, len(counts))
	eventCount := 0
	for index, count := range counts {
		event, stepErr := state.Step(count, inhibited[index])
		if stepErr != nil {
			return nil, 0, stepErr
		}
		events[index] = uint8(event)
		eventCount += event
	}
	return events, eventCount, nil
}
