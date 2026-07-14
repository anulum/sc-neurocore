// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for configurable threshold-linear rate

package services

import (
	"errors"
	"math"
)

var (
	ErrThresholdLinearRateInvalidInput    = errors.New("threshold-linear rate current must be finite")
	ErrThresholdLinearRateInvalidState    = errors.New("threshold-linear rate state and parameters must be finite with non-negative rate and gain")
	ErrThresholdLinearRateNonFiniteOutput = errors.New("threshold-linear rate output must remain finite and non-negative")
	ErrThresholdLinearRateInvalidSteps    = errors.New("threshold-linear rate step count must be non-negative")
)

// ThresholdLinearRateNeuronState stores the latest output and fixed transfer parameters.
type ThresholdLinearRateNeuronState struct {
	R     float64
	Theta float64
	Gain  float64
}

// NewThresholdLinearRateNeuron creates the maintained factory-default transfer.
func NewThresholdLinearRateNeuron() *ThresholdLinearRateNeuronState {
	return &ThresholdLinearRateNeuronState{R: 0.0, Theta: 0.0, Gain: 1.0}
}

// Valid reports whether the complete mutable numeric contract is well formed.
func (s ThresholdLinearRateNeuronState) Valid() bool {
	return thresholdLinearRateFinite(s.R, s.Theta, s.Gain) && s.R >= 0.0 && s.Gain >= 0.0
}

// Step evaluates one input. Invalid inputs leave the cached output unchanged.
func (s *ThresholdLinearRateNeuronState) Step(current float64) (float64, error) {
	if !thresholdLinearRateFinite(current) {
		return s.R, ErrThresholdLinearRateInvalidInput
	}
	if !s.Valid() {
		return s.R, ErrThresholdLinearRateInvalidState
	}
	drive := current - s.Theta
	if drive < 0.0 {
		drive = 0.0
	}
	nextR := s.Gain * drive
	if !thresholdLinearRateFinite(nextR) || nextR < 0.0 {
		return s.R, ErrThresholdLinearRateNonFiniteOutput
	}
	s.R = nextR
	return nextR, nil
}

// Reset clears the cached rate without changing threshold or gain.
func (s *ThresholdLinearRateNeuronState) Reset() {
	s.R = 0.0
}

// SimulateThresholdLinearRateTrace evaluates a configurable constant-input batch.
// The input state is copied so every error is atomic to caller-owned state.
func SimulateThresholdLinearRateTrace(
	initial ThresholdLinearRateNeuronState,
	nSteps int,
	current float64,
) ([]float64, ThresholdLinearRateNeuronState, error) {
	if nSteps < 0 {
		return nil, initial, ErrThresholdLinearRateInvalidSteps
	}
	state := initial
	if !state.Valid() || !thresholdLinearRateFinite(current) {
		return nil, initial, ErrThresholdLinearRateInvalidState
	}
	trace := make([]float64, nSteps)
	for index := range trace {
		value, err := state.Step(current)
		if err != nil {
			return nil, initial, err
		}
		trace[index] = value
	}
	return trace, state, nil
}

// SimulateThresholdLinearRateNeuron preserves the historical default helper.
// Continuous rates are not binary spikes, so the second return is always zero.
func SimulateThresholdLinearRateNeuron(nSteps int, current float64) ([]float64, int) {
	trace, _, err := SimulateThresholdLinearRateTrace(*NewThresholdLinearRateNeuron(), nSteps, current)
	if err != nil {
		panic(err)
	}
	return trace, 0
}

func thresholdLinearRateFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}
