// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for threshold_linear_rate

package services

import (
	"errors"
	"math"
)

var (
	ErrThresholdLinearRateInvalidInput    = errors.New("threshold linear rate input current must be finite")
	ErrThresholdLinearRateInvalidState    = errors.New("threshold linear rate state parameters must be finite with non-negative rate and gain")
	ErrThresholdLinearRateNonFiniteOutput = errors.New("threshold linear rate output must remain finite and non-negative")
)

// ThresholdLinearRateNeuronState holds the neuron state
type ThresholdLinearRateNeuronState struct {
	R     float64
	Theta float64
	Gain  float64
}

// NewThresholdLinearRateNeuron creates a new ThresholdLinearRateNeuron neuron with default parameters
func NewThresholdLinearRateNeuron() *ThresholdLinearRateNeuronState {
	return &ThresholdLinearRateNeuronState{
		R:     0.0,
		Theta: 0.0,
		Gain:  1.0,
	}
}

func thresholdLinearRateFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *ThresholdLinearRateNeuronState) Valid() bool {
	return thresholdLinearRateFinite(s.R, s.Theta, s.Gain) &&
		s.R >= 0.0 &&
		s.Gain >= 0.0
}

// Step advances the neuron by one timestep
func (s *ThresholdLinearRateNeuronState) Step(iExt float64) (float64, error) {
	if !thresholdLinearRateFinite(iExt) {
		return s.R, ErrThresholdLinearRateInvalidInput
	}
	if !s.Valid() {
		return s.R, ErrThresholdLinearRateInvalidState
	}

	drive := iExt - s.Theta
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

// SimulateThresholdLinearRateNeuron runs the neuron for n steps
func SimulateThresholdLinearRateNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewThresholdLinearRateNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.R
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
