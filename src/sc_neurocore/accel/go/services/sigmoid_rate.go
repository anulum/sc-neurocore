// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for sigmoid_rate

package services

import (
	"errors"
	"math"
)

// SigmoidRateNeuronState holds the neuron state
type SigmoidRateNeuronState struct {
	R     float64
	Tau   float64
	Beta  float64
	Theta float64
	Dt    float64
}

// NewSigmoidRateNeuron creates a new SigmoidRateNeuron neuron with default parameters
func NewSigmoidRateNeuron() *SigmoidRateNeuronState {
	return &SigmoidRateNeuronState{
		R:     0.0,
		Tau:   10.0,
		Beta:  1.0,
		Theta: 0.0,
		Dt:    0.1,
	}
}

// Valid reports whether the complete mutable rate contract is well formed.
func (s SigmoidRateNeuronState) Valid() bool {
	return sigmoidRateFinite(s.R) &&
		sigmoidRateFinite(s.Tau) &&
		sigmoidRateFinite(s.Beta) &&
		sigmoidRateFinite(s.Theta) &&
		sigmoidRateFinite(s.Dt) &&
		s.R >= 0.0 &&
		s.R <= 1.0 &&
		s.Tau > 0.0 &&
		s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *SigmoidRateNeuronState) Step(iExt float64) (float64, error) {
	if !sigmoidRateFinite(iExt) || !s.Valid() {
		return s.R, ErrSigmoidRateInvalidState
	}
	sigma, err := sigmoidRateTransfer(s.Beta, iExt, s.Theta)
	if err != nil {
		return s.R, err
	}
	nextR := sigmoidRateExactRelaxation(s.R, sigma, s.Dt, s.Tau)
	if !sigmoidRateFinite(nextR) || nextR < 0.0 || nextR > 1.0 {
		return s.R, ErrSigmoidRateNonFiniteUpdate
	}
	s.R = nextR
	return nextR, nil
}

func (s *SigmoidRateNeuronState) Reset() {
	s.R = 0.0
}

// SimulateSigmoidRateTrace runs a configurable constant-current batch.
// The input state is copied, so failures never mutate caller-owned state.
func SimulateSigmoidRateTrace(
	initial SigmoidRateNeuronState,
	nSteps int,
	iExt float64,
) ([]float64, SigmoidRateNeuronState, error) {
	if nSteps < 0 {
		return nil, initial, ErrSigmoidRateInvalidStepCount
	}
	state := initial
	if !state.Valid() || !sigmoidRateFinite(iExt) {
		return nil, initial, ErrSigmoidRateInvalidState
	}
	trace := make([]float64, nSteps)
	for index := range trace {
		value, err := state.Step(iExt)
		if err != nil {
			return nil, initial, err
		}
		trace[index] = value
	}
	return trace, state, nil
}

// SimulateSigmoidRateNeuron preserves the historical default-only helper.
// A rate model has no binary spikes, so the second return is always zero.
func SimulateSigmoidRateNeuron(nSteps int, iExt float64) ([]float64, int) {
	trace, _, err := SimulateSigmoidRateTrace(*NewSigmoidRateNeuron(), nSteps, iExt)
	if err != nil {
		panic(err)
	}
	return trace, 0
}

var (
	ErrSigmoidRateInvalidState     = errors.New("sigmoid-rate state/current must be finite and well-formed")
	ErrSigmoidRateNonFiniteUpdate  = errors.New("sigmoid-rate exact relaxation update became non-finite or left [0,1]")
	ErrSigmoidRateInvalidStepCount = errors.New("sigmoid-rate step count must be non-negative")
)

func sigmoidRateFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func sigmoidRateExactRelaxation(r float64, sigma float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*r + (1.0-decay)*sigma
}

func sigmoidRateTransfer(beta float64, current float64, theta float64) (float64, error) {
	z := beta * (current - theta)
	if math.IsInf(z, 0) {
		if z > 0.0 {
			return 1.0, nil
		}
		return 0.0, nil
	}
	if !sigmoidRateFinite(z) {
		return 0.0, ErrSigmoidRateNonFiniteUpdate
	}
	if z >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-z)), nil
	}
	expZ := math.Exp(z)
	return expZ / (1.0 + expZ), nil
}
