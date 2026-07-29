// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for kilinc_bhatt_map_neuron

package services

import (
	"errors"
	"math"
)

// KilincBhattMapNeuronState holds the neuron state
type KilincBhattMapNeuronState struct {
	X          float64
	Theta      float64
	K          float64
	Beta       float64
	Gamma      float64
	ThetaSpike float64
	XThreshold float64
}

// NewKilincBhattMapNeuron creates a new KilincBhattMapNeuron neuron with default parameters
func NewKilincBhattMapNeuron() *KilincBhattMapNeuronState {
	return &KilincBhattMapNeuronState{
		X:          0.0,
		Theta:      0.0,
		K:          1.5,
		Beta:       0.95,
		Gamma:      0.3,
		ThetaSpike: 0.8,
		XThreshold: 0.8,
	}
}

func kilincBhattFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func kilincBhattBounded(value, lower, upper float64) bool {
	return kilincBhattFinite(value) && value >= lower && value <= upper
}

// Valid reports whether state and parameters satisfy the public model bounds.
func (s *KilincBhattMapNeuronState) Valid() bool {
	return kilincBhattBounded(s.X, -5.0, 5.0) &&
		kilincBhattBounded(s.Theta, -5.0, 5.0) &&
		kilincBhattBounded(s.K, 0.0, 5.0) &&
		kilincBhattBounded(s.Beta, 0.0, 1.0) &&
		kilincBhattBounded(s.Gamma, 0.0, 2.0) &&
		kilincBhattBounded(s.ThetaSpike, 0.0, 2.0) &&
		kilincBhattBounded(s.XThreshold, 0.0, 2.0)
}

func kilincBhattSigmoid(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	expValue := math.Exp(value)
	return expValue / (1.0 + expValue)
}

// TryStep advances the neuron atomically or returns a validation error.
func (s *KilincBhattMapNeuronState) TryStep(iExt float64) (int, error) {
	if !kilincBhattFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !s.Valid() {
		return 0, errors.New("Kilinc-Bhatt state and parameters must satisfy the public bounds")
	}

	xPrev := s.X
	sig := kilincBhattSigmoid((s.X - s.Theta) * 4.0)
	xNew := -s.X + s.K*sig + iExt
	spiked := 0.0
	if s.X >= s.ThetaSpike {
		spiked = 1.0
	}
	thetaNew := s.Beta*s.Theta + s.Gamma*spiked
	if !kilincBhattFinite(xNew) || !kilincBhattFinite(thetaNew) {
		return 0, errors.New("Kilinc-Bhatt candidate state became non-finite")
	}

	s.X = math.Max(-5.0, math.Min(5.0, xNew))
	s.Theta = math.Max(-5.0, math.Min(5.0, thetaNew))
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// Step advances the neuron by one timestep and fails closed on invalid input.
func (s *KilincBhattMapNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// SimulateKilincBhattMapNeuron runs the neuron for n steps
func SimulateKilincBhattMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewKilincBhattMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
