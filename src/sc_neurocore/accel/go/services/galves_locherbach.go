// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for galves_locherbach

package services

import (
	"math"
)

// GalvesLocherbachNeuronState holds the neuron state
type GalvesLocherbachNeuronState struct {
	V             float64
	VRest         float64
	Decay         float64
	ThresholdRate float64
	Steepness     float64
	Dt            float64
}

// NewGalvesLocherbachNeuron creates a new GalvesLocherbachNeuron neuron with default parameters
func NewGalvesLocherbachNeuron() *GalvesLocherbachNeuronState {
	return &GalvesLocherbachNeuronState{
		V:             0.0,
		VRest:         0.0,
		Decay:         0.95,
		ThresholdRate: 0.5,
		Steepness:     5.0,
		Dt:            1.0,
	}
}

// Step advances the neuron by one timestep
func (s *GalvesLocherbachNeuronState) Step(iExt float64) int {
	if !ValidateGalvesLocherbach(s) || !finite(iExt) {
		return 0
	}
	s.V = s.Decay*s.V + iExt
	p := s.FiringProbability() * s.Dt
	if p >= 1.0 {
		s.V = s.VRest
		return 1
	}
	return 0
}

// FiringProbability returns a numerically stable logistic firing probability.
func (s *GalvesLocherbachNeuronState) FiringProbability() float64 {
	z := s.Steepness * (s.V - s.ThresholdRate)
	if z >= 0.0 {
		tail := math.Exp(-z)
		return 1.0 / (1.0 + tail)
	}
	tail := math.Exp(z)
	return tail / (1.0 + tail)
}

// ValidateGalvesLocherbach enforces finite stochastic-history parameters.
func ValidateGalvesLocherbach(s *GalvesLocherbachNeuronState) bool {
	if s == nil {
		return false
	}
	return finite(s.V) && finite(s.VRest) && finite(s.ThresholdRate) &&
		finite(s.Decay) && s.Decay >= 0.0 && s.Decay <= 1.0 &&
		finite(s.Steepness) && s.Steepness > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0 && s.Dt <= 1.0
}

// SimulateGalvesLocherbachNeuron runs the neuron for n steps
func SimulateGalvesLocherbachNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGalvesLocherbachNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
