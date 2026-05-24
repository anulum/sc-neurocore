// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lnm

package services

import (
	"math"
)

// LearnableNeuronModelState holds the neuron state
type LearnableNeuronModelState struct {
	V          float64
	Alpha      float64
	Beta       float64
	Gamma      float64
	VThreshold float64
	VReset     float64
	FSlope     float64
	FShift     float64
}

// NewLearnableNeuronModel creates a new LearnableNeuronModel neuron with default parameters
func NewLearnableNeuronModel() *LearnableNeuronModelState {
	return &LearnableNeuronModelState{
		V:          0.0,
		Alpha:      0.9,
		Beta:       0.1,
		Gamma:      0.05,
		VThreshold: 1.0,
		VReset:     0.0,
		FSlope:     5.0,
		FShift:     0.5,
	}
}

// ValidateLearnableNeuronModel checks that the learnable neuron parameters are finite.
func ValidateLearnableNeuronModel(s *LearnableNeuronModelState) bool {
	return s != nil &&
		!math.IsNaN(s.V) && !math.IsInf(s.V, 0) &&
		!math.IsNaN(s.Alpha) && !math.IsInf(s.Alpha, 0) &&
		!math.IsNaN(s.Beta) && !math.IsInf(s.Beta, 0) &&
		!math.IsNaN(s.Gamma) && !math.IsInf(s.Gamma, 0) &&
		!math.IsNaN(s.VThreshold) && !math.IsInf(s.VThreshold, 0) && s.VThreshold > 0 &&
		!math.IsNaN(s.VReset) && !math.IsInf(s.VReset, 0) &&
		!math.IsNaN(s.FSlope) && !math.IsInf(s.FSlope, 0) && s.FSlope > 0 &&
		!math.IsNaN(s.FShift) && !math.IsInf(s.FShift, 0)
}

func lnmSigmoid(value float64) float64 {
	if value >= 0 {
		z := math.Exp(-value)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(value)
	return z / (1.0 + z)
}

// Step advances the neuron by one timestep
func (s *LearnableNeuronModelState) Step(iExt float64) int {
	if !ValidateLearnableNeuronModel(s) || math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0
	}

	fV := lnmSigmoid(s.FSlope * (s.V - s.FShift))
	s.V = s.Alpha*s.V + s.Beta*iExt + s.Gamma*fV
	if s.V >= s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateLearnableNeuronModel runs the neuron for n steps
func SimulateLearnableNeuronModel(nSteps int, iExt float64) ([]float64, int) {
	s := NewLearnableNeuronModel()
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

var _ = math.Exp
