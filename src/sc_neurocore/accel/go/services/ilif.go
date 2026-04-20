// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ilif

package services

import (
	"math"
)

// InhibitoryLIFNeuronState holds the neuron state
type InhibitoryLIFNeuronState struct {
	V float64
	InhTrace float64
	TauM float64
	TauInh float64
	VThreshold float64
	VReset float64
	InhStrength float64
	Dt float64
	AlphaM float64
	AlphaInh float64
}

// NewInhibitoryLIFNeuron creates a new InhibitoryLIFNeuron neuron with default parameters
func NewInhibitoryLIFNeuron() *InhibitoryLIFNeuronState {
	return &InhibitoryLIFNeuronState{
		V: 0.0,
		InhTrace: 0.0,
		TauM: 10.0,
		TauInh: 5.0,
		VThreshold: 1.0,
		VReset: 0.0,
		InhStrength: 0.5,
		Dt: 1.0,
		AlphaM: 0.0,
		AlphaInh: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *InhibitoryLIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateInhibitoryLIFNeuron runs the neuron for n steps
func SimulateInhibitoryLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewInhibitoryLIFNeuron()
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
