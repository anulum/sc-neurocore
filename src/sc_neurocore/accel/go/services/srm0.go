// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for srm0

package services

import (
	"math"
)

// SRM0NeuronState holds the neuron state
type SRM0NeuronState struct {
	V float64
	VRest float64
	VThreshold float64
	TauM float64
	TauEta float64
	EtaReset float64
	Resistance float64
	Dt float64
}

// NewSRM0Neuron creates a new SRM0Neuron neuron with default parameters
func NewSRM0Neuron() *SRM0NeuronState {
	return &SRM0NeuronState{
		V: 0.0,
		VRest: 0.0,
		VThreshold: 1.0,
		TauM: 20.0,
		TauEta: 50.0,
		EtaReset: 5.0,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *SRM0NeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = 0.0
		return 1
	}
	return 0
}

// SimulateSRM0Neuron runs the neuron for n steps
func SimulateSRM0Neuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSRM0Neuron()
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
