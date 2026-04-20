// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for sfa

package services

import (
	"math"
)

// SFANeuronState holds the neuron state
type SFANeuronState struct {
	V float64
	GSfa float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	TauSfa float64
	DeltaG float64
	EK float64
	Resistance float64
	Dt float64
}

// NewSFANeuron creates a new SFANeuron neuron with default parameters
func NewSFANeuron() *SFANeuronState {
	return &SFANeuronState{
		V: -70.0,
		GSfa: 0.0,
		VRest: -70.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 10.0,
		TauSfa: 200.0,
		DeltaG: 0.5,
		EK: -80.0,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *SFANeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateSFANeuron runs the neuron for n steps
func SimulateSFANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSFANeuron()
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
