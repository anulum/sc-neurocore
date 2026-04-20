// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for expif

package services

import (
	"math"
)

// ExpIFNeuronState holds the neuron state
type ExpIFNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	VRh float64
	DeltaT float64
	Tau float64
	Dt float64
}

// NewExpIFNeuron creates a new ExpIFNeuron neuron with default parameters
func NewExpIFNeuron() *ExpIFNeuronState {
	return &ExpIFNeuronState{
		V: -65.0,
		VRest: -65.0,
		VReset: -68.0,
		VThreshold: -50.0,
		VRh: -55.0,
		DeltaT: 2.0,
		Tau: 20.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *ExpIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateExpIFNeuron runs the neuron for n steps
func SimulateExpIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewExpIFNeuron()
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
