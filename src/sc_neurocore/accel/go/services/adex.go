// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for adex

package services

import (
	"math"
)

// AdExNeuronState holds the neuron state
type AdExNeuronState struct {
	V float64
	W float64
	VRest float64
	VReset float64
	VThreshold float64
	VRh float64
	DeltaT float64
	Tau float64
	TauW float64
	A float64
	B float64
	CM float64
	Dt float64
}

// NewAdExNeuron creates a new AdExNeuron neuron with default parameters
func NewAdExNeuron() *AdExNeuronState {
	return &AdExNeuronState{
		V: -65.0,
		W: 0.0,
		VRest: -65.0,
		VReset: -68.0,
		VThreshold: -50.0,
		VRh: -55.0,
		DeltaT: 2.0,
		Tau: 20.0,
		TauW: 100.0,
		A: 0.5,
		B: 7.0,
		CM: 200.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *AdExNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateAdExNeuron runs the neuron for n steps
func SimulateAdExNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAdExNeuron()
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
