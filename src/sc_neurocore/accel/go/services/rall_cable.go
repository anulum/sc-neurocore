// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for rall_cable

package services

import (
	"math"
)

// RallCableNeuronState holds the neuron state
type RallCableNeuronState struct {
	NComp float64
	TauM float64
	VRest float64
	GRatio float64
	VThreshold float64
	VReset float64
	Dt float64
	V float64
}

// NewRallCableNeuron creates a new RallCableNeuron neuron with default parameters
func NewRallCableNeuron() *RallCableNeuronState {
	return &RallCableNeuronState{
		NComp: 5.0,
		TauM: 20.0,
		VRest: -65.0,
		GRatio: 0.5,
		VThreshold: -50.0,
		VReset: -65.0,
		Dt: 0.1,
		V: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *RallCableNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateRallCableNeuron runs the neuron for n steps
func SimulateRallCableNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewRallCableNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.NComp
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
