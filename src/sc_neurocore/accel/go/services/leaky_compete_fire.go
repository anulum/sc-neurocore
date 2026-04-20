// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for leaky_compete_fire

package services

import (
	"math"
)

// LeakyCompeteFireNeuronState holds the neuron state
type LeakyCompeteFireNeuronState struct {
	NUnits float64
	V float64
	Tau float64
	VThreshold float64
	WInh float64
	Dt float64
}

// NewLeakyCompeteFireNeuron creates a new LeakyCompeteFireNeuron neuron with default parameters
func NewLeakyCompeteFireNeuron() *LeakyCompeteFireNeuronState {
	return &LeakyCompeteFireNeuronState{
		NUnits: 4.0,
		V: 0.0,
		Tau: 10.0,
		VThreshold: 1.0,
		WInh: 0.5,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *LeakyCompeteFireNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = 0.0
		return 1
	}
	return 0
}

// SimulateLeakyCompeteFireNeuron runs the neuron for n steps
func SimulateLeakyCompeteFireNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLeakyCompeteFireNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.NUnits
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
