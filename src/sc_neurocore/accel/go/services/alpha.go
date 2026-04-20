// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for alpha

package services

import (
	"math"
)

// AlphaNeuronState holds the neuron state
type AlphaNeuronState struct {
	V float64
	IExc float64
	IInh float64
	VRest float64
	VThreshold float64
	TauV float64
	TauExc float64
	TauInh float64
	Dt float64
}

// NewAlphaNeuron creates a new AlphaNeuron neuron with default parameters
func NewAlphaNeuron() *AlphaNeuronState {
	return &AlphaNeuronState{
		V: 0.0,
		IExc: 0.0,
		IInh: 0.0,
		VRest: 0.0,
		VThreshold: 1.0,
		TauV: 20.0,
		TauExc: 5.0,
		TauInh: 10.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *AlphaNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = 0.0
		return 1
	}
	return 0
}

// SimulateAlphaNeuron runs the neuron for n steps
func SimulateAlphaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAlphaNeuron()
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
