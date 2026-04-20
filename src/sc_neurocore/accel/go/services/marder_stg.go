// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for marder_stg

package services

import (
	"math"
)

// MarderSTGNeuronState holds the neuron state
type MarderSTGNeuronState struct {
	V float64
	MNa float64
	HNa float64
	MCat float64
	HCat float64
	MCas float64
	MA float64
	HA float64
	MKd float64
	MH float64
	Ca float64
	GNa float64
	GCat float64
	GCas float64
	GA float64
	GKca float64
	GKd float64
	GH float64
	GL float64
	ENa float64
	VThreshold float64
}

// NewMarderSTGNeuron creates a new MarderSTGNeuron neuron with default parameters
func NewMarderSTGNeuron() *MarderSTGNeuronState {
	return &MarderSTGNeuronState{
		V: -60.0,
		MNa: 0.0,
		HNa: 0.9,
		MCat: 0.0,
		HCat: 0.9,
		MCas: 0.0,
		MA: 0.0,
		HA: 0.9,
		MKd: 0.0,
		MH: 0.0,
		Ca: 0.05,
		GNa: 200.0,
		GCat: 2.5,
		GCas: 4.0,
		GA: 50.0,
		GKca: 25.0,
		GKd: 75.0,
		GH: 0.01,
		GL: 0.01,
		ENa: 50.0,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *MarderSTGNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -60.0
		return 1
	}
	return 0
}

// SimulateMarderSTGNeuron runs the neuron for n steps
func SimulateMarderSTGNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMarderSTGNeuron()
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
