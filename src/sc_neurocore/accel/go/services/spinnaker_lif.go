// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for spinnaker_lif

package services

import (
	"math"
)

// SpiNNakerLIFNeuronState holds the neuron state
type SpiNNakerLIFNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	IOffset float64
	TauRefrac float64
	RefracCount float64
	Dt float64
}

// NewSpiNNakerLIFNeuron creates a new SpiNNakerLIFNeuron neuron with default parameters
func NewSpiNNakerLIFNeuron() *SpiNNakerLIFNeuronState {
	return &SpiNNakerLIFNeuronState{
		V: -70.0,
		VRest: -70.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 20.0,
		IOffset: 0.0,
		TauRefrac: 2.0,
		RefracCount: 0.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *SpiNNakerLIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateSpiNNakerLIFNeuron runs the neuron for n steps
func SimulateSpiNNakerLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSpiNNakerLIFNeuron()
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
