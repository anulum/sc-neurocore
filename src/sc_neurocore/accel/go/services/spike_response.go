// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for spike_response

package services

import (
	"math"
)

// SpikeResponseNeuronState holds the neuron state
type SpikeResponseNeuronState struct {
	V float64
	VThreshold float64
	TauEta float64
	TauKappa float64
	EtaReset float64
	TimeSinceSpike float64
	Dt float64
}

// NewSpikeResponseNeuron creates a new SpikeResponseNeuron neuron with default parameters
func NewSpikeResponseNeuron() *SpikeResponseNeuronState {
	return &SpikeResponseNeuronState{
		V: 0.0,
		VThreshold: 1.0,
		TauEta: 10.0,
		TauKappa: 5.0,
		EtaReset: -5.0,
		TimeSinceSpike: 1000.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *SpikeResponseNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = 0.0
		return 1
	}
	return 0
}

// SimulateSpikeResponseNeuron runs the neuron for n steps
func SimulateSpikeResponseNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSpikeResponseNeuron()
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
