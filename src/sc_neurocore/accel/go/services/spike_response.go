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

// SpikeResponseNeuronState holds the SRM kernel state.
type SpikeResponseNeuronState struct {
	V              float64
	VThreshold     float64
	TauEta         float64
	TauKappa       float64
	EtaReset       float64
	TimeSinceSpike float64
	Dt             float64
}

// NewSpikeResponseNeuron creates a Spike Response neuron with documented defaults.
func NewSpikeResponseNeuron() *SpikeResponseNeuronState {
	return &SpikeResponseNeuronState{
		V:              0.0,
		VThreshold:     1.0,
		TauEta:         10.0,
		TauKappa:       5.0,
		EtaReset:       -5.0,
		TimeSinceSpike: 1000.0,
		Dt:             1.0,
	}
}

// Step advances the SRM kernel by one timestep.
func (s *SpikeResponseNeuronState) Step(weightedInput float64) int {
	if !s.Valid() || !isFiniteSpikeResponse(weightedInput) {
		return -1
	}
	eta := 0.0
	if s.TimeSinceSpike < 100.0 {
		eta = s.EtaReset * math.Exp(-s.TimeSinceSpike/s.TauEta)
	}
	kappa := weightedInput * (1.0 - math.Exp(-s.Dt/s.TauKappa))
	nextV := eta + kappa
	if !isFiniteSpikeResponse(nextV) {
		return -1
	}
	s.V = nextV
	s.TimeSinceSpike += s.Dt
	if s.V >= s.VThreshold {
		s.TimeSinceSpike = 0.0
		s.V = 0.0
		return 1
	}
	return 0
}

// Reset restores the documented resting kernel state.
func (s *SpikeResponseNeuronState) Reset() {
	s.V = 0.0
	s.TimeSinceSpike = 1000.0
}

// Valid reports whether the SRM kernel scalar contract is satisfied.
func (s *SpikeResponseNeuronState) Valid() bool {
	return isFiniteSpikeResponse(s.V) &&
		isFiniteSpikeResponse(s.VThreshold) &&
		isFiniteSpikeResponse(s.TauEta) &&
		s.TauEta > 0.0 &&
		isFiniteSpikeResponse(s.TauKappa) &&
		s.TauKappa > 0.0 &&
		isFiniteSpikeResponse(s.EtaReset) &&
		isFiniteSpikeResponse(s.TimeSinceSpike) &&
		s.TimeSinceSpike >= 0.0 &&
		isFiniteSpikeResponse(s.Dt) &&
		s.Dt > 0.0
}

func isFiniteSpikeResponse(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// SimulateSpikeResponseNeuron runs the neuron for n steps.
func SimulateSpikeResponseNeuron(nSteps int, weightedInput float64) ([]float64, int) {
	s := NewSpikeResponseNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(weightedInput)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
