// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for chay_keizer

package services

import (
	"math"
)

// ChayKeizerNeuronState holds the neuron state
type ChayKeizerNeuronState struct {
	V float64
	N float64
	Ca float64
	GCa float64
	GK float64
	GKca float64
	GL float64
	ECa float64
	EK float64
	EL float64
	KD float64
	FCa float64
	KCa float64
	Dt float64
	VThreshold float64
}

// NewChayKeizerNeuron creates a new ChayKeizerNeuron neuron with default parameters
func NewChayKeizerNeuron() *ChayKeizerNeuronState {
	return &ChayKeizerNeuronState{
		V: -50.0,
		N: 0.01,
		Ca: 0.1,
		GCa: 20.0,
		GK: 25.0,
		GKca: 12.0,
		GL: 0.1,
		ECa: 100.0,
		EK: -75.0,
		EL: -40.0,
		KD: 1.0,
		FCa: 0.004,
		KCa: 0.03,
		Dt: 0.02,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *ChayKeizerNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulateChayKeizerNeuron runs the neuron for n steps
func SimulateChayKeizerNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewChayKeizerNeuron()
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
