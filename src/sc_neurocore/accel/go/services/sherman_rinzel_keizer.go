// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for sherman_rinzel_keizer

package services

import (
	"math"
)

// ShermanRinzelKeizerNeuronState holds the neuron state
type ShermanRinzelKeizerNeuronState struct {
	V float64
	N float64
	S float64
	GCa float64
	GK float64
	GS float64
	ECa float64
	EK float64
	TauS float64
	Dt float64
	VThreshold float64
}

// NewShermanRinzelKeizerNeuron creates a new ShermanRinzelKeizerNeuron neuron with default parameters
func NewShermanRinzelKeizerNeuron() *ShermanRinzelKeizerNeuronState {
	return &ShermanRinzelKeizerNeuronState{
		V: -50.0,
		N: 0.1,
		S: 0.1,
		GCa: 3.6,
		GK: 10.0,
		GS: 4.0,
		ECa: 25.0,
		EK: -75.0,
		TauS: 5000.0,
		Dt: 0.5,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *ShermanRinzelKeizerNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulateShermanRinzelKeizerNeuron runs the neuron for n steps
func SimulateShermanRinzelKeizerNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewShermanRinzelKeizerNeuron()
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
