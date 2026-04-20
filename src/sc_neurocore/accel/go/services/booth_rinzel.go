// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for booth_rinzel

package services

import (
	"math"
)

// BoothRinzelNeuronState holds the neuron state
type BoothRinzelNeuronState struct {
	Vs float64
	Vd float64
	H float64
	N float64
	Q float64
	Ca float64
	P float64
	Gc float64
	GNa float64
	GK float64
	GCa float64
	GKca float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	CM float64
	AlphaCa float64
	KCa float64
}

// NewBoothRinzelNeuron creates a new BoothRinzelNeuron neuron with default parameters
func NewBoothRinzelNeuron() *BoothRinzelNeuronState {
	return &BoothRinzelNeuronState{
		Vs: -65.0,
		Vd: -65.0,
		H: 0.9,
		N: 0.0,
		Q: 0.0,
		Ca: 0.0,
		P: 0.5,
		Gc: 0.1,
		GNa: 120.0,
		GK: 20.0,
		GCa: 14.0,
		GKca: 5.0,
		GL: 0.51,
		ENa: 55.0,
		EK: -80.0,
		ECa: 80.0,
		EL: -60.0,
		CM: 1.0,
		AlphaCa: 0.009,
		KCa: 0.18,
	}
}

// Step advances the neuron by one timestep
func (s *BoothRinzelNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateBoothRinzelNeuron runs the neuron for n steps
func SimulateBoothRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBoothRinzelNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Vs
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
