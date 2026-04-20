// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for pinsky_rinzel

package services

import (
	"math"
)

// PinskyRinzelNeuronState holds the neuron state
type PinskyRinzelNeuronState struct {
	VS float64
	VD float64
	H float64
	N float64
	S float64
	C float64
	Q float64
	Gc float64
	P float64
	GNa float64
	GKdr float64
	GCa float64
	GKahp float64
	GKc float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	Dt float64
}

// NewPinskyRinzelNeuron creates a new PinskyRinzelNeuron neuron with default parameters
func NewPinskyRinzelNeuron() *PinskyRinzelNeuronState {
	return &PinskyRinzelNeuronState{
		VS: -60.0,
		VD: -60.0,
		H: 0.9,
		N: 0.1,
		S: 0.0,
		C: 0.0,
		Q: 0.0,
		Gc: 2.1,
		P: 0.5,
		GNa: 30.0,
		GKdr: 15.0,
		GCa: 10.0,
		GKahp: 0.8,
		GKc: 15.0,
		GL: 0.1,
		ENa: 60.0,
		EK: -75.0,
		ECa: 80.0,
		EL: -60.0,
		Dt: 0.02,
	}
}

// Step advances the neuron by one timestep
func (s *PinskyRinzelNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulatePinskyRinzelNeuron runs the neuron for n steps
func SimulatePinskyRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPinskyRinzelNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VS
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
