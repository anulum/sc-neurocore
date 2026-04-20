// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for butera_respiratory

package services

import (
	"math"
)

// ButeraRespiratoryNeuronState holds the neuron state
type ButeraRespiratoryNeuronState struct {
	V float64
	N float64
	HNap float64
	GNa float64
	GNap float64
	GK float64
	GL float64
	ENa float64
	EK float64
	EL float64
	ESyn float64
	TauH float64
	Dt float64
	VThreshold float64
}

// NewButeraRespiratoryNeuron creates a new ButeraRespiratoryNeuron neuron with default parameters
func NewButeraRespiratoryNeuron() *ButeraRespiratoryNeuronState {
	return &ButeraRespiratoryNeuronState{
		V: -50.0,
		N: 0.01,
		HNap: 0.5,
		GNa: 28.0,
		GNap: 2.8,
		GK: 11.2,
		GL: 2.8,
		ENa: 50.0,
		EK: -85.0,
		EL: -65.0,
		ESyn: -10.0,
		TauH: 10000.0,
		Dt: 0.1,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *ButeraRespiratoryNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulateButeraRespiratoryNeuron runs the neuron for n steps
func SimulateButeraRespiratoryNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewButeraRespiratoryNeuron()
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
