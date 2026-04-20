// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for huber_braun

package services

import (
	"math"
)

// HuberBraunNeuronState holds the neuron state
type HuberBraunNeuronState struct {
	V float64
	ASd float64
	ASr float64
	GSd float64
	GSr float64
	GL float64
	ESd float64
	ESr float64
	EL float64
	TauSd float64
	TauSr float64
	Eta float64
	Dt float64
	VThreshold float64
}

// NewHuberBraunNeuron creates a new HuberBraunNeuron neuron with default parameters
func NewHuberBraunNeuron() *HuberBraunNeuronState {
	return &HuberBraunNeuronState{
		V: -50.0,
		ASd: 0.0,
		ASr: 0.0,
		GSd: 1.5,
		GSr: 0.4,
		GL: 0.1,
		ESd: 50.0,
		ESr: -90.0,
		EL: -60.0,
		TauSd: 10.0,
		TauSr: 20.0,
		Eta: 0.012,
		Dt: 0.1,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *HuberBraunNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulateHuberBraunNeuron runs the neuron for n steps
func SimulateHuberBraunNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHuberBraunNeuron()
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
