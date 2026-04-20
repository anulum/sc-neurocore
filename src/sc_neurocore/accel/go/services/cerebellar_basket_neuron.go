// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for cerebellar_basket_neuron

package services

import (
	"math"
)

// CerebellarBasketNeuronState holds the neuron state
type CerebellarBasketNeuronState struct {
	V float64
	H float64
	N float64
	A float64
	B float64
	Ca float64
	GNa float64
	GK float64
	GA float64
	GKca float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CM float64
	Phi float64
	Dt float64
	VThreshold float64
}

// NewCerebellarBasketNeuron creates a new CerebellarBasketNeuron neuron with default parameters
func NewCerebellarBasketNeuron() *CerebellarBasketNeuronState {
	return &CerebellarBasketNeuronState{
		V: -65.0,
		H: 0.8,
		N: 0.1,
		A: 0.0,
		B: 0.9,
		Ca: 0.05,
		GNa: 35.0,
		GK: 9.0,
		GA: 3.0,
		GKca: 2.0,
		GL: 0.1,
		ENa: 55.0,
		EK: -90.0,
		EL: -65.0,
		CM: 1.0,
		Phi: 5.0,
		Dt: 0.01,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *CerebellarBasketNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateCerebellarBasketNeuron runs the neuron for n steps
func SimulateCerebellarBasketNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewCerebellarBasketNeuron()
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
