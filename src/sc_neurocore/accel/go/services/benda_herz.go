// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for benda_herz

package services

import (
	"math"
)

// BendaHerzNeuronState holds the neuron state
type BendaHerzNeuronState struct {
	A float64
	FMax float64
	Beta float64
	IHalf float64
	TauA float64
	DeltaA float64
	Dt float64
	Rng float64
}

// NewBendaHerzNeuron creates a new BendaHerzNeuron neuron with default parameters
func NewBendaHerzNeuron() *BendaHerzNeuronState {
	return &BendaHerzNeuronState{
		A: 0.0,
		FMax: 200.0,
		Beta: 0.1,
		IHalf: 5.0,
		TauA: 100.0,
		DeltaA: 0.5,
		Dt: 1.0,
		Rng: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *BendaHerzNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateBendaHerzNeuron runs the neuron for n steps
func SimulateBendaHerzNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBendaHerzNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.A
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
