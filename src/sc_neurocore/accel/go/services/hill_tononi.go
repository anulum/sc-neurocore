// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hill_tononi

package services

import (
	"math"
)

// HillTononiNeuronState holds the neuron state
type HillTononiNeuronState struct {
	V float64
	HNa float64
	NK float64
	MH float64
	HT float64
	NaI float64
	GNa float64
	GK float64
	GH float64
	GT float64
	GKna float64
	GL float64
	ENa float64
	EK float64
	EH float64
	ECa float64
	EL float64
	NaPumpMax float64
	NaEq float64
	Dt float64
	VThreshold float64
}

// NewHillTononiNeuron creates a new HillTononiNeuron neuron with default parameters
func NewHillTononiNeuron() *HillTononiNeuronState {
	return &HillTononiNeuronState{
		V: -65.0,
		HNa: 0.6,
		NK: 0.3,
		MH: 0.0,
		HT: 0.9,
		NaI: 5.0,
		GNa: 50.0,
		GK: 5.0,
		GH: 1.0,
		GT: 3.0,
		GKna: 1.33,
		GL: 0.02,
		ENa: 50.0,
		EK: -90.0,
		EH: -43.0,
		ECa: 120.0,
		EL: -70.0,
		NaPumpMax: 20.0,
		NaEq: 9.5,
		Dt: 0.05,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *HillTononiNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateHillTononiNeuron runs the neuron for n steps
func SimulateHillTononiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHillTononiNeuron()
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
