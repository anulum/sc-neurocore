// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for yamada

package services

import (
	"math"
)

// YamadaNeuronState holds the neuron state
type YamadaNeuronState struct {
	V float64
	N float64
	Q float64
	GNa float64
	GK float64
	GQ float64
	GL float64
	ENa float64
	EK float64
	EQ float64
	EL float64
	TauQ float64
	Dt float64
	VThreshold float64
}

// NewYamadaNeuron creates a new YamadaNeuron neuron with default parameters
func NewYamadaNeuron() *YamadaNeuronState {
	return &YamadaNeuronState{
		V: -60.0,
		N: 0.1,
		Q: 0.0,
		GNa: 20.0,
		GK: 10.0,
		GQ: 5.0,
		GL: 0.5,
		ENa: 60.0,
		EK: -80.0,
		EQ: -80.0,
		EL: -60.0,
		TauQ: 300.0,
		Dt: 0.05,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *YamadaNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -60.0
		return 1
	}
	return 0
}

// SimulateYamadaNeuron runs the neuron for n steps
func SimulateYamadaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewYamadaNeuron()
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
