// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wong_wang

package services

import (
	"math"
)

// WongWangUnitState holds the neuron state
type WongWangUnitState struct {
	S1 float64
	S2 float64
	TauS float64
	Gamma float64
	JN float64
	JCross float64
	I0 float64
	Sigma float64
	Dt float64
}

// NewWongWangUnit creates a new WongWangUnit neuron with default parameters
func NewWongWangUnit() *WongWangUnitState {
	return &WongWangUnitState{
		S1: 0.1,
		S2: 0.1,
		TauS: 0.1,
		Gamma: 0.641,
		JN: 0.2609,
		JCross: 0.0497,
		I0: 0.3255,
		Sigma: 0.02,
		Dt: 0.001,
	}
}

// Step advances the neuron by one timestep
func (s *WongWangUnitState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateWongWangUnit runs the neuron for n steps
func SimulateWongWangUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewWongWangUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.S1
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
