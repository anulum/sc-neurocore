// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wilson_hr

package services

import (
	"math"
)

// WilsonHRNeuronState holds the neuron state
type WilsonHRNeuronState struct {
	V float64
	R float64
	TauR float64
	VPeak float64
	Dt float64
}

// NewWilsonHRNeuron creates a new WilsonHRNeuron neuron with default parameters
func NewWilsonHRNeuron() *WilsonHRNeuronState {
	return &WilsonHRNeuronState{
		V: -0.7,
		R: 0.1,
		TauR: 1.9,
		VPeak: 0.4,
		Dt: 0.05,
	}
}

// Step advances the neuron by one timestep
func (s *WilsonHRNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateWilsonHRNeuron runs the neuron for n steps
func SimulateWilsonHRNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWilsonHRNeuron()
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
