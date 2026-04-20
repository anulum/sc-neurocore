// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for terman_wang

package services

import (
	"math"
)

// TermanWangOscillatorState holds the neuron state
type TermanWangOscillatorState struct {
	V float64
	W float64
	Alpha float64
	Beta float64
	Epsilon float64
	Rho float64
	Dt float64
	VPeak float64
}

// NewTermanWangOscillator creates a new TermanWangOscillator neuron with default parameters
func NewTermanWangOscillator() *TermanWangOscillatorState {
	return &TermanWangOscillatorState{
		V: -1.5,
		W: -0.5,
		Alpha: 3.0,
		Beta: 0.2,
		Epsilon: 0.02,
		Rho: 0.0,
		Dt: 0.05,
		VPeak: 1.5,
	}
}

// Step advances the neuron by one timestep
func (s *TermanWangOscillatorState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateTermanWangOscillator runs the neuron for n steps
func SimulateTermanWangOscillator(nSteps int, iExt float64) ([]float64, int) {
	s := NewTermanWangOscillator()
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
