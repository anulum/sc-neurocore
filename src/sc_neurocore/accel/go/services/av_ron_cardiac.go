// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for av_ron_cardiac

package services

import (
	"math"
)

// AvRonCardiacNeuronState holds the neuron state
type AvRonCardiacNeuronState struct {
	V float64
	H float64
	N float64
	S float64
	GNa float64
	GK float64
	GS float64
	GL float64
	ENa float64
	EK float64
	ES float64
	EL float64
	Dt float64
	VThreshold float64
}

// NewAvRonCardiacNeuron creates a new AvRonCardiacNeuron neuron with default parameters
func NewAvRonCardiacNeuron() *AvRonCardiacNeuronState {
	return &AvRonCardiacNeuronState{
		V: -60.0,
		H: 0.6,
		N: 0.3,
		S: 0.5,
		GNa: 80.0,
		GK: 40.0,
		GS: 20.0,
		GL: 0.1,
		ENa: 40.0,
		EK: -80.0,
		ES: -25.0,
		EL: -60.0,
		Dt: 0.02,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *AvRonCardiacNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -60.0
		return 1
	}
	return 0
}

// SimulateAvRonCardiacNeuron runs the neuron for n steps
func SimulateAvRonCardiacNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAvRonCardiacNeuron()
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
