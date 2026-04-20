// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for prescott

package services

import (
	"math"
)

// PrescottNeuronState holds the neuron state
type PrescottNeuronState struct {
	V float64
	W float64
	GFast float64
	GSlow float64
	GL float64
	EFast float64
	ESlow float64
	EL float64
	BetaW float64
	GammaW float64
	TauW float64
	Phi float64
	Dt float64
	VThreshold float64
}

// NewPrescottNeuron creates a new PrescottNeuron neuron with default parameters
func NewPrescottNeuron() *PrescottNeuronState {
	return &PrescottNeuronState{
		V: -65.0,
		W: 0.0,
		GFast: 20.0,
		GSlow: 20.0,
		GL: 2.0,
		EFast: 50.0,
		ESlow: -100.0,
		EL: -70.0,
		BetaW: -21.0,
		GammaW: 15.0,
		TauW: 100.0,
		Phi: 0.15,
		Dt: 0.1,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *PrescottNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulatePrescottNeuron runs the neuron for n steps
func SimulatePrescottNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPrescottNeuron()
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
