// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for compte_wm

package services

import (
	"math"
)

// CompteWMNeuronState holds the neuron state
type CompteWMNeuronState struct {
	V float64
	SAmpa float64
	SNmda float64
	XNmda float64
	SGaba float64
	GL float64
	GAmpa float64
	GNmda float64
	GGaba float64
	EL float64
	EExc float64
	EInh float64
	CM float64
	Mg float64
	TauAmpa float64
	TauNmda float64
	TauX float64
	AlphaNmda float64
	VThreshold float64
	VReset float64
}

// NewCompteWMNeuron creates a new CompteWMNeuron neuron with default parameters
func NewCompteWMNeuron() *CompteWMNeuronState {
	return &CompteWMNeuronState{
		V: -70.0,
		SAmpa: 0.0,
		SNmda: 0.0,
		XNmda: 0.0,
		SGaba: 0.0,
		GL: 0.025,
		GAmpa: 0.005,
		GNmda: 0.165,
		GGaba: 0.013,
		EL: -70.0,
		EExc: 0.0,
		EInh: -70.0,
		CM: 0.5,
		Mg: 1.0,
		TauAmpa: 2.0,
		TauNmda: 100.0,
		TauX: 2.0,
		AlphaNmda: 0.5,
		VThreshold: -50.0,
		VReset: -55.0,
	}
}

// Step advances the neuron by one timestep
func (s *CompteWMNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateCompteWMNeuron runs the neuron for n steps
func SimulateCompteWMNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewCompteWMNeuron()
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
