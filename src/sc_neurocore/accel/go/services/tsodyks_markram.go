// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for tsodyks_markram

package services

import (
	"math"
)

// TsodyksMarkramNeuronState holds the neuron state
type TsodyksMarkramNeuronState struct {
	V float64
	X float64
	U float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	TauD float64
	TauF float64
	USe float64
	ASe float64
	RM float64
	Dt float64
}

// NewTsodyksMarkramNeuron creates a new TsodyksMarkramNeuron neuron with default parameters
func NewTsodyksMarkramNeuron() *TsodyksMarkramNeuronState {
	return &TsodyksMarkramNeuronState{
		V: -65.0,
		X: 1.0,
		U: 0.2,
		VRest: -65.0,
		VReset: -65.0,
		VThreshold: -50.0,
		TauM: 20.0,
		TauD: 200.0,
		TauF: 600.0,
		USe: 0.2,
		ASe: 50.0,
		RM: 1.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *TsodyksMarkramNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateTsodyksMarkramNeuron runs the neuron for n steps
func SimulateTsodyksMarkramNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTsodyksMarkramNeuron()
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
