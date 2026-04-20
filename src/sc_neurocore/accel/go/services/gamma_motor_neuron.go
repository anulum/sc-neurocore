// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gamma_motor_neuron

package services

import (
	"math"
)

// GammaMotorNeuronState holds the neuron state
type GammaMotorNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	Tau float64
	Adapt float64
	TauAdapt float64
	AAdapt float64
	Gain float64
	Dynamic float64
	Dt float64
}

// NewGammaMotorNeuron creates a new GammaMotorNeuron neuron with default parameters
func NewGammaMotorNeuron() *GammaMotorNeuronState {
	return &GammaMotorNeuronState{
		V: -65.0,
		VRest: -65.0,
		VReset: -70.0,
		VThreshold: -50.0,
		Tau: 8.0,
		Adapt: 0.0,
		TauAdapt: 100.0,
		AAdapt: 0.3,
		Gain: 1.0,
		Dynamic: 1.0,
		Dt: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *GammaMotorNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateGammaMotorNeuron runs the neuron for n steps
func SimulateGammaMotorNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGammaMotorNeuron()
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
