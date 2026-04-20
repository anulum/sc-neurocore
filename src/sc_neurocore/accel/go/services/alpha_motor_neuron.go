// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for alpha_motor_neuron

package services

import (
	"math"
)

// AlphaMotorNeuronState holds the neuron state
type AlphaMotorNeuronState struct {
	V float64
	H float64
	N float64
	MPic float64
	HPic float64
	Ca float64
	CaBuf float64
	GNa float64
	GK float64
	GPic float64
	GAhp float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	CM float64
	Phi float64
	TauCa float64
	BufRatio float64
	VThreshold float64
}

// NewAlphaMotorNeuron creates a new AlphaMotorNeuron neuron with default parameters
func NewAlphaMotorNeuron() *AlphaMotorNeuronState {
	return &AlphaMotorNeuronState{
		V: -65.0,
		H: 0.8,
		N: 0.1,
		MPic: 0.0,
		HPic: 1.0,
		Ca: 0.0,
		CaBuf: 0.0,
		GNa: 35.0,
		GK: 9.0,
		GPic: 0.15,
		GAhp: 3.0,
		GL: 0.3,
		ENa: 55.0,
		EK: -90.0,
		ECa: 120.0,
		EL: -65.0,
		CM: 1.5,
		Phi: 4.0,
		TauCa: 150.0,
		BufRatio: 0.003,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *AlphaMotorNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateAlphaMotorNeuron runs the neuron for n steps
func SimulateAlphaMotorNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAlphaMotorNeuron()
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
