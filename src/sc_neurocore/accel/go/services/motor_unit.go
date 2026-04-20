// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for motor_unit

package services

import (
	"math"
)

// MotorUnitState holds the neuron state
type MotorUnitState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	Adapt float64
	TauAdapt float64
	AAdapt float64
	Gain float64
	Force float64
	TwitchAmp float64
	TauTwitch float64
	ForceDecay float64
	Dt float64
}

// NewMotorUnit creates a new MotorUnit neuron with default parameters
func NewMotorUnit() *MotorUnitState {
	return &MotorUnitState{
		V: -65.0,
		VRest: -65.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 10.0,
		Adapt: 0.0,
		TauAdapt: 100.0,
		AAdapt: 0.2,
		Gain: 1.0,
		Force: 0.0,
		TwitchAmp: 0.05,
		TauTwitch: 90.0,
		ForceDecay: 0.0,
		Dt: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *MotorUnitState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateMotorUnit runs the neuron for n steps
func SimulateMotorUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewMotorUnit()
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
