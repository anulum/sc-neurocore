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
	V          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	TauM       float64
	Adapt      float64
	TauAdapt   float64
	AAdapt     float64
	Gain       float64
	Force      float64
	TwitchAmp  float64
	TauTwitch  float64
	ForceDecay float64
	Dt         float64
}

// NewMotorUnit creates a new MotorUnit neuron with default parameters
func NewMotorUnit() *MotorUnitState {
	return &MotorUnitState{
		V:          -65.0,
		VRest:      -65.0,
		VReset:     -70.0,
		VThreshold: -50.0,
		TauM:       10.0,
		Adapt:      0.0,
		TauAdapt:   100.0,
		AAdapt:     0.2,
		Gain:       1.0,
		Force:      0.0,
		TwitchAmp:  0.05,
		TauTwitch:  90.0,
		ForceDecay: 0.0,
		Dt:         0.5,
	}
}

func NewSlowMotorUnit() *MotorUnitState {
	return NewMotorUnit()
}

func NewFastMotorUnit() *MotorUnitState {
	unit := NewMotorUnit()
	unit.TauM = 6.0
	unit.TauAdapt = 50.0
	unit.AAdapt = 0.1
	unit.TwitchAmp = 0.3
	unit.TauTwitch = 30.0
	return unit
}

func motorUnitFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func motorUnitVoltage(value float64) bool {
	return motorUnitFinite(value) && value >= -150.0 && value <= 100.0
}

func motorUnitForce(value float64) bool {
	return motorUnitFinite(value) && value >= 0.0 && value <= 1.0
}

func motorUnitExactRelax(previous, steady, tau, dt float64) (float64, bool) {
	if !motorUnitFinite(previous, steady, tau, dt) || tau <= 0.0 || dt <= 0.0 {
		return previous, false
	}
	return steady + (previous-steady)*math.Exp(-dt/tau), true
}

func (s *MotorUnitState) valid() bool {
	return motorUnitVoltage(s.V) &&
		motorUnitVoltage(s.VRest) &&
		motorUnitVoltage(s.VReset) &&
		motorUnitVoltage(s.VThreshold) &&
		motorUnitForce(s.Force) &&
		motorUnitFinite(s.TauM, s.Adapt, s.TauAdapt, s.AAdapt, s.Gain, s.TwitchAmp, s.TauTwitch, s.ForceDecay, s.Dt) &&
		s.TauM > 0.0 &&
		s.TauAdapt > 0.0 &&
		s.TauTwitch > 0.0 &&
		s.Dt > 0.0 &&
		s.Gain >= 0.0 &&
		s.TwitchAmp >= 0.0 &&
		s.VReset < s.VThreshold
}

// Step advances the neuron by one timestep
func (s *MotorUnitState) Step(iExt float64) int {
	if !motorUnitFinite(iExt) || !s.valid() {
		return 0
	}

	force := s.Force * math.Exp(-s.Dt/s.TauTwitch)
	inputDrive := s.Gain*math.Max(0.0, iExt) - s.Adapt
	vTarget := s.VRest + inputDrive
	vCandidate, ok := motorUnitExactRelax(s.V, vTarget, s.TauM, s.Dt)
	if !ok || !motorUnitVoltage(vCandidate) {
		return 0
	}
	adaptTarget := s.AAdapt * (vCandidate - s.VRest)
	adaptCandidate, ok := motorUnitExactRelax(s.Adapt, adaptTarget, s.TauAdapt, s.Dt)
	if !ok || !motorUnitFinite(adaptCandidate) {
		return 0
	}

	spike := 0
	if vCandidate >= s.VThreshold {
		vCandidate = s.VReset
		force = math.Min(1.0, force+s.TwitchAmp)
		spike = 1
	}
	if !motorUnitVoltage(vCandidate) || !motorUnitForce(force) {
		return 0
	}

	s.V = vCandidate
	s.Adapt = adaptCandidate
	s.Force = force
	return spike
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
