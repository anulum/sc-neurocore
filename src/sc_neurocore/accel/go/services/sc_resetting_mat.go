// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go SC resetting-MAT service

package services

import "math"

const scResettingMATVMin = -200.0
const scResettingMATVMax = 100.0
const scResettingMATThetaMax = 1.0e9

// SCResettingMATNeuronState contains the historical SC RK4/reset recurrence.
type SCResettingMATNeuronState struct {
	V              float64
	Theta1         float64
	Theta2         float64
	VRest          float64
	VReset         float64
	VThresholdBase float64
	TauM           float64
	Tau1           float64
	Tau2           float64
	H1             float64
	H2             float64
	Resistance     float64
	Dt             float64
}

// NewSCResettingMATNeuron returns the preserved SC defaults.
func NewSCResettingMATNeuron() *SCResettingMATNeuronState {
	return &SCResettingMATNeuronState{
		V: -70.0, Theta1: 0.0, Theta2: 0.0, VRest: -70.0, VReset: -70.0,
		VThresholdBase: -50.0, TauM: 10.0, Tau1: 10.0, Tau2: 200.0,
		H1: 5.0, H2: 3.0, Resistance: 1.0, Dt: 1.0,
	}
}

func scResettingMATFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func (s *SCResettingMATNeuronState) validate() bool {
	for _, value := range []float64{s.V, s.Theta1, s.Theta2, s.VRest, s.VReset, s.VThresholdBase, s.TauM, s.Tau1, s.Tau2, s.H1, s.H2, s.Resistance, s.Dt} {
		if !scResettingMATFinite(value) {
			return false
		}
	}
	return s.V >= scResettingMATVMin && s.V <= scResettingMATVMax &&
		s.VReset >= scResettingMATVMin && s.VReset <= scResettingMATVMax &&
		s.Theta1 >= 0.0 && s.Theta1 <= scResettingMATThetaMax &&
		s.Theta2 >= 0.0 && s.Theta2 <= scResettingMATThetaMax &&
		s.H1 >= 0.0 && s.H1 <= scResettingMATThetaMax &&
		s.H2 >= 0.0 && s.H2 <= scResettingMATThetaMax &&
		s.TauM > 0.0 && s.Tau1 > 0.0 && s.Tau2 > 0.0 && s.Resistance > 0.0 && s.Dt > 0.0
}

func (s *SCResettingMATNeuronState) derivatives(v, theta1, theta2, current float64) (float64, float64, float64) {
	return (-(v - s.VRest) + s.Resistance*current) / s.TauM, -theta1 / s.Tau1, -theta2 / s.Tau2
}

func (s *SCResettingMATNeuronState) candidate(current float64) (float64, float64, float64) {
	k1v, k1t1, k1t2 := s.derivatives(s.V, s.Theta1, s.Theta2, current)
	k2v, k2t1, k2t2 := s.derivatives(s.V+0.5*s.Dt*k1v, s.Theta1+0.5*s.Dt*k1t1, s.Theta2+0.5*s.Dt*k1t2, current)
	k3v, k3t1, k3t2 := s.derivatives(s.V+0.5*s.Dt*k2v, s.Theta1+0.5*s.Dt*k2t1, s.Theta2+0.5*s.Dt*k2t2, current)
	k4v, k4t1, k4t2 := s.derivatives(s.V+s.Dt*k3v, s.Theta1+s.Dt*k3t1, s.Theta2+s.Dt*k3t2, current)
	scale := s.Dt / 6.0
	return s.V + scale*(k1v+2.0*k2v+2.0*k3v+k4v),
		s.Theta1 + scale*(k1t1+2.0*k2t1+2.0*k3t1+k4t1),
		s.Theta2 + scale*(k1t2+2.0*k2t2+2.0*k3t2+k4t2)
}

// Step advances one atomic candidate-first RK4/reset step.
// Invalid input or state returns -1 without mutation.
func (s *SCResettingMATNeuronState) Step(current float64) int {
	if !scResettingMATFinite(current) || !s.validate() {
		return -1
	}
	v, theta1, theta2 := s.candidate(current)
	if !scResettingMATFinite(v) || !scResettingMATFinite(theta1) || !scResettingMATFinite(theta2) || v < scResettingMATVMin || v > scResettingMATVMax || theta1 < 0.0 || theta1 > scResettingMATThetaMax || theta2 < 0.0 || theta2 > scResettingMATThetaMax {
		return -1
	}
	spike := v >= s.VThresholdBase+theta1+theta2
	if spike {
		theta1 += s.H1
		theta2 += s.H2
		if theta1 > scResettingMATThetaMax || theta2 > scResettingMATThetaMax {
			return -1
		}
		v = s.VReset
	}
	s.V, s.Theta1, s.Theta2 = v, theta1, theta2
	if spike {
		return 1
	}
	return 0
}

// Reset clears dynamic state while preserving configuration.
func (s *SCResettingMATNeuronState) Reset() {
	s.V, s.Theta1, s.Theta2 = s.VRest, 0.0, 0.0
}

// SimulateSCResettingMATNeuron runs a constant-current SC trace.
func SimulateSCResettingMATNeuron(nSteps int, current float64) ([]float64, int) {
	state := NewSCResettingMATNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for index := range trace {
		result := state.Step(current)
		trace[index] = state.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
