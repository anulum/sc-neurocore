// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mat

package services

import (
	"errors"
	"math"
)

const matVMin = -200.0
const matVMax = 100.0
const matThetaMax = 1.0e9

// MATNeuronState holds the neuron state
type MATNeuronState struct {
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

// NewMATNeuron creates a new MATNeuron neuron with default parameters
func NewMATNeuron() *MATNeuronState {
	return &MATNeuronState{
		V:              -70.0,
		Theta1:         0.0,
		Theta2:         0.0,
		VRest:          -70.0,
		VReset:         -70.0,
		VThresholdBase: -50.0,
		TauM:           10.0,
		Tau1:           10.0,
		Tau2:           200.0,
		H1:             5.0,
		H2:             3.0,
		Resistance:     1.0,
		Dt:             1.0,
	}
}

func matFinite(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func matNonnegative(value float64) bool { return matFinite(value) && value >= 0.0 }

func (s *MATNeuronState) validate() error {
	for _, value := range []float64{s.V, s.VRest, s.VReset, s.VThresholdBase} {
		if !matFinite(value) {
			return errors.New("finite MAT voltage parameter invalid")
		}
	}
	if s.V < matVMin || s.V > matVMax || s.VReset < matVMin || s.VReset > matVMax {
		return errors.New("MAT voltage outside safety envelope")
	}
	if !matNonnegative(s.Theta1) || !matNonnegative(s.Theta2) || s.Theta1 > matThetaMax || s.Theta2 > matThetaMax {
		return errors.New("MAT threshold adaptation outside safety envelope")
	}
	if !matNonnegative(s.H1) || !matNonnegative(s.H2) || s.H1 > matThetaMax || s.H2 > matThetaMax {
		return errors.New("MAT threshold increments outside safety envelope")
	}
	for _, value := range []float64{s.TauM, s.Tau1, s.Tau2, s.Resistance, s.Dt} {
		if !matFinite(value) || value <= 0.0 {
			return errors.New("positive MAT parameter invalid")
		}
	}
	return nil
}

func (s *MATNeuronState) derivatives(v, theta1, theta2, iExt float64) (float64, float64, float64) {
	dv := (-(v - s.VRest) + s.Resistance*iExt) / s.TauM
	return dv, -theta1 / s.Tau1, -theta2 / s.Tau2
}

func (s *MATNeuronState) rk4Candidate(v, theta1, theta2, iExt float64) (float64, float64, float64) {
	k1v, k1t1, k1t2 := s.derivatives(v, theta1, theta2, iExt)
	k2v, k2t1, k2t2 := s.derivatives(v+0.5*s.Dt*k1v, theta1+0.5*s.Dt*k1t1, theta2+0.5*s.Dt*k1t2, iExt)
	k3v, k3t1, k3t2 := s.derivatives(v+0.5*s.Dt*k2v, theta1+0.5*s.Dt*k2t1, theta2+0.5*s.Dt*k2t2, iExt)
	k4v, k4t1, k4t2 := s.derivatives(v+s.Dt*k3v, theta1+s.Dt*k3t1, theta2+s.Dt*k3t2, iExt)
	scale := s.Dt / 6.0
	return v + scale*(k1v+2.0*k2v+2.0*k3v+k4v),
		theta1 + scale*(k1t1+2.0*k2t1+2.0*k3t1+k4t1),
		theta2 + scale*(k1t2+2.0*k2t2+2.0*k3t2+k4t2)
}

// Step advances the neuron by one timestep
func (s *MATNeuronState) Step(iExt float64) int {
	if !matFinite(iExt) || s.validate() != nil {
		return -1
	}
	vCandidate, theta1Candidate, theta2Candidate := s.rk4Candidate(s.V, s.Theta1, s.Theta2, iExt)
	if !matFinite(vCandidate) || !matFinite(theta1Candidate) || !matFinite(theta2Candidate) {
		return -1
	}
	if vCandidate < matVMin || vCandidate > matVMax || theta1Candidate < 0.0 || theta1Candidate > matThetaMax || theta2Candidate < 0.0 || theta2Candidate > matThetaMax {
		return -1
	}
	threshold := s.VThresholdBase + theta1Candidate + theta2Candidate
	if vCandidate >= threshold {
		theta1AfterSpike := theta1Candidate + s.H1
		theta2AfterSpike := theta2Candidate + s.H2
		if !matFinite(theta1AfterSpike) || !matFinite(theta2AfterSpike) || theta1AfterSpike > matThetaMax || theta2AfterSpike > matThetaMax {
			return -1
		}
		s.V = s.VReset
		s.Theta1 = theta1AfterSpike
		s.Theta2 = theta2AfterSpike
		return 1
	}
	s.V = vCandidate
	s.Theta1 = theta1Candidate
	s.Theta2 = theta2Candidate
	return 0
}

// SimulateMATNeuron runs the neuron for n steps
func SimulateMATNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMATNeuron()
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
