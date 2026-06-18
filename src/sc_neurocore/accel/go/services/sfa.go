// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for sfa

package services

import (
	"errors"
	"math"
)

const sfaVMin = -200.0
const sfaVMax = 100.0
const sfaGMax = 1.0e9

// SFANeuronState holds the neuron state
type SFANeuronState struct {
	V          float64
	GSfa       float64
	VRest      float64
	VReset     float64
	VThreshold float64
	TauM       float64
	TauSfa     float64
	DeltaG     float64
	EK         float64
	Resistance float64
	Dt         float64
}

// NewSFANeuron creates a new SFANeuron neuron with default parameters
func NewSFANeuron() *SFANeuronState {
	return &SFANeuronState{
		V:          -70.0,
		GSfa:       0.0,
		VRest:      -70.0,
		VReset:     -70.0,
		VThreshold: -50.0,
		TauM:       10.0,
		TauSfa:     200.0,
		DeltaG:     0.5,
		EK:         -80.0,
		Resistance: 1.0,
		Dt:         1.0,
	}
}

func sfaFinite(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func sfaNonnegative(value float64) bool { return sfaFinite(value) && value >= 0.0 }

func (s *SFANeuronState) validate() error {
	for _, value := range []float64{s.V, s.VRest, s.VReset, s.VThreshold, s.EK} {
		if !sfaFinite(value) {
			return errors.New("finite SFA voltage parameter invalid")
		}
	}
	if s.V < sfaVMin || s.V > sfaVMax || s.VReset < sfaVMin || s.VReset > sfaVMax {
		return errors.New("SFA voltage outside safety envelope")
	}
	if !sfaNonnegative(s.GSfa) || s.GSfa > sfaGMax {
		return errors.New("SFA adaptation conductance outside safety envelope")
	}
	for _, value := range []float64{s.TauM, s.TauSfa, s.Resistance, s.Dt} {
		if !sfaFinite(value) || value <= 0.0 {
			return errors.New("positive SFA parameter invalid")
		}
	}
	if !sfaNonnegative(s.DeltaG) || s.DeltaG > sfaGMax {
		return errors.New("SFA adaptation increment outside safety envelope")
	}
	return nil
}

func (s *SFANeuronState) derivatives(v, gSfa, iExt float64) (float64, float64) {
	dv := (-(v - s.VRest) - gSfa*(v-s.EK) + s.Resistance*iExt) / s.TauM
	return dv, -gSfa / s.TauSfa
}

func (s *SFANeuronState) rk4Candidate(v, gSfa, iExt float64) (float64, float64) {
	k1v, k1g := s.derivatives(v, gSfa, iExt)
	k2v, k2g := s.derivatives(v+0.5*s.Dt*k1v, gSfa+0.5*s.Dt*k1g, iExt)
	k3v, k3g := s.derivatives(v+0.5*s.Dt*k2v, gSfa+0.5*s.Dt*k2g, iExt)
	k4v, k4g := s.derivatives(v+s.Dt*k3v, gSfa+s.Dt*k3g, iExt)
	return v + (s.Dt/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v),
		gSfa + (s.Dt/6.0)*(k1g+2.0*k2g+2.0*k3g+k4g)
}

// Step advances the neuron by one timestep
func (s *SFANeuronState) Step(iExt float64) int {
	if !sfaFinite(iExt) || s.validate() != nil {
		return -1
	}
	vCandidate, gCandidate := s.rk4Candidate(s.V, s.GSfa, iExt)
	if !sfaFinite(vCandidate) || !sfaFinite(gCandidate) {
		return -1
	}
	if vCandidate < sfaVMin || vCandidate > sfaVMax || gCandidate < 0.0 || gCandidate > sfaGMax {
		return -1
	}
	if vCandidate >= s.VThreshold {
		gAfterSpike := gCandidate + s.DeltaG
		if !sfaFinite(gAfterSpike) || gAfterSpike > sfaGMax {
			return -1
		}
		s.V = s.VReset
		s.GSfa = gAfterSpike
		return 1
	}
	s.V = vCandidate
	s.GSfa = gCandidate
	return 0
}

// SimulateSFANeuron runs the neuron for n steps
func SimulateSFANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSFANeuron()
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
