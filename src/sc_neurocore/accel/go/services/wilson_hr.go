// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wilson_hr

package services

import (
	"errors"
	"math"
)

type WilsonHRNeuronState struct {
	V     float64
	R     float64
	TauR  float64
	VPeak float64
	Dt    float64
}

func NewWilsonHRNeuron() *WilsonHRNeuronState {
	return &WilsonHRNeuronState{V: -0.7, R: 0.1, TauR: 1.9, VPeak: 0.4, Dt: 0.05}
}

func finiteWilsonHR(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func ValidateWilsonHR(s *WilsonHRNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteWilsonHR(s.V) && finiteWilsonHR(s.R) &&
		finiteWilsonHR(s.TauR) && s.TauR > 0.0 &&
		finiteWilsonHR(s.VPeak) && finiteWilsonHR(s.Dt) && s.Dt > 0.0
}

func wilsonHRPoly(v float64) float64 {
	return -(17.81 + 47.71*v + 32.63*v*v) * (v - 0.55)
}

func (s *WilsonHRNeuronState) derivatives(v, r, iExt float64) (float64, float64, bool) {
	if !finiteWilsonHR(v) || !finiteWilsonHR(r) || !finiteWilsonHR(iExt) {
		return 0, 0, false
	}
	poly := wilsonHRPoly(v)
	syn := -26.0 * r * (v + 0.92)
	dv := poly + syn + iExt
	dr := (-r + 1.35*v + 1.03) / s.TauR
	if !finiteWilsonHR(poly) || !finiteWilsonHR(syn) || !finiteWilsonHR(dv) || !finiteWilsonHR(dr) {
		return 0, 0, false
	}
	return dv, dr, true
}

func (s *WilsonHRNeuronState) rk4Candidate(iExt float64) (float64, float64, bool) {
	v0, r0, dt := s.V, s.R, s.Dt
	k1v, k1r, ok := s.derivatives(v0, r0, iExt)
	if !ok {
		return 0, 0, false
	}
	k2v, k2r, ok := s.derivatives(v0+0.5*dt*k1v, r0+0.5*dt*k1r, iExt)
	if !ok {
		return 0, 0, false
	}
	k3v, k3r, ok := s.derivatives(v0+0.5*dt*k2v, r0+0.5*dt*k2r, iExt)
	if !ok {
		return 0, 0, false
	}
	k4v, k4r, ok := s.derivatives(v0+dt*k3v, r0+dt*k3r, iExt)
	if !ok {
		return 0, 0, false
	}
	nextV := v0 + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextR := r0 + dt*(k1r+2.0*k2r+2.0*k3r+k4r)/6.0
	return nextV, nextR, finiteWilsonHR(nextV) && finiteWilsonHR(nextR)
}

func (s *WilsonHRNeuronState) Step(iExt float64) (int, error) {
	if !ValidateWilsonHR(s) {
		return 0, errors.New("invalid Wilson-HR runtime state")
	}
	if !finiteWilsonHR(iExt) {
		return 0, errors.New("invalid Wilson-HR external current")
	}
	nextV, nextR, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0, errors.New("invalid Wilson-HR candidate state")
	}
	s.V = nextV
	s.R = nextR
	if s.V >= s.VPeak {
		s.V = -0.7
		return 1, nil
	}
	return 0, nil
}

func SimulateWilsonHRNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWilsonHRNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
