// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Terman-Wang

package services

import (
	"errors"
	"math"
)

// TermanWangOscillatorState holds the neuron state.
type TermanWangOscillatorState struct {
	V       float64
	W       float64
	Alpha   float64
	Beta    float64
	Epsilon float64
	Rho     float64
	Dt      float64
	VPeak   float64
}

// NewTermanWangOscillator creates a new Terman-Wang oscillator with default parameters.
func NewTermanWangOscillator() *TermanWangOscillatorState {
	return &TermanWangOscillatorState{V: -1.5, W: -0.5, Alpha: 3.0, Beta: 0.2, Epsilon: 0.02, Rho: 0.0, Dt: 0.05, VPeak: 1.5}
}

func isFiniteTermanWang(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// ValidateTermanWang checks the mutable runtime state and parameters.
func ValidateTermanWang(s *TermanWangOscillatorState) bool {
	return s != nil &&
		isFiniteTermanWang(s.V) && isFiniteTermanWang(s.W) &&
		isFiniteTermanWang(s.Alpha) && isFiniteTermanWang(s.Beta) && s.Beta > 0 &&
		isFiniteTermanWang(s.Epsilon) && s.Epsilon > 0 &&
		isFiniteTermanWang(s.Rho) && isFiniteTermanWang(s.Dt) && s.Dt > 0 &&
		isFiniteTermanWang(s.VPeak)
}

func (s *TermanWangOscillatorState) derivatives(v, w, current float64) (float64, float64, bool) {
	if !isFiniteTermanWang(v) || !isFiniteTermanWang(w) || !isFiniteTermanWang(current) {
		return 0, 0, false
	}
	f := 3.0*v - v*v*v + 2.0
	g := s.Alpha * (1.0 + math.Tanh(v/s.Beta))
	dv := f - w + current + s.Rho
	dw := s.Epsilon * (g - w)
	if !isFiniteTermanWang(dv) || !isFiniteTermanWang(dw) {
		return 0, 0, false
	}
	return dv, dw, true
}

func (s *TermanWangOscillatorState) rk4Candidate(current float64) (float64, float64, bool) {
	dt := s.Dt
	k1v, k1w, ok := s.derivatives(s.V, s.W, current)
	if !ok {
		return 0, 0, false
	}
	k2v, k2w, ok := s.derivatives(s.V+0.5*dt*k1v, s.W+0.5*dt*k1w, current)
	if !ok {
		return 0, 0, false
	}
	k3v, k3w, ok := s.derivatives(s.V+0.5*dt*k2v, s.W+0.5*dt*k2w, current)
	if !ok {
		return 0, 0, false
	}
	k4v, k4w, ok := s.derivatives(s.V+dt*k3v, s.W+dt*k3w, current)
	if !ok {
		return 0, 0, false
	}
	v := s.V + dt*(k1v+2*k2v+2*k3v+k4v)/6.0
	w := s.W + dt*(k1w+2*k2w+2*k3w+k4w)/6.0
	if !isFiniteTermanWang(v) || !isFiniteTermanWang(w) {
		return 0, 0, false
	}
	return v, w, true
}

// Step advances the neuron by one timestep.
func (s *TermanWangOscillatorState) Step(iExt float64) (int, error) {
	if !ValidateTermanWang(s) {
		return 0, errors.New("invalid Terman-Wang runtime state")
	}
	if !isFiniteTermanWang(iExt) {
		return 0, errors.New("invalid Terman-Wang external current")
	}
	vPrev := s.V
	v, w, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0, errors.New("non-finite Terman-Wang RK4 candidate")
	}
	s.V, s.W = v, w
	if s.V >= s.VPeak && vPrev < s.VPeak {
		return 1, nil
	}
	return 0, nil
}

// SimulateTermanWangOscillator runs the neuron for n steps.
func SimulateTermanWangOscillator(nSteps int, iExt float64) ([]float64, int) {
	s := NewTermanWangOscillator()
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
