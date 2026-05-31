// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for McKean

package services

import "math"

// McKeanNeuronState holds the neuron state.
type McKeanNeuronState struct {
	V       float64
	W       float64
	A       float64
	Epsilon float64
	Gamma   float64
	Dt      float64
	VPeak   float64
}

// NewMcKeanNeuron creates a new McKean neuron with default parameters.
func NewMcKeanNeuron() *McKeanNeuronState {
	return &McKeanNeuronState{V: 0.0, W: 0.0, A: 0.25, Epsilon: 0.01, Gamma: 0.5, Dt: 0.1, VPeak: 0.8}
}

func isFiniteMcKean(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// ValidateMcKeanNeuron checks that the piecewise-linear dynamics are finite and well-formed.
func ValidateMcKeanNeuron(s *McKeanNeuronState) bool {
	return s != nil && isFiniteMcKean(s.V) && isFiniteMcKean(s.W) && isFiniteMcKean(s.A) && s.A > 0 && s.A < 1 && isFiniteMcKean(s.Epsilon) && s.Epsilon > 0 && isFiniteMcKean(s.Gamma) && s.Gamma > 0 && isFiniteMcKean(s.Dt) && s.Dt > 0 && isFiniteMcKean(s.VPeak)
}

func (s *McKeanNeuronState) f(v float64) float64 {
	mid1 := s.A / 2.0
	mid2 := (1.0 + s.A) / 2.0
	if v < mid1 {
		return -v
	}
	if v < mid2 {
		return v - s.A
	}
	return 1.0 - v
}

func (s *McKeanNeuronState) derivatives(v, w, current float64) (float64, float64, bool) {
	if !isFiniteMcKean(v) || !isFiniteMcKean(w) || !isFiniteMcKean(current) {
		return 0, 0, false
	}
	dv := s.f(v) - w + current
	dw := s.Epsilon * (v - s.Gamma*w)
	if !isFiniteMcKean(dv) || !isFiniteMcKean(dw) {
		return 0, 0, false
	}
	return dv, dw, true
}

func (s *McKeanNeuronState) rk4Candidate(current float64) (float64, float64, bool) {
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
	if !isFiniteMcKean(v) || !isFiniteMcKean(w) {
		return 0, 0, false
	}
	return v, w, true
}

// Step advances the neuron by one timestep.
func (s *McKeanNeuronState) Step(iExt float64) int {
	if !ValidateMcKeanNeuron(s) || !isFiniteMcKean(iExt) {
		return 0
	}
	vPrev := s.V
	v, w, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V, s.W = v, w
	if s.V >= s.VPeak && vPrev < s.VPeak {
		return 1
	}
	return 0
}

// SimulateMcKeanNeuron runs the neuron for n steps.
func SimulateMcKeanNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMcKeanNeuron()
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
