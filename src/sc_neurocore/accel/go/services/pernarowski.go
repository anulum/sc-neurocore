// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Pernarowski

package services

import "math"

// PernarowskiNeuronState holds the neuron state.
type PernarowskiNeuronState struct {
	V          float64
	W          float64
	Z          float64
	Alpha      float64
	Beta       float64
	Eps1       float64
	Eps2       float64
	Gamma      float64
	Dt         float64
	VThreshold float64
}

// NewPernarowskiNeuron creates a new Pernarowski neuron with default parameters.
func NewPernarowskiNeuron() *PernarowskiNeuronState {
	return &PernarowskiNeuronState{
		V:          -1.0,
		W:          0.0,
		Z:          0.0,
		Alpha:      0.1,
		Beta:       0.5,
		Eps1:       0.1,
		Eps2:       0.001,
		Gamma:      0.5,
		Dt:         0.1,
		VThreshold: 0.5,
	}
}

func isFinitePernarowski(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// ValidatePernarowskiNeuron checks that the three-state burster contract is finite.
func ValidatePernarowskiNeuron(s *PernarowskiNeuronState) bool {
	return s != nil &&
		isFinitePernarowski(s.V) &&
		isFinitePernarowski(s.W) &&
		isFinitePernarowski(s.Z) &&
		isFinitePernarowski(s.Alpha) &&
		isFinitePernarowski(s.Beta) &&
		isFinitePernarowski(s.Eps1) && s.Eps1 > 0 &&
		isFinitePernarowski(s.Eps2) && s.Eps2 > 0 &&
		isFinitePernarowski(s.Gamma) && s.Gamma > 0 &&
		isFinitePernarowski(s.Dt) && s.Dt > 0 &&
		isFinitePernarowski(s.VThreshold)
}

func (s *PernarowskiNeuronState) derivatives(v, w, z, current float64) (float64, float64, float64, bool) {
	if !isFinitePernarowski(v) || !isFinitePernarowski(w) || !isFinitePernarowski(z) || !isFinitePernarowski(current) {
		return 0, 0, 0, false
	}
	dv := v - math.Pow(v, 3)/3.0 - w - z + current
	dw := s.Eps1 * (v - s.Gamma*w + s.Alpha)
	dz := s.Eps2 * (s.Beta*(v+0.7) - z)
	if !isFinitePernarowski(dv) || !isFinitePernarowski(dw) || !isFinitePernarowski(dz) {
		return 0, 0, 0, false
	}
	return dv, dw, dz, true
}

func (s *PernarowskiNeuronState) rk4Candidate(current float64) (float64, float64, float64, bool) {
	dt := s.Dt
	k1v, k1w, k1z, ok := s.derivatives(s.V, s.W, s.Z, current)
	if !ok {
		return 0, 0, 0, false
	}
	k2v, k2w, k2z, ok := s.derivatives(s.V+0.5*dt*k1v, s.W+0.5*dt*k1w, s.Z+0.5*dt*k1z, current)
	if !ok {
		return 0, 0, 0, false
	}
	k3v, k3w, k3z, ok := s.derivatives(s.V+0.5*dt*k2v, s.W+0.5*dt*k2w, s.Z+0.5*dt*k2z, current)
	if !ok {
		return 0, 0, 0, false
	}
	k4v, k4w, k4z, ok := s.derivatives(s.V+dt*k3v, s.W+dt*k3w, s.Z+dt*k3z, current)
	if !ok {
		return 0, 0, 0, false
	}
	v := s.V + dt*(k1v+2*k2v+2*k3v+k4v)/6.0
	w := s.W + dt*(k1w+2*k2w+2*k3w+k4w)/6.0
	z := s.Z + dt*(k1z+2*k2z+2*k3z+k4z)/6.0
	if !isFinitePernarowski(v) || !isFinitePernarowski(w) || !isFinitePernarowski(z) {
		return 0, 0, 0, false
	}
	return v, w, z, true
}

// Step advances the neuron by one timestep.
func (s *PernarowskiNeuronState) Step(iExt float64) int {
	if !ValidatePernarowskiNeuron(s) || !isFinitePernarowski(iExt) {
		return 0
	}

	vPrev := s.V
	v, w, z, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V, s.W, s.Z = v, w, z
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePernarowskiNeuron runs the neuron for n steps.
func SimulatePernarowskiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPernarowskiNeuron()
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
