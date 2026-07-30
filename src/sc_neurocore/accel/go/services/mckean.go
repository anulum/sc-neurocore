// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import "math"

// McKeanNeuronState is the source-bound space-clamped Heaviside system.
type McKeanNeuronState struct{ V, W, A, Lambda, Mu, B, Dt float64 }

// NewMcKeanNeuron constructs the normalized McKean/Tonnelier profile.
func NewMcKeanNeuron() *McKeanNeuronState { return &McKeanNeuronState{0, 0, .25, 1, 1, .01, .1} }
func (s *McKeanNeuronState) rhs(v, w, current float64) (float64, float64) {
	h := 0.0
	if v >= s.A {
		h = 1
	}
	return -s.Lambda*v + s.Mu*h - w + current, s.B * v
}
func (s *McKeanNeuronState) candidate(current float64) (float64, float64) {
	d := s.Dt
	k1v, k1w := s.rhs(s.V, s.W, current)
	k2v, k2w := s.rhs(s.V+d*k1v/2, s.W+d*k1w/2, current)
	k3v, k3w := s.rhs(s.V+d*k2v/2, s.W+d*k2w/2, current)
	k4v, k4w := s.rhs(s.V+d*k3v, s.W+d*k3w, current)
	return s.V + d*(k1v+2*k2v+2*k3v+k4v)/6, s.W + d*(k1w+2*k2w+2*k3w+k4w)/6
}

// Valid reports whether the source constraints and safety envelope hold.
func (s *McKeanNeuronState) Valid() bool {
	xs := []float64{s.V, s.W, s.A, s.Lambda, s.Mu, s.B, s.Dt}
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return math.Abs(s.V) <= 1e6 && math.Abs(s.W) <= 1e6 && s.A > 0 && s.Lambda > 0 && s.Mu > s.Lambda*s.A && s.B > 0 && s.Dt > 0 && s.Dt <= 1
}

// Step advances atomically and returns -1 on invalid input or state.
func (s *McKeanNeuronState) Step(current float64) int {
	if !s.Valid() || math.IsNaN(current) || math.IsInf(current, 0) {
		return -1
	}
	previous := s.V
	v, w := s.candidate(current)
	if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(w) || math.IsInf(w, 0) || math.Abs(v) > 1e6 || math.Abs(w) > 1e6 {
		return -1
	}
	s.V, s.W = v, w
	if previous < s.A && v >= s.A {
		return 1
	}
	return 0
}
