// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import "math"

// NonlinearLIFState mirrors the production nonlinear LIF state contract.
type NonlinearLIFState struct {
	V          float64
	W          float64
	VRest      float64
	VCrit      float64
	VThreshold float64
	VReset     float64
	A          float64
	B          float64
	TauW       float64
	CM         float64
	DT         float64
}

// DefaultNonlinearLIFState returns the canonical NLIF parameters.
func DefaultNonlinearLIFState() NonlinearLIFState {
	return NonlinearLIFState{
		V:          -65.0,
		W:          0.0,
		VRest:      -65.0,
		VCrit:      -40.0,
		VThreshold: -20.0,
		VReset:     -65.0,
		A:          0.04,
		B:          0.5,
		TauW:       100.0,
		CM:         1.0,
		DT:         0.1,
	}
}

func nlifFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Valid reports whether the state satisfies the production NLIF contract.
func (s NonlinearLIFState) Valid() bool {
	return nlifFinite(s.V, s.W, s.VRest, s.VCrit, s.VThreshold, s.VReset, s.A, s.B, s.TauW, s.CM, s.DT) &&
		s.VRest < s.VCrit &&
		s.VCrit < s.VThreshold &&
		s.VReset < s.VThreshold &&
		s.A >= 0.0 &&
		s.B >= 0.0 &&
		s.TauW > 0.0 &&
		s.CM > 0.0 &&
		s.DT > 0.0 &&
		s.DT <= s.TauW
}

func (s NonlinearLIFState) derivatives(v float64, w float64, current float64) (float64, float64) {
	nonlinear := s.A * (v - s.VRest) * (v - s.VCrit)
	dv := (nonlinear - w + current) / s.CM
	dw := (s.B*(v-s.VRest) - w) / s.TauW
	return dv, dw
}

func (s NonlinearLIFState) rk4Candidate(current float64) (float64, float64, bool) {
	k1v, k1w := s.derivatives(s.V, s.W, current)
	k2v, k2w := s.derivatives(s.V+0.5*s.DT*k1v, s.W+0.5*s.DT*k1w, current)
	k3v, k3w := s.derivatives(s.V+0.5*s.DT*k2v, s.W+0.5*s.DT*k2w, current)
	k4v, k4w := s.derivatives(s.V+s.DT*k3v, s.W+s.DT*k3w, current)
	nextV := s.V + (s.DT/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v)
	nextW := s.W + (s.DT/6.0)*(k1w+2.0*k2w+2.0*k3w+k4w)
	return nextV, nextW, nlifFinite(nextV, nextW)
}

// Step advances one candidate-first RK4 step and returns 1 on spike. Invalid inputs do not mutate state.
func (s *NonlinearLIFState) Step(current float64) int {
	if !nlifFinite(current) || !s.Valid() {
		return -1
	}

	nextV, nextW, ok := s.rk4Candidate(current)
	if !ok {
		return -1
	}
	s.V = nextV
	s.W = nextW
	if nextV >= s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// Reset restores dynamic state without changing parameters.
func (s *NonlinearLIFState) Reset() {
	s.V = s.VRest
	s.W = 0.0
}
