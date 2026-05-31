// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for terman_wang

package services

import (
	"errors"
	"math"
)

// TermanWangOscillatorState holds the neuron state
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

// NewTermanWangOscillator creates a new TermanWangOscillator neuron with default parameters
func NewTermanWangOscillator() *TermanWangOscillatorState {
	return &TermanWangOscillatorState{
		V:       -1.5,
		W:       -0.5,
		Alpha:   3.0,
		Beta:    0.2,
		Epsilon: 0.02,
		Rho:     0.0,
		Dt:      0.05,
		VPeak:   1.5,
	}
}

// ValidateTermanWang checks the mutable runtime state and parameters.
func ValidateTermanWang(s *TermanWangOscillatorState) bool {
	if s == nil {
		return false
	}
	return math.IsInf(s.V, 0) == false &&
		math.IsInf(s.W, 0) == false &&
		math.IsInf(s.Alpha, 0) == false &&
		math.IsInf(s.Beta, 0) == false &&
		math.IsInf(s.Epsilon, 0) == false &&
		math.IsInf(s.Rho, 0) == false &&
		math.IsInf(s.Dt, 0) == false &&
		math.IsInf(s.VPeak, 0) == false &&
		!math.IsNaN(s.V) &&
		!math.IsNaN(s.W) &&
		!math.IsNaN(s.Alpha) &&
		!math.IsNaN(s.Beta) &&
		!math.IsNaN(s.Epsilon) &&
		!math.IsNaN(s.Rho) &&
		!math.IsNaN(s.Dt) &&
		!math.IsNaN(s.VPeak) &&
		s.Beta > 0 &&
		s.Epsilon > 0 &&
		s.Dt > 0
}

// Step advances the neuron by one timestep
func (s *TermanWangOscillatorState) Step(iExt float64) (int, error) {
	if !ValidateTermanWang(s) {
		return 0, errors.New("invalid Terman-Wang runtime state")
	}
	if math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0, errors.New("invalid Terman-Wang external current")
	}

	f := 3.0*s.V - s.V*s.V*s.V + 2.0
	g := s.Alpha * (1.0 + math.Tanh(s.V/s.Beta))
	dv := (f - s.W + iExt + s.Rho) * s.Dt
	dw := s.Epsilon * (g - s.W) * s.Dt
	nextV := s.V + dv
	nextW := s.W + dw
	if math.IsNaN(dv) || math.IsInf(dv, 0) ||
		math.IsNaN(dw) || math.IsInf(dw, 0) ||
		math.IsNaN(nextV) || math.IsInf(nextV, 0) ||
		math.IsNaN(nextW) || math.IsInf(nextW, 0) {
		return 0, errors.New("non-finite Terman-Wang update")
	}

	vPrev := s.V
	s.V = nextV
	s.W = nextW
	if s.V >= s.VPeak && vPrev < s.VPeak {
		return 1, nil
	}
	return 0, nil
}

// SimulateTermanWangOscillator runs the neuron for n steps
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
