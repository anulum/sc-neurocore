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

// WilsonHRNeuronState holds the neuron state
type WilsonHRNeuronState struct {
	V     float64
	R     float64
	TauR  float64
	VPeak float64
	Dt    float64
}

// NewWilsonHRNeuron creates a new WilsonHRNeuron neuron with default parameters
func NewWilsonHRNeuron() *WilsonHRNeuronState {
	return &WilsonHRNeuronState{
		V:     -0.7,
		R:     0.1,
		TauR:  1.9,
		VPeak: 0.4,
		Dt:    0.05,
	}
}

func finiteWilsonHR(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// ValidateWilsonHR checks runtime state and numerical parameters.
func ValidateWilsonHR(s *WilsonHRNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteWilsonHR(s.V) &&
		finiteWilsonHR(s.R) &&
		finiteWilsonHR(s.TauR) && s.TauR > 0.0 &&
		finiteWilsonHR(s.VPeak) &&
		finiteWilsonHR(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep
func (s *WilsonHRNeuronState) Step(iExt float64) (int, error) {
	if !ValidateWilsonHR(s) {
		return 0, errors.New("invalid Wilson-HR runtime state")
	}
	if !finiteWilsonHR(iExt) {
		return 0, errors.New("invalid Wilson-HR external current")
	}

	poly := -(17.81 + 47.71*s.V + 32.63*s.V*s.V) * (s.V - 0.55)
	syn := -26.0 * s.R * (s.V + 0.92)
	dv := (poly + syn + iExt) * s.Dt
	dr := (-s.R + 1.35*s.V + 1.03) / s.TauR * s.Dt
	nextV := s.V + dv
	nextR := s.R + dr
	if !finiteWilsonHR(poly) || !finiteWilsonHR(syn) ||
		!finiteWilsonHR(dv) || !finiteWilsonHR(dr) ||
		!finiteWilsonHR(nextV) || !finiteWilsonHR(nextR) {
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

// SimulateWilsonHRNeuron runs the neuron for n steps
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
