// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for rulkov_map

package services

import (
	"errors"
	"math"
)

// RulkovMapNeuronState holds the neuron state
type RulkovMapNeuronState struct {
	X          float64
	Y          float64
	Alpha      float64
	Sigma      float64
	Mu         float64
	XThreshold float64
}

// NewRulkovMapNeuron creates a new RulkovMapNeuron neuron with default parameters
func NewRulkovMapNeuron() *RulkovMapNeuronState {
	return &RulkovMapNeuronState{
		X:          -1.0,
		Y:          -3.0,
		Alpha:      4.0,
		Sigma:      -1.6,
		Mu:         0.001,
		XThreshold: 0.0,
	}
}

func finiteRulkovMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// ValidateRulkovMap checks discrete-map state and numerical parameters.
func ValidateRulkovMap(s *RulkovMapNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteRulkovMap(s.X) &&
		finiteRulkovMap(s.Y) &&
		finiteRulkovMap(s.Alpha) && s.Alpha > 0.0 &&
		finiteRulkovMap(s.Sigma) &&
		finiteRulkovMap(s.Mu) && s.Mu > 0.0 &&
		finiteRulkovMap(s.XThreshold)
}

// Step advances the neuron by one timestep
func (s *RulkovMapNeuronState) Step(iExt float64) (int, error) {
	if !ValidateRulkovMap(s) {
		return 0, errors.New("invalid Rulkov map runtime state")
	}
	if !finiteRulkovMap(iExt) {
		return 0, errors.New("invalid Rulkov map current")
	}

	xPrev := s.X
	branchBoundary := s.Alpha + s.Y + iExt
	if !finiteRulkovMap(branchBoundary) {
		return 0, errors.New("invalid Rulkov map branch boundary")
	}
	xNew := -1.0
	if s.X <= 0.0 {
		denominator := 1.0 - s.X
		if denominator <= 0.0 || !finiteRulkovMap(denominator) {
			return 0, errors.New("invalid Rulkov map branch denominator")
		}
		xNew = s.Alpha/denominator + s.Y + iExt
	} else if s.X < branchBoundary {
		xNew = branchBoundary
	}
	yNew := s.Y - s.Mu*(s.X+1.0) + s.Mu*s.Sigma
	if !finiteRulkovMap(xNew) || !finiteRulkovMap(yNew) {
		return 0, errors.New("invalid Rulkov map candidate state")
	}

	s.X = xNew
	s.Y = yNew
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateRulkovMapNeuron runs the neuron for n steps
func SimulateRulkovMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewRulkovMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
