// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service mirror for the SC two-state chaotic map

package services

import (
	"errors"
	"math"
)

// SCChaoticMapNeuronState holds the two project-defined map states.
type SCChaoticMapNeuronState struct {
	X          float64
	Y          float64
	KF         float64
	KS         float64
	Alpha      float64
	Delta      float64
	XThreshold float64
}

// NewSCChaoticMapNeuron creates the preserved SC engineering model.
func NewSCChaoticMapNeuron() *SCChaoticMapNeuronState {
	return &SCChaoticMapNeuronState{
		X: 0.0, Y: 0.0, KF: 0.7, KS: 0.95, Alpha: 2.0, Delta: 0.05, XThreshold: 0.5,
	}
}

func finiteSCChaoticMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func logisticSCChaoticMap(z float64) float64 {
	if z >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	expZ := math.Exp(z)
	return expZ / (1.0 + expZ)
}

// ValidateSCChaoticMap checks state and numerical parameters.
func ValidateSCChaoticMap(s *SCChaoticMapNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteSCChaoticMap(s.X) && finiteSCChaoticMap(s.Y) &&
		finiteSCChaoticMap(s.KF) && s.KF >= 0.0 &&
		finiteSCChaoticMap(s.KS) && finiteSCChaoticMap(s.Alpha) &&
		finiteSCChaoticMap(s.Delta) && s.Delta >= 0.0 &&
		finiteSCChaoticMap(s.XThreshold)
}

// Step advances both states simultaneously and returns an upward crossing.
func (s *SCChaoticMapNeuronState) Step(current float64) (int, error) {
	if !ValidateSCChaoticMap(s) {
		return 0, errors.New("invalid SC chaotic map runtime state")
	}
	if !finiteSCChaoticMap(current) {
		return 0, errors.New("invalid SC chaotic map current")
	}

	xPrev := s.X
	xNew := s.KF*s.X*logisticSCChaoticMap(s.X+s.Alpha) - s.Y + current
	yNew := s.KS*s.Y + s.Delta*s.X
	if !finiteSCChaoticMap(xNew) || !finiteSCChaoticMap(yNew) {
		return 0, errors.New("invalid SC chaotic map candidate state")
	}
	s.X = math.Min(10.0, math.Max(-10.0, xNew))
	s.Y = math.Min(10.0, math.Max(-10.0, yNew))
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateSCChaoticMapNeuron runs the map for n steps.
func SimulateSCChaoticMapNeuron(nSteps int, current float64) ([]float64, int) {
	s := NewSCChaoticMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(current)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.X
		spikes += result
	}
	return trace, spikes
}
