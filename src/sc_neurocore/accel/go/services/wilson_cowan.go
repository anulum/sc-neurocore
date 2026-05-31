// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wilson_cowan

package services

import (
	"errors"
	"math"
)

// WilsonCowanUnitState holds the neuron state
type WilsonCowanUnitState struct {
	E     float64
	I     float64
	WEe   float64
	WEi   float64
	WIe   float64
	WIi   float64
	TauE  float64
	TauI  float64
	A     float64
	Theta float64
	Dt    float64
}

// NewWilsonCowanUnit creates a new WilsonCowanUnit neuron with default parameters
func NewWilsonCowanUnit() *WilsonCowanUnitState {
	return &WilsonCowanUnitState{
		E:     0.1,
		I:     0.05,
		WEe:   10.0,
		WEi:   6.0,
		WIe:   10.0,
		WIi:   1.0,
		TauE:  1.0,
		TauI:  2.0,
		A:     1.2,
		Theta: 4.0,
		Dt:    0.1,
	}
}

func finiteWilsonCowan(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func logisticWilsonCowan(z float64) float64 {
	if z >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	expZ := math.Exp(z)
	return expZ / (1.0 + expZ)
}

func finiteWilsonCowanRate(value float64, a float64, theta float64) bool {
	baseline := logisticWilsonCowan(-a * theta)
	return finiteWilsonCowan(value) && value >= -baseline && value <= 1.0-baseline
}

func sigmoidWilsonCowan(a float64, theta float64, x float64) (float64, error) {
	if !finiteWilsonCowan(a) || !finiteWilsonCowan(theta) || !finiteWilsonCowan(x) {
		return 0, errors.New("invalid Wilson-Cowan sigmoid input")
	}
	return logisticWilsonCowan(a*(x-theta)) - logisticWilsonCowan(-a*theta), nil
}

// ValidateWilsonCowan checks the runtime rate state and numerical parameters.
func ValidateWilsonCowan(s *WilsonCowanUnitState) bool {
	if s == nil {
		return false
	}
	return finiteWilsonCowan(s.WEe) && s.WEe >= 0.0 &&
		finiteWilsonCowan(s.WEi) && s.WEi >= 0.0 &&
		finiteWilsonCowan(s.WIe) && s.WIe >= 0.0 &&
		finiteWilsonCowan(s.WIi) && s.WIi >= 0.0 &&
		finiteWilsonCowan(s.TauE) && s.TauE > 0.0 &&
		finiteWilsonCowan(s.TauI) && s.TauI > 0.0 &&
		finiteWilsonCowan(s.A) && s.A > 0.0 &&
		finiteWilsonCowan(s.Theta) &&
		finiteWilsonCowan(s.Dt) && s.Dt > 0.0 &&
		finiteWilsonCowanRate(s.E, s.A, s.Theta) &&
		finiteWilsonCowanRate(s.I, s.A, s.Theta)
}

// Step advances the neuron by one timestep
func (s *WilsonCowanUnitState) Step(iExt float64) (float64, error) {
	if !ValidateWilsonCowan(s) {
		return 0, errors.New("invalid Wilson-Cowan runtime state")
	}
	if !finiteWilsonCowan(iExt) {
		return 0, errors.New("invalid Wilson-Cowan external input")
	}

	se, err := sigmoidWilsonCowan(s.A, s.Theta, s.WEe*s.E-s.WEi*s.I+iExt)
	if err != nil {
		return 0, err
	}
	si, err := sigmoidWilsonCowan(s.A, s.Theta, s.WIe*s.E-s.WIi*s.I)
	if err != nil {
		return 0, err
	}
	nextE := s.E + (-s.E+se)/s.TauE*s.Dt
	nextI := s.I + (-s.I+si)/s.TauI*s.Dt
	if !finiteWilsonCowanRate(nextE, s.A, s.Theta) || !finiteWilsonCowanRate(nextI, s.A, s.Theta) {
		return 0, errors.New("invalid Wilson-Cowan candidate state")
	}

	s.E = nextE
	s.I = nextI
	return s.E, nil
}

// SimulateWilsonCowanUnit runs the neuron for n steps
func SimulateWilsonCowanUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewWilsonCowanUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		_, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.E
	}
	return trace, spikes
}
