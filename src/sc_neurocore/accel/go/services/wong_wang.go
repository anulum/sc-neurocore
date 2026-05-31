// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wong_wang

package services

import (
	"errors"
	"math"
)

// WongWangUnitState holds the neuron state
type WongWangUnitState struct {
	S1     float64
	S2     float64
	TauS   float64
	Gamma  float64
	JN     float64
	JCross float64
	I0     float64
	Sigma  float64
	Dt     float64
}

// NewWongWangUnit creates a new WongWangUnit neuron with default parameters
func NewWongWangUnit() *WongWangUnitState {
	return &WongWangUnitState{
		S1:     0.1,
		S2:     0.1,
		TauS:   0.1,
		Gamma:  0.641,
		JN:     0.2609,
		JCross: 0.0497,
		I0:     0.3255,
		Sigma:  0.02,
		Dt:     0.001,
	}
}

func finiteWongWang(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteWongWangGate(value float64) bool {
	return finiteWongWang(value) && value >= 0.0 && value <= 1.0
}

// ValidateWongWang checks the two-pool state and numerical parameters.
func ValidateWongWang(s *WongWangUnitState) bool {
	if s == nil {
		return false
	}
	return finiteWongWangGate(s.S1) &&
		finiteWongWangGate(s.S2) &&
		finiteWongWang(s.TauS) && s.TauS > 0.0 &&
		finiteWongWang(s.Gamma) && s.Gamma > 0.0 &&
		finiteWongWang(s.JN) && s.JN >= 0.0 &&
		finiteWongWang(s.JCross) && s.JCross >= 0.0 &&
		finiteWongWang(s.I0) &&
		finiteWongWang(s.Sigma) && s.Sigma >= 0.0 &&
		finiteWongWang(s.Dt) && s.Dt > 0.0
}

func phiWongWang(iSyn float64) (float64, error) {
	if !finiteWongWang(iSyn) {
		return 0, errors.New("invalid Wong-Wang synaptic current")
	}
	x := 270.0*iSyn - 108.0
	if math.Abs(x) < 1.0e-6 {
		return 1.0 / 0.154, nil
	}
	exponent := -0.154 * x
	if exponent > 700.0 {
		return 0.0, nil
	}
	response := x / (1.0 - math.Exp(exponent))
	if !finiteWongWang(response) || response < 0.0 {
		return 0, errors.New("invalid Wong-Wang transfer response")
	}
	return response, nil
}

// Step advances the neuron by one timestep
func (s *WongWangUnitState) Step(stim1 float64, stim2 float64, xi1 float64, xi2 float64) (float64, float64, error) {
	if !ValidateWongWang(s) {
		return 0, 0, errors.New("invalid Wong-Wang runtime state")
	}
	if !finiteWongWang(stim1) || !finiteWongWang(stim2) || !finiteWongWang(xi1) || !finiteWongWang(xi2) {
		return 0, 0, errors.New("invalid Wong-Wang stimuli or noise")
	}

	i1 := s.JN*s.S1 - s.JCross*s.S2 + s.I0 + stim1 + s.Sigma*xi1
	i2 := s.JN*s.S2 - s.JCross*s.S1 + s.I0 + stim2 + s.Sigma*xi2
	r1, err := phiWongWang(i1)
	if err != nil {
		return 0, 0, err
	}
	r2, err := phiWongWang(i2)
	if err != nil {
		return 0, 0, err
	}
	nextS1 := s.S1 + (-s.S1/s.TauS+(1.0-s.S1)*s.Gamma*r1)*s.Dt
	nextS2 := s.S2 + (-s.S2/s.TauS+(1.0-s.S2)*s.Gamma*r2)*s.Dt
	if !finiteWongWang(nextS1) || !finiteWongWang(nextS2) {
		return 0, 0, errors.New("invalid Wong-Wang candidate state")
	}
	s.S1 = math.Min(1.0, math.Max(0.0, nextS1))
	s.S2 = math.Min(1.0, math.Max(0.0, nextS2))
	return r1, r2, nil
}

// SimulateWongWangUnit runs the neuron for n steps
func SimulateWongWangUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewWongWangUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		_, _, err := s.Step(iExt, 0.0, 0.0, 0.0)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.S1
	}
	return trace, spikes
}
