// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service mirror for Wong-Wang 2006

package services

import (
	"errors"
	"math"
)

// WongWangUnitState contains the four dynamic states and published parameters.
type WongWangUnitState struct {
	S1      float64
	S2      float64
	Noise1  float64
	Noise2  float64
	TauS    float64
	TauAMPA float64
	Gamma   float64
	JN      float64
	JCross  float64
	I0      float64
	Sigma   float64
	Dt      float64
}

// NewWongWangUnit returns the published reduced-model defaults.
func NewWongWangUnit() *WongWangUnitState {
	return &WongWangUnitState{
		S1: 0.1, S2: 0.1, Noise1: 0.0, Noise2: 0.0,
		TauS: 0.1, TauAMPA: 0.002, Gamma: 0.641,
		JN: 0.2609, JCross: 0.0497, I0: 0.3255, Sigma: 0.02, Dt: 0.0001,
	}
}

func finiteWongWang(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteWongWangGate(value float64) bool {
	return finiteWongWang(value) && value >= 0.0 && value <= 1.0
}

// ValidateWongWang checks state and parameter domains before an update.
func ValidateWongWang(state *WongWangUnitState) bool {
	return state != nil && finiteWongWangGate(state.S1) && finiteWongWangGate(state.S2) &&
		finiteWongWang(state.Noise1) && finiteWongWang(state.Noise2) &&
		finiteWongWang(state.TauS) && state.TauS > 0.0 &&
		finiteWongWang(state.TauAMPA) && state.TauAMPA > 0.0 &&
		finiteWongWang(state.Gamma) && state.Gamma > 0.0 &&
		finiteWongWang(state.JN) && state.JN >= 0.0 &&
		finiteWongWang(state.JCross) && state.JCross >= 0.0 &&
		finiteWongWang(state.I0) && finiteWongWang(state.Sigma) && state.Sigma >= 0.0 &&
		finiteWongWang(state.Dt) && state.Dt > 0.0
}

func phiWongWang(iSyn float64) (float64, error) {
	if !finiteWongWang(iSyn) {
		return 0.0, errors.New("invalid Wong-Wang synaptic current")
	}
	x := 270.0*iSyn - 108.0
	scaled := -0.154 * x
	response := 0.0
	if scaled > 700.0 {
		response = 0.0
	} else if math.Abs(x) < 1.0e-7 {
		response = 1.0 / 0.154
	} else {
		response = x / -math.Expm1(scaled)
	}
	if !finiteWongWang(response) {
		return 0.0, errors.New("invalid Wong-Wang transfer response")
	}
	return math.Max(0.0, response), nil
}

// Step advances one explicit-Euler NMDA and AMPA Ornstein-Uhlenbeck update.
func (state *WongWangUnitState) Step(
	stim1 float64,
	stim2 float64,
	xi1 float64,
	xi2 float64,
) (float64, float64, error) {
	if !ValidateWongWang(state) {
		return 0.0, 0.0, errors.New("invalid Wong-Wang runtime state")
	}
	if !finiteWongWang(stim1) || !finiteWongWang(stim2) ||
		!finiteWongWang(xi1) || !finiteWongWang(xi2) {
		return 0.0, 0.0, errors.New("invalid Wong-Wang stimulus or Gaussian sample")
	}
	rate1, err := phiWongWang(state.JN*state.S1 - state.JCross*state.S2 + state.I0 + stim1 + state.Noise1)
	if err != nil {
		return 0.0, 0.0, err
	}
	rate2, err := phiWongWang(state.JN*state.S2 - state.JCross*state.S1 + state.I0 + stim2 + state.Noise2)
	if err != nil {
		return 0.0, 0.0, err
	}
	noiseScale := math.Sqrt(state.Dt/state.TauAMPA) * state.Sigma
	noiseDecay := state.Dt / state.TauAMPA
	nextS1 := state.S1 + state.Dt*(-state.S1/state.TauS+(1.0-state.S1)*state.Gamma*rate1)
	nextS2 := state.S2 + state.Dt*(-state.S2/state.TauS+(1.0-state.S2)*state.Gamma*rate2)
	nextNoise1 := state.Noise1 - noiseDecay*state.Noise1 + noiseScale*xi1
	nextNoise2 := state.Noise2 - noiseDecay*state.Noise2 + noiseScale*xi2
	if !finiteWongWangGate(nextS1) || !finiteWongWangGate(nextS2) ||
		!finiteWongWang(nextNoise1) || !finiteWongWang(nextNoise2) {
		return 0.0, 0.0, errors.New("invalid Wong-Wang candidate state")
	}
	state.S1, state.S2 = nextS1, nextS2
	state.Noise1, state.Noise2 = nextNoise1, nextNoise2
	return rate1, rate2, nil
}

// Reset restores only dynamic states and preserves configured parameters.
func (state *WongWangUnitState) Reset() {
	state.S1, state.S2 = 0.1, 0.1
	state.Noise1, state.Noise2 = 0.0, 0.0
}

// SimulateWongWangUnit returns a deterministic zero-noise S1 trace.
func SimulateWongWangUnit(nSteps int, iExt float64) ([]float64, int) {
	state := NewWongWangUnit()
	state.Sigma = 0.0
	trace := make([]float64, nSteps)
	for step := 0; step < nSteps; step++ {
		if _, _, err := state.Step(iExt, 0.0, 0.0, 0.0); err != nil {
			trace[step] = math.NaN()
			continue
		}
		trace[step] = state.S1
	}
	return trace, 0
}
