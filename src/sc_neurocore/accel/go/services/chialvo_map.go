// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Checked Go service for the Chialvo map

package services

import (
	"errors"
	"math"
)

// ChialvoMapNeuronState holds the two state variables and map parameters.
type ChialvoMapNeuronState struct {
	X          float64
	Y          float64
	A          float64
	B          float64
	C          float64
	K          float64
	XThreshold float64
}

// NewChialvoMapNeuron returns the source-paper parameter set used by Python.
func NewChialvoMapNeuron() *ChialvoMapNeuronState {
	return &ChialvoMapNeuronState{
		X:          0.0,
		Y:          0.0,
		A:          0.89,
		B:          0.6,
		C:          0.28,
		K:          0.04,
		XThreshold: 1.0,
	}
}

func finiteChialvoMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func safeExpChialvoMap(value float64) float64 {
	return math.Exp(math.Min(500.0, math.Max(-500.0, value)))
}

// ValidateChialvoMap checks that every state and parameter field is finite.
func ValidateChialvoMap(state *ChialvoMapNeuronState) bool {
	if state == nil {
		return false
	}
	return finiteChialvoMap(state.X) &&
		finiteChialvoMap(state.Y) &&
		finiteChialvoMap(state.A) &&
		finiteChialvoMap(state.B) &&
		finiteChialvoMap(state.C) &&
		finiteChialvoMap(state.K) &&
		finiteChialvoMap(state.XThreshold)
}

// Step advances one simultaneous map iteration under an additive perturbation.
func (state *ChialvoMapNeuronState) Step(current float64) (int, error) {
	if !ValidateChialvoMap(state) {
		return 0, errors.New("invalid Chialvo map runtime state")
	}
	if !finiteChialvoMap(current) {
		return 0, errors.New("invalid Chialvo map current")
	}

	xPrevious := state.X
	xSquared := state.X * state.X
	exponential := safeExpChialvoMap(state.Y - state.X)
	xNext := xSquared*exponential + state.K + current
	yNext := state.A*state.Y - state.B*state.X + state.C
	if !finiteChialvoMap(xNext) || !finiteChialvoMap(yNext) {
		return 0, errors.New("invalid Chialvo map candidate state")
	}
	state.X = xNext
	state.Y = yNext
	if xPrevious < state.XThreshold && state.X >= state.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateComplete runs a failure-atomic batch and returns both state traces.
func (state *ChialvoMapNeuronState) SimulateComplete(
	nSteps int,
	current float64,
) ([]float64, []float64, int64, error) {
	if nSteps < 0 {
		return nil, nil, 0, errors.New("nSteps must be non-negative")
	}
	if !ValidateChialvoMap(state) || !finiteChialvoMap(current) {
		return nil, nil, 0, errors.New("invalid Chialvo map batch input")
	}
	candidate := *state
	xTrace := make([]float64, nSteps)
	yTrace := make([]float64, nSteps)
	var spikes int64
	for index := range xTrace {
		spiked, err := candidate.Step(current)
		if err != nil {
			return nil, nil, 0, err
		}
		xTrace[index] = candidate.X
		yTrace[index] = candidate.Y
		spikes += int64(spiked)
	}
	state.X = candidate.X
	state.Y = candidate.Y
	return xTrace, yTrace, spikes, nil
}

// Simulate runs the complete checked batch and returns its fast-state trace.
func (state *ChialvoMapNeuronState) Simulate(
	nSteps int,
	current float64,
) ([]float64, int64, error) {
	xTrace, _, spikes, err := state.SimulateComplete(nSteps, current)
	return xTrace, spikes, err
}

// Reset restores state variables while preserving configured parameters.
func (state *ChialvoMapNeuronState) Reset() {
	state.X = 0.0
	state.Y = 0.0
}

// SimulateChialvoMapNeuron runs the default model for compatibility callers.
func SimulateChialvoMapNeuron(nSteps int, current float64) ([]float64, int64, error) {
	return NewChialvoMapNeuron().Simulate(nSteps, current)
}
