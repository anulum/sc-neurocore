// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for theta

package services

import (
	"errors"
	"math"
)

// ThetaNeuronState holds the neuron state.
type ThetaNeuronState struct {
	Theta float64
	Dt    float64
}

// NewThetaNeuron creates a new ThetaNeuron neuron with default parameters.
func NewThetaNeuron() *ThetaNeuronState {
	return &ThetaNeuronState{
		Theta: 0.0,
		Dt:    0.01,
	}
}

// Valid reports whether the state satisfies the theta phase contract.
func (s ThetaNeuronState) Valid() bool {
	return finiteTheta(s.Theta) && finiteTheta(s.Dt) && s.Dt > 0.0
}

func finiteTheta(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func wrapTheta(theta float64) float64 {
	return math.Mod(theta+math.Pi, 2.0*math.Pi) - math.Pi
}

func (s ThetaNeuronState) exactCandidate(iExt float64) (float64, bool) {
	y := math.Tan(s.Theta / 2.0)
	if iExt > 0.0 {
		rootI := math.Sqrt(iExt)
		phase := math.Atan(y / rootI)
		nextPhase := phase + rootI*s.Dt
		if math.Abs(math.Cos(nextPhase)) <= 1.0e-15 {
			return -math.Pi, nextPhase >= math.Pi/2.0
		}
		return wrapTheta(2.0 * math.Atan(rootI*math.Tan(nextPhase))), nextPhase >= math.Pi/2.0
	}
	if iExt == 0.0 {
		denominator := 1.0 - y*s.Dt
		if math.Abs(denominator) <= 1.0e-15 {
			return -math.Pi, true
		}
		return wrapTheta(2.0 * math.Atan(y/denominator)), denominator <= 0.0
	}

	rootI := math.Sqrt(-iExt)
	if math.Abs(y+rootI) <= 1.0e-15 {
		return s.Theta, false
	}
	ratio := (y - rootI) / (y + rootI)
	evolved := ratio * math.Exp(2.0*rootI*s.Dt)
	denominator := 1.0 - evolved
	spiked := (ratio < 1.0 && evolved >= 1.0) || math.Abs(denominator) <= 1.0e-15
	if spiked && math.Abs(denominator) <= 1.0e-15 {
		return -math.Pi, true
	}
	return wrapTheta(2.0 * math.Atan(rootI*(1.0+evolved)/denominator)), spiked
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *ThetaNeuronState) Step(iExt float64) (int, error) {
	if !finiteTheta(iExt) || !s.Valid() {
		return 0, ErrThetaInvalidState
	}

	nextTheta, spiked := s.exactCandidate(iExt)
	if !finiteTheta(nextTheta) {
		return 0, ErrThetaNonFiniteUpdate
	}

	s.Theta = wrapTheta(nextTheta)
	if spiked {
		return 1, nil
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *ThetaNeuronState) Reset() {
	s.Theta = 0.0
}

// SimulateThetaNeuron runs the neuron for n steps.
func SimulateThetaNeuron(nSteps int, iExt float64) ([]float64, int) {
	trace, spikes, _, err := SimulateThetaTrace(*NewThetaNeuron(), nSteps, iExt)
	if err != nil {
		panic(err)
	}
	return trace, spikes
}

// SimulateThetaTrace executes a complete phase and integration contract.
func SimulateThetaTrace(
	initial ThetaNeuronState,
	nSteps int,
	iExt float64,
) ([]float64, int, float64, error) {
	if nSteps < 0 || !finiteTheta(iExt) || !initial.Valid() {
		return nil, 0, initial.Theta, ErrThetaInvalidState
	}
	s := initial
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			return nil, 0, initial.Theta, err
		}
		trace[t] = s.Theta
		spikes += result
	}
	return trace, spikes, s.Theta, nil
}

var (
	ErrThetaInvalidState    = errors.New("theta state/current must be finite with positive dt")
	ErrThetaNonFiniteUpdate = errors.New("theta exact-flow update became non-finite")
)
