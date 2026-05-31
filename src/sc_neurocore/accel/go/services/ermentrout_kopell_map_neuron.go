// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ermentrout_kopell_map_neuron

package services

import (
	"errors"
	"math"
)

// ErmentroutKopellMapNeuronState holds the neuron state
type ErmentroutKopellMapNeuronState struct {
	Theta          float64
	Dt             float64
	Gain           float64
	ThetaThreshold float64
}

// NewErmentroutKopellMapNeuron creates a new ErmentroutKopellMapNeuron neuron with default parameters
func NewErmentroutKopellMapNeuron() *ErmentroutKopellMapNeuronState {
	return &ErmentroutKopellMapNeuronState{
		Theta:          0.0,
		Dt:             0.1,
		Gain:           1.0,
		ThetaThreshold: math.Pi,
	}
}

func finiteErmentroutKopell(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// ValidateErmentroutKopell checks phase-map state and numerical parameters.
func ValidateErmentroutKopell(s *ErmentroutKopellMapNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteErmentroutKopell(s.Theta) &&
		finiteErmentroutKopell(s.Dt) && s.Dt > 0.0 &&
		finiteErmentroutKopell(s.Gain) &&
		finiteErmentroutKopell(s.ThetaThreshold)
}

// Step advances the neuron by one timestep
func (s *ErmentroutKopellMapNeuronState) Step(iExt float64) (int, error) {
	if !ValidateErmentroutKopell(s) {
		return 0, errors.New("invalid Ermentrout-Kopell runtime state")
	}
	if !finiteErmentroutKopell(iExt) {
		return 0, errors.New("invalid Ermentrout-Kopell current")
	}
	inp := s.Gain * iExt
	if !finiteErmentroutKopell(inp) {
		return 0, errors.New("invalid Ermentrout-Kopell input drive")
	}
	thetaPrev := s.Theta
	cosTheta := math.Cos(s.Theta)
	dTheta := 1.0 - cosTheta + (1.0+cosTheta)*inp
	thetaNext := s.Theta + s.Dt*dTheta
	if !finiteErmentroutKopell(dTheta) || !finiteErmentroutKopell(thetaNext) {
		return 0, errors.New("invalid Ermentrout-Kopell candidate phase")
	}
	fired := 0
	if thetaNext >= s.ThetaThreshold && thetaPrev < s.ThetaThreshold {
		fired = 1
	}
	s.Theta = math.Mod(thetaNext, 2.0*math.Pi)
	if s.Theta < 0.0 {
		s.Theta += 2.0 * math.Pi
	}
	return fired, nil
}

// SimulateErmentroutKopellMapNeuron runs the neuron for n steps
func SimulateErmentroutKopellMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewErmentroutKopellMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.Theta
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
