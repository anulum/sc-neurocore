// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for clif

package services

import (
	"errors"
	"math"
)

const clifVMax = 1.0e12

// ComplementaryLIFNeuronState holds the neuron state.
type ComplementaryLIFNeuronState struct {
	VPos       float64
	VNeg       float64
	Tau        float64
	VThreshold float64
	Dt         float64
	Alpha      float64
}

// NewComplementaryLIFNeuron creates a new ComplementaryLIFNeuron neuron with default parameters.
func NewComplementaryLIFNeuron() *ComplementaryLIFNeuronState {
	s := &ComplementaryLIFNeuronState{VPos: 0.0, VNeg: 0.0, Tau: 10.0, VThreshold: 1.0, Dt: 1.0}
	s.Alpha = math.Exp(-s.Dt / s.Tau)
	return s
}

func clifFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func (s *ComplementaryLIFNeuronState) validatedAlpha() (float64, error) {
	if !clifFinite(s.Tau) || s.Tau <= 0.0 {
		return 0, errors.New("tau must be positive")
	}
	if !clifFinite(s.Dt) || s.Dt <= 0.0 {
		return 0, errors.New("dt must be positive")
	}
	ratio := -s.Dt / s.Tau
	if ratio < -700.0 {
		return 0.0, nil
	}
	alpha := math.Exp(ratio)
	if !clifFinite(alpha) || alpha < 0.0 || alpha >= 1.0 {
		return 0, errors.New("alpha must be in [0, 1)")
	}
	return alpha, nil
}

func (s *ComplementaryLIFNeuronState) validate() (float64, error) {
	if !clifFinite(s.VPos) || !clifFinite(s.VNeg) || math.Abs(s.VPos) > clifVMax || math.Abs(s.VNeg) > clifVMax {
		return 0, errors.New("CLIF membrane paths outside safety envelope")
	}
	if !clifFinite(s.VThreshold) || s.VThreshold <= 0.0 {
		return 0, errors.New("threshold must be positive")
	}
	return s.validatedAlpha()
}

// Step advances the neuron by one timestep.
func (s *ComplementaryLIFNeuronState) Step(iExt float64) (int, error) {
	if !clifFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	alpha, err := s.validate()
	if err != nil {
		return 0, err
	}
	inpPos := math.Max(iExt, 0.0)
	inpNeg := math.Max(-iExt, 0.0)
	vPosNext := alpha*s.VPos + inpPos
	vNegNext := alpha*s.VNeg + inpNeg
	diff := vPosNext - vNegNext
	if !clifFinite(vPosNext) || !clifFinite(vNegNext) || !clifFinite(diff) || math.Abs(vPosNext) > clifVMax || math.Abs(vNegNext) > clifVMax {
		return 0, errors.New("CLIF membrane candidate outside safety envelope")
	}
	s.Alpha = alpha
	if diff >= s.VThreshold {
		s.VPos, s.VNeg = 0.0, 0.0
		return 1, nil
	}
	if diff <= -s.VThreshold {
		s.VPos, s.VNeg = 0.0, 0.0
		return -1, nil
	}
	s.VPos, s.VNeg = vPosNext, vNegNext
	return 0, nil
}

// SimulateComplementaryLIFNeuron runs the neuron for n steps.
func SimulateComplementaryLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewComplementaryLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.VPos - s.VNeg
		if result != 0 {
			spikes++
		}
	}
	return trace, spikes
}
