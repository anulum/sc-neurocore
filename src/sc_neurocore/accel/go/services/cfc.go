// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for cfc

package services

import (
	"math"
)

// ClosedFormContinuousNeuronState holds the neuron state
type ClosedFormContinuousNeuronState struct {
	X          float64
	WTau       float64
	WX         float64
	WIn        float64
	TauBase    float64
	Bias       float64
	VThreshold float64
	Dt         float64
}

// NewClosedFormContinuousNeuron creates a new ClosedFormContinuousNeuron neuron with default parameters
func NewClosedFormContinuousNeuron() *ClosedFormContinuousNeuronState {
	return &ClosedFormContinuousNeuronState{
		X:          0.0,
		WTau:       -0.5,
		WX:         0.8,
		WIn:        1.0,
		TauBase:    10.0,
		Bias:       0.0,
		VThreshold: 1.0,
		Dt:         1.0,
	}
}

// ValidateClosedFormContinuousNeuron checks that the CfC parameters define a finite dynamical system.
func ValidateClosedFormContinuousNeuron(s *ClosedFormContinuousNeuronState) bool {
	return s != nil &&
		!math.IsNaN(s.X) && !math.IsInf(s.X, 0) &&
		!math.IsNaN(s.WTau) && !math.IsInf(s.WTau, 0) &&
		!math.IsNaN(s.WX) && !math.IsInf(s.WX, 0) &&
		!math.IsNaN(s.WIn) && !math.IsInf(s.WIn, 0) &&
		!math.IsNaN(s.TauBase) && !math.IsInf(s.TauBase, 0) && s.TauBase > 0 &&
		!math.IsNaN(s.Bias) && !math.IsInf(s.Bias, 0) &&
		!math.IsNaN(s.VThreshold) && !math.IsInf(s.VThreshold, 0) && s.VThreshold > 0 &&
		!math.IsNaN(s.Dt) && !math.IsInf(s.Dt, 0) && s.Dt > 0
}

func cfcSigmoid(value float64) float64 {
	if value >= 0 {
		z := math.Exp(-value)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(value)
	return z / (1.0 + z)
}

// Step advances the neuron by one timestep
func (s *ClosedFormContinuousNeuronState) Step(iExt float64) int {
	if !ValidateClosedFormContinuousNeuron(s) || math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0
	}

	sigmaTau := cfcSigmoid(s.WTau*iExt + s.Bias)
	tauEff := math.Max(s.TauBase*sigmaTau, 0.1)
	fTarget := math.Tanh(s.WX*s.X + s.WIn*iExt)
	decay := math.Exp(-s.Dt / tauEff)
	s.X = s.X*decay + fTarget*(1.0-decay)
	if s.X >= s.VThreshold {
		s.X = 0.0
		return 1
	}
	return 0
}

// SimulateClosedFormContinuousNeuron runs the neuron for n steps
func SimulateClosedFormContinuousNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewClosedFormContinuousNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
