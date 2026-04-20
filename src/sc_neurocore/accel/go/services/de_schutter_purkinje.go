// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for de_schutter_purkinje

package services

import (
	"math"
)

// DeSchutterPurkinjeNeuronState holds the neuron state
type DeSchutterPurkinjeNeuronState struct {
	V float64
	HNa float64
	NK float64
	MCap float64
	HCap float64
	QKca float64
	Ca float64
	GNa float64
	GK float64
	GCap float64
	GKca float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	CaDecay float64
	FCa float64
	Dt float64
	VThreshold float64
}

// NewDeSchutterPurkinjeNeuron creates a new DeSchutterPurkinjeNeuron neuron with default parameters
func NewDeSchutterPurkinjeNeuron() *DeSchutterPurkinjeNeuronState {
	return &DeSchutterPurkinjeNeuronState{
		V: -68.0,
		HNa: 0.8,
		NK: 0.1,
		MCap: 0.0,
		HCap: 0.9,
		QKca: 0.0,
		Ca: 0.0001,
		GNa: 125.0,
		GK: 10.0,
		GCap: 45.0,
		GKca: 35.0,
		GL: 0.5,
		ENa: 45.0,
		EK: -85.0,
		ECa: 135.0,
		EL: -68.0,
		CaDecay: 0.02,
		FCa: 0.00024,
		Dt: 0.01,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *DeSchutterPurkinjeNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -68.0
		return 1
	}
	return 0
}

// SimulateDeSchutterPurkinjeNeuron runs the neuron for n steps
func SimulateDeSchutterPurkinjeNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDeSchutterPurkinjeNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
