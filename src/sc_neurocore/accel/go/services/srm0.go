// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for srm0

package services

import (
	"math"
)

// SRM0NeuronState holds the neuron state
type SRM0NeuronState struct {
	V          float64
	VRest      float64
	VThreshold float64
	TauM       float64
	TauEta     float64
	EtaReset   float64
	Resistance float64
	Dt         float64
	Eta        float64
	T          float64
	LastSpikeT float64
}

// NewSRM0Neuron creates a new SRM0Neuron neuron with default parameters
func NewSRM0Neuron() *SRM0NeuronState {
	return &SRM0NeuronState{
		V:          0.0,
		VRest:      0.0,
		VThreshold: 1.0,
		TauM:       20.0,
		TauEta:     50.0,
		EtaReset:   5.0,
		Resistance: 1.0,
		Dt:         1.0,
		Eta:        0.0,
		T:          0.0,
		LastSpikeT: -1000.0,
	}
}

// Step advances the neuron by one timestep
func (s *SRM0NeuronState) Step(iExt float64) int {
	if !validateSRM0State(s) || !finiteSRM0(iExt) {
		return -1
	}
	nextV, nextEta, ok := s.exactCandidate(iExt)
	if !ok {
		return -1
	}
	nextT := s.T + s.Dt
	if nextV >= s.VThreshold {
		s.V = s.VRest
		s.Eta = -s.EtaReset
		s.T = nextT
		s.LastSpikeT = nextT
		return 1
	}
	s.V = nextV
	s.Eta = nextEta
	s.T = nextT
	return 0
}

// SimulateSRM0Neuron runs the neuron for n steps
func SimulateSRM0Neuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSRM0Neuron()
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

func (s *SRM0NeuronState) exactCandidate(iExt float64) (float64, float64, bool) {
	membraneDecay := math.Exp(-s.Dt / s.TauM)
	etaDecay := math.Exp(-s.Dt / s.TauEta)
	rateDelta := (1.0 / s.TauM) - (1.0 / s.TauEta)
	var etaCoupling float64
	if math.Abs(rateDelta) < 1.0e-14 {
		etaCoupling = s.Dt * membraneDecay / s.TauM
	} else {
		etaCoupling = (etaDecay - membraneDecay) / (s.TauM * rateDelta)
	}
	steady := s.VRest + s.Resistance*iExt
	nextEta := s.Eta * etaDecay
	nextV := steady + (s.V-steady)*membraneDecay + s.Eta*etaCoupling
	ok := finiteSRM0(nextV) && finiteSRM0(nextEta)
	return nextV, nextEta, ok
}

func finiteSRM0(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateSRM0State(s *SRM0NeuronState) bool {
	return finiteSRM0(s.V) &&
		finiteSRM0(s.VRest) &&
		finiteSRM0(s.VThreshold) &&
		finiteSRM0(s.TauM) &&
		s.TauM > 0.0 &&
		finiteSRM0(s.TauEta) &&
		s.TauEta > 0.0 &&
		finiteSRM0(s.EtaReset) &&
		s.EtaReset >= 0.0 &&
		finiteSRM0(s.Resistance) &&
		finiteSRM0(s.Dt) &&
		s.Dt > 0.0 &&
		finiteSRM0(s.Eta) &&
		finiteSRM0(s.T) &&
		finiteSRM0(s.LastSpikeT)
}
