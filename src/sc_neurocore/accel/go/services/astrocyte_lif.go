// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for astrocyte_lif

package services

import (
	"errors"
	"math"
)

// AstrocyteLIFNeuronState holds the neuron state
type AstrocyteLIFNeuronState struct {
	TauM     float64
	TauCa    float64
	EL       float64
	Theta    float64
	VReset   float64
	CaDelta  float64
	CaThresh float64
	GGlio    float64
	Dt       float64
	V        float64
	Ca       float64
}

// NewAstrocyteLIFNeuron creates a new AstrocyteLIFNeuron neuron with default parameters
func NewAstrocyteLIFNeuron() *AstrocyteLIFNeuronState {
	return &AstrocyteLIFNeuronState{
		TauM:     20.0,
		TauCa:    500.0,
		EL:       -65.0,
		Theta:    -50.0,
		VReset:   -65.0,
		CaDelta:  0.1,
		CaThresh: 0.5,
		GGlio:    2.0,
		Dt:       0.1,
		V:        -65.0,
		Ca:       0.0,
	}
}

func astrocyteLIFFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// ValidateAstrocyteLIFNeuron checks the model's physical state contract.
func ValidateAstrocyteLIFNeuron(s *AstrocyteLIFNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.TauM, s.TauCa, s.Dt} {
		if !astrocyteLIFFinite(value) || value <= 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.EL, s.Theta, s.VReset, s.V} {
		if !astrocyteLIFFinite(value) {
			return false
		}
	}
	if s.Theta <= s.VReset {
		return false
	}
	for _, value := range []float64{s.CaDelta, s.CaThresh, s.GGlio, s.Ca} {
		if !astrocyteLIFFinite(value) || value < 0.0 {
			return false
		}
	}
	return true
}

// StepWithPre advances the neuron with a presynaptic spike flag.
func (s *AstrocyteLIFNeuronState) StepWithPre(iExt float64, preSpike bool) (int, error) {
	if !ValidateAstrocyteLIFNeuron(s) || !astrocyteLIFFinite(iExt) {
		return 0, errors.New("invalid astrocyte LIF state or input")
	}
	dCa := -s.Ca / s.TauCa
	if preSpike {
		dCa += s.CaDelta / s.Dt
	}
	caNext := math.Max(s.Ca+dCa*s.Dt, 0.0)
	if !astrocyteLIFFinite(caNext) || caNext < 0.0 {
		return 0, errors.New("invalid astrocyte calcium candidate")
	}
	iGlio := 0.0
	if caNext > s.CaThresh {
		iGlio = s.GGlio
	}
	if !astrocyteLIFFinite(iGlio) {
		return 0, errors.New("invalid gliotransmitter current")
	}
	dV := (-(s.V - s.EL) + iExt + iGlio) / s.TauM
	vNext := s.V + dV*s.Dt
	if !astrocyteLIFFinite(vNext) {
		return 0, errors.New("invalid membrane candidate")
	}
	s.Ca = caNext
	if vNext >= s.Theta {
		s.V = s.VReset
		return 1, nil
	}
	s.V = vNext
	return 0, nil
}

// Step advances the neuron by one timestep.
func (s *AstrocyteLIFNeuronState) Step(iExt float64) (int, error) {
	return s.StepWithPre(iExt, false)
}

// SimulateAstrocyteLIFNeuron runs the neuron for n steps.
func SimulateAstrocyteLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewAstrocyteLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
