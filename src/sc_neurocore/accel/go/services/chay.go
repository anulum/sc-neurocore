// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for chay

package services

import (
	"errors"
	"math"
)

const chayMaxSubstep = 0.001
const chayVMin = -200.0
const chayVMax = 200.0
const chayCAMax = 100.0

// ChayNeuronState holds the neuron state.
type ChayNeuronState struct {
	V          float64
	N          float64
	Ca         float64
	GCa        float64
	GK         float64
	GKca       float64
	GL         float64
	ECa        float64
	EK         float64
	EL         float64
	Rho        float64
	AlphaCa    float64
	KCa        float64
	Dt         float64
	VThreshold float64
}

// NewChayNeuron creates a new ChayNeuron neuron with default parameters.
func NewChayNeuron() *ChayNeuronState {
	return &ChayNeuronState{
		V:          -50.0,
		N:          0.1,
		Ca:         0.1,
		GCa:        25.0,
		GK:         1400.0,
		GKca:       12.0,
		GL:         7.0,
		ECa:        100.0,
		EK:         -75.0,
		EL:         -40.0,
		Rho:        0.00015,
		AlphaCa:    0.002,
		KCa:        0.04,
		Dt:         0.02,
		VThreshold: -20.0,
	}
}

func chayFinite(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func chayProbability(value float64) bool { return chayFinite(value) && value >= 0.0 && value <= 1.0 }
func chayNonnegative(value float64) bool { return chayFinite(value) && value >= 0.0 }

func chayCheckedExp(exponent float64) (float64, error) {
	if !chayFinite(exponent) {
		return 0, errors.New("exponent must be finite")
	}
	if exponent < -700.0 {
		return 0, nil
	}
	if exponent > 700.0 {
		return math.Exp(700.0), nil
	}
	return math.Exp(exponent), nil
}

func chayGateInf(exponent float64) (float64, error) {
	value, err := chayCheckedExp(exponent)
	if err != nil {
		return 0, err
	}
	return 1.0 / (1.0 + value), nil
}

func (s *ChayNeuronState) validate() (int, float64, error) {
	if !chayFinite(s.V) || s.V < chayVMin || s.V > chayVMax {
		return 0, 0, errors.New("v outside Chay safety envelope")
	}
	if !chayProbability(s.N) {
		return 0, 0, errors.New("n must be in [0, 1]")
	}
	if !chayNonnegative(s.Ca) || s.Ca > chayCAMax {
		return 0, 0, errors.New("ca outside Chay safety envelope")
	}
	for _, value := range []float64{s.GCa, s.GK, s.GKca, s.GL, s.Rho, s.AlphaCa, s.KCa} {
		if !chayNonnegative(value) {
			return 0, 0, errors.New("non-negative Chay parameter invalid")
		}
	}
	for _, value := range []float64{s.ECa, s.EK, s.EL, s.VThreshold} {
		if !chayFinite(value) {
			return 0, 0, errors.New("finite Chay parameter invalid")
		}
	}
	if !chayFinite(s.Dt) || s.Dt <= 0.0 {
		return 0, 0, errors.New("dt must be positive")
	}
	substeps := int(math.Ceil(s.Dt / chayMaxSubstep))
	if substeps < 1 {
		substeps = 1
	}
	if substeps > 10000 {
		return 0, 0, errors.New("dt requires too many Chay safety substeps")
	}
	return substeps, s.Dt / float64(substeps), nil
}

func (s *ChayNeuronState) candidate(v, n, ca, h, iExt float64) (float64, float64, float64, error) {
	mInf, err := chayGateInf(-(v + 25.0) / 8.0)
	if err != nil {
		return 0, 0, 0, err
	}
	nInf, err := chayGateInf(-(v + 18.0) / 14.0)
	if err != nil {
		return 0, 0, 0, err
	}
	tauN := 1.0 / (0.01 * math.Max(math.Abs(v+18.0), 0.01))
	caDenominator := ca + 1.0
	if caDenominator <= 0.0 {
		return 0, 0, 0, errors.New("calcium activation denominator must be positive")
	}

	iCa := s.GCa * mInf * (v - s.ECa)
	kcaAct := ca / caDenominator
	iK := s.GK * n * (v - s.EK)
	iKca := s.GKca * kcaAct * (v - s.EK)
	iL := s.GL * (v - s.EL)

	vNext := v + (-iCa-iK-iKca-iL+iExt)*h
	nNext := n + (nInf-n)/math.Max(tauN, 0.01)*h
	caNext := ca + s.Rho*(-s.AlphaCa*iCa-s.KCa*ca)*h

	if !chayFinite(vNext) || vNext < chayVMin || vNext > chayVMax {
		return 0, 0, 0, errors.New("Chay voltage candidate outside safety envelope")
	}
	if !chayProbability(nNext) {
		return 0, 0, 0, errors.New("Chay n-gate candidate outside [0, 1]")
	}
	if !chayNonnegative(caNext) || caNext > chayCAMax {
		return 0, 0, 0, errors.New("Chay calcium candidate outside safety envelope")
	}
	return vNext, nNext, caNext, nil
}

// Step advances the neuron by one timestep.
func (s *ChayNeuronState) Step(iExt float64) (int, error) {
	if !chayFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	substeps, h, err := s.validate()
	if err != nil {
		return 0, err
	}
	vInitial := s.V
	v, n, ca := s.V, s.N, s.Ca
	crossed := false
	for step := 0; step < substeps; step++ {
		vNext, nNext, caNext, err := s.candidate(v, n, ca, h, iExt)
		if err != nil {
			return 0, err
		}
		crossed = crossed || (vNext >= s.VThreshold && v < s.VThreshold)
		v, n, ca = vNext, nNext, caNext
	}
	s.V, s.N, s.Ca = v, n, ca
	if crossed && vInitial < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateChayNeuron runs the neuron for n steps.
func SimulateChayNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewChayNeuron()
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
