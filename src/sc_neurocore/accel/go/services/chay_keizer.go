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

const chayKeizerMaxSubstep = 0.001
const chayKeizerVMin = -200.0
const chayKeizerVMax = 200.0
const chayKeizerCAMax = 100.0

// ChayKeizerNeuronState holds the neuron state.
type ChayKeizerNeuronState struct {
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
	KD         float64
	FCa        float64
	KCa        float64
	Dt         float64
	VThreshold float64
}

// NewChayKeizerNeuron creates a new ChayNeuron neuron with default parameters.
func NewChayKeizerNeuron() *ChayKeizerNeuronState {
	return &ChayKeizerNeuronState{
		V:          -50.0,
		N:          0.1,
		Ca:         0.1,
		GCa:        20.0,
		GK:         25.0,
		GKca:       12.0,
		GL:         0.1,
		ECa:        100.0,
		EK:         -75.0,
		EL:         -40.0,
		KD:         1.0,
		FCa:        0.004,
		KCa:        0.03,
		Dt:         0.02,
		VThreshold: -20.0,
	}
}

func chayKeizerFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func chayKeizerProbability(value float64) bool {
	return chayKeizerFinite(value) && value >= 0.0 && value <= 1.0
}
func chayKeizerNonnegative(value float64) bool { return chayKeizerFinite(value) && value >= 0.0 }

func chayKeizerCheckedExp(exponent float64) (float64, error) {
	if !chayKeizerFinite(exponent) {
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

func chayKeizerGateInf(exponent float64) (float64, error) {
	value, err := chayKeizerCheckedExp(exponent)
	if err != nil {
		return 0, err
	}
	return 1.0 / (1.0 + value), nil
}

func (s *ChayKeizerNeuronState) validate() (int, float64, error) {
	if !chayKeizerFinite(s.V) || s.V < chayKeizerVMin || s.V > chayKeizerVMax {
		return 0, 0, errors.New("v outside Chay-Keizer safety envelope")
	}
	if !chayKeizerProbability(s.N) {
		return 0, 0, errors.New("n must be in [0, 1]")
	}
	if !chayKeizerNonnegative(s.Ca) || s.Ca > chayKeizerCAMax {
		return 0, 0, errors.New("ca outside Chay-Keizer safety envelope")
	}
	for _, value := range []float64{s.GCa, s.GK, s.GKca, s.GL, s.KD, s.FCa, s.KCa} {
		if !chayKeizerNonnegative(value) {
			return 0, 0, errors.New("non-negative Chay-Keizer parameter invalid")
		}
	}
	for _, value := range []float64{s.ECa, s.EK, s.EL, s.VThreshold} {
		if !chayKeizerFinite(value) {
			return 0, 0, errors.New("finite Chay-Keizer parameter invalid")
		}
	}
	if !chayKeizerFinite(s.Dt) || s.Dt <= 0.0 {
		return 0, 0, errors.New("dt must be positive")
	}
	substeps := int(math.Ceil(s.Dt / chayKeizerMaxSubstep))
	if substeps < 1 {
		substeps = 1
	}
	if substeps > 10000 {
		return 0, 0, errors.New("dt requires too many Chay-Keizer safety substeps")
	}
	return substeps, s.Dt / float64(substeps), nil
}

func (s *ChayKeizerNeuronState) candidate(v, n, ca, h, iExt float64) (float64, float64, float64, error) {
	mInf, err := chayKeizerGateInf(-(v + 25.0) / 8.0)
	if err != nil {
		return 0, 0, 0, err
	}
	nInf, err := chayKeizerGateInf(-(v + 18.0) / 14.0)
	if err != nil {
		return 0, 0, 0, err
	}
	tauExp, err := chayKeizerCheckedExp((v + 18.0) / 14.0)
	if err != nil {
		return 0, 0, 0, err
	}
	tauN := 20.0 / (1.0 + tauExp)
	caDenominator := ca + s.KD
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
	caNext := ca + (-s.FCa*iCa-s.KCa*ca)*h

	if !chayKeizerFinite(vNext) || vNext < chayKeizerVMin || vNext > chayKeizerVMax {
		return 0, 0, 0, errors.New("Chay-Keizer voltage candidate outside safety envelope")
	}
	if !chayKeizerProbability(nNext) {
		return 0, 0, 0, errors.New("Chay-Keizer n-gate candidate outside [0, 1]")
	}
	if !chayKeizerNonnegative(caNext) || caNext > chayKeizerCAMax {
		return 0, 0, 0, errors.New("Chay-Keizer calcium candidate outside safety envelope")
	}
	return vNext, nNext, caNext, nil
}

// Step advances the neuron by one timestep.
func (s *ChayKeizerNeuronState) Step(iExt float64) (int, error) {
	if !chayKeizerFinite(iExt) {
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

// SimulateChayKeizerNeuron runs the neuron for n steps.
func SimulateChayKeizerNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewChayKeizerNeuron()
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
