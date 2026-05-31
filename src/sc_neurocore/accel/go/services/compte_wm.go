// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for compte_wm

package services

import (
	"errors"
	"math"
)

const compteVMin = -200.0
const compteVMax = 100.0
const compteGateMax = 1.0e6
const compteGabaTau = 5.0

// CompteWMNeuronState holds the neuron state.
type CompteWMNeuronState struct {
	V          float64
	SAmpa      float64
	SNmda      float64
	XNmda      float64
	SGaba      float64
	GL         float64
	GAmpa      float64
	GNmda      float64
	GGaba      float64
	EL         float64
	EExc       float64
	EInh       float64
	CM         float64
	Mg         float64
	TauAmpa    float64
	TauNmda    float64
	TauX       float64
	AlphaNmda  float64
	VThreshold float64
	VReset     float64
	Dt         float64
}

// NewCompteWMNeuron creates a new CompteWMNeuron neuron with default parameters.
func NewCompteWMNeuron() *CompteWMNeuronState {
	return &CompteWMNeuronState{V: -70.0, SAmpa: 0.0, SNmda: 0.0, XNmda: 0.0, SGaba: 0.0, GL: 0.025, GAmpa: 0.005, GNmda: 0.165, GGaba: 0.013, EL: -70.0, EExc: 0.0, EInh: -70.0, CM: 0.5, Mg: 1.0, TauAmpa: 2.0, TauNmda: 100.0, TauX: 2.0, AlphaNmda: 0.5, VThreshold: -50.0, VReset: -55.0, Dt: 0.1}
}

func compteFinite(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func compteNonnegative(value float64) bool { return compteFinite(value) && value >= 0.0 }

func compteDecay(dt, tau float64) (float64, error) {
	ratio := -dt / tau
	if ratio < -700.0 {
		return 0.0, nil
	}
	decay := math.Exp(ratio)
	if !compteFinite(decay) || decay < 0.0 || decay >= 1.0 {
		return 0, errors.New("decay must be in [0, 1)")
	}
	return decay, nil
}

func (s *CompteWMNeuronState) validateGate(value float64) bool {
	return compteNonnegative(value) && value <= compteGateMax
}

func (s *CompteWMNeuronState) validate() (float64, float64, float64, error) {
	if !compteFinite(s.V) || s.V < compteVMin || s.V > compteVMax {
		return 0, 0, 0, errors.New("v outside Compte WM safety envelope")
	}
	if !s.validateGate(s.SAmpa) || !s.validateGate(s.SNmda) || !s.validateGate(s.XNmda) || !s.validateGate(s.SGaba) || s.SNmda > 1.0 {
		return 0, 0, 0, errors.New("synaptic gate outside Compte WM safety envelope")
	}
	for _, value := range []float64{s.GL, s.GAmpa, s.GNmda, s.GGaba, s.Mg, s.AlphaNmda} {
		if !compteNonnegative(value) {
			return 0, 0, 0, errors.New("non-negative Compte parameter invalid")
		}
	}
	for _, value := range []float64{s.CM, s.TauAmpa, s.TauNmda, s.TauX, s.Dt} {
		if !compteFinite(value) || value <= 0.0 {
			return 0, 0, 0, errors.New("positive Compte parameter invalid")
		}
	}
	for _, value := range []float64{s.EL, s.EExc, s.EInh, s.VThreshold, s.VReset} {
		if !compteFinite(value) {
			return 0, 0, 0, errors.New("finite Compte parameter invalid")
		}
	}
	if s.VReset < compteVMin || s.VReset > compteVMax {
		return 0, 0, 0, errors.New("v_reset outside Compte WM safety envelope")
	}
	decayAmpa, err := compteDecay(s.Dt, s.TauAmpa)
	if err != nil {
		return 0, 0, 0, err
	}
	decayX, err := compteDecay(s.Dt, s.TauX)
	if err != nil {
		return 0, 0, 0, err
	}
	decayGaba, err := compteDecay(s.Dt, compteGabaTau)
	if err != nil {
		return 0, 0, 0, err
	}
	return decayAmpa, decayX, decayGaba, nil
}

func (s *CompteWMNeuronState) mgBlock(v float64) (float64, error) {
	exponent := -0.062 * v
	expValue := 0.0
	if exponent >= -700.0 {
		expValue = math.Exp(math.Min(exponent, 700.0))
	}
	denominator := 1.0 + s.Mg/3.57*expValue
	if !compteFinite(denominator) || denominator <= 0.0 {
		return 0, errors.New("Mg block denominator invalid")
	}
	block := 1.0 / denominator
	if block < 0.0 || block > 1.0 {
		return 0, errors.New("Mg block outside [0, 1]")
	}
	return block, nil
}

// Step advances the neuron by one timestep without presynaptic spike input.
func (s *CompteWMNeuronState) Step(iExt float64) (int, error) { return s.StepWithSpike(iExt, false) }

// StepWithSpike advances the neuron by one timestep with optional presynaptic spike input.
func (s *CompteWMNeuronState) StepWithSpike(iExt float64, spikeIn bool) (int, error) {
	if !compteFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	decayAmpa, decayX, decayGaba, err := s.validate()
	if err != nil {
		return 0, err
	}
	spikeIncrement := 0.0
	if spikeIn {
		spikeIncrement = 1.0
	}
	sAmpaPre := s.SAmpa + spikeIncrement
	xNmdaPre := s.XNmda + spikeIncrement
	if sAmpaPre > compteGateMax || xNmdaPre > compteGateMax {
		return 0, errors.New("spike input gate candidate outside Compte safety envelope")
	}
	sAmpaCandidate := sAmpaPre * decayAmpa
	sNmdaCandidate := s.SNmda + (-s.SNmda/s.TauNmda+s.AlphaNmda*xNmdaPre*(1.0-s.SNmda))*s.Dt
	xNmdaCandidate := xNmdaPre * decayX
	sGabaCandidate := s.SGaba * decayGaba
	for _, value := range []float64{sAmpaCandidate, sNmdaCandidate, xNmdaCandidate, sGabaCandidate} {
		if !compteFinite(value) || value < 0.0 || value > compteGateMax {
			return 0, errors.New("gate candidate outside Compte safety envelope")
		}
	}
	if sNmdaCandidate > 1.0 {
		return 0, errors.New("NMDA gate candidate must remain bounded by 1")
	}
	block, err := s.mgBlock(s.V)
	if err != nil {
		return 0, err
	}
	iL := s.GL * (s.V - s.EL)
	iAmpa := s.GAmpa * sAmpaCandidate * (s.V - s.EExc)
	iNmda := s.GNmda * block * sNmdaCandidate * (s.V - s.EExc)
	iGaba := s.GGaba * sGabaCandidate * (s.V - s.EInh)
	dv := (-iL - iAmpa - iNmda - iGaba + iExt) / s.CM * s.Dt
	vCandidate := s.V + dv
	for _, value := range []float64{iL, iAmpa, iNmda, iGaba, dv, vCandidate} {
		if !compteFinite(value) {
			return 0, errors.New("Compte current candidate must be finite")
		}
	}
	if vCandidate < compteVMin || vCandidate > compteVMax {
		return 0, errors.New("voltage candidate outside Compte WM safety envelope")
	}
	if vCandidate >= s.VThreshold {
		gabaAfterSpike := sGabaCandidate + 1.0
		if gabaAfterSpike > compteGateMax {
			return 0, errors.New("GABA spike candidate outside Compte safety envelope")
		}
		s.V = s.VReset
		s.SAmpa = sAmpaCandidate
		s.SNmda = sNmdaCandidate
		s.XNmda = xNmdaCandidate
		s.SGaba = gabaAfterSpike
		return 1, nil
	}
	s.V = vCandidate
	s.SAmpa = sAmpaCandidate
	s.SNmda = sNmdaCandidate
	s.XNmda = xNmdaCandidate
	s.SGaba = sGabaCandidate
	return 0, nil
}

// SimulateCompteWMNeuron runs the neuron for n steps.
func SimulateCompteWMNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewCompteWMNeuron()
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
