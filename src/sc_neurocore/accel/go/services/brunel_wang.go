// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for brunel_wang

package services

import (
	"errors"
	"math"
)

// BrunelWangNeuronState holds the neuron state
type BrunelWangNeuronState struct {
	V            float64
	VRest        float64
	VReset       float64
	VThreshold   float64
	TauM         float64
	TauRef       float64
	TauAmpa      float64
	TauNmdaRise  float64
	TauNmdaDecay float64
	TauGaba      float64
	GAmpaExt     float64
	GAmpaRec     float64
	GNmda        float64
	GGaba        float64
	VAmpa        float64
	VNmda        float64
	VGaba        float64
	CM           float64
	MgConc       float64
	Dt           float64
	RefRemaining float64
}

// NewBrunelWangNeuron creates a new BrunelWangNeuron neuron with default parameters
func NewBrunelWangNeuron() *BrunelWangNeuronState {
	return &BrunelWangNeuronState{
		V:            -70.0,
		VRest:        -70.0,
		VReset:       -55.0,
		VThreshold:   -50.0,
		TauM:         20.0,
		TauRef:       2.0,
		TauAmpa:      2.0,
		TauNmdaRise:  2.0,
		TauNmdaDecay: 100.0,
		TauGaba:      5.0,
		GAmpaExt:     2.1,
		GAmpaRec:     0.05,
		GNmda:        0.165,
		GGaba:        1.3,
		VAmpa:        0.0,
		VNmda:        0.0,
		VGaba:        -70.0,
		CM:           0.5,
		MgConc:       1.0,
		Dt:           0.1,
		RefRemaining: 0.0,
	}
}

func finiteBrunelWang(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func positiveBrunelWang(value float64) bool {
	return finiteBrunelWang(value) && value > 0.0
}

func nonnegativeBrunelWang(value float64) bool {
	return finiteBrunelWang(value) && value >= 0.0
}

func gateBrunelWang(value float64) bool {
	return finiteBrunelWang(value) && value >= 0.0 && value <= 1.0
}

// ValidateBrunelWang checks membrane, refractory, and synaptic parameters.
func ValidateBrunelWang(s *BrunelWangNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteBrunelWang(s.V) &&
		finiteBrunelWang(s.VRest) &&
		finiteBrunelWang(s.VReset) &&
		finiteBrunelWang(s.VThreshold) &&
		positiveBrunelWang(s.TauM) &&
		positiveBrunelWang(s.TauRef) &&
		positiveBrunelWang(s.TauAmpa) &&
		positiveBrunelWang(s.TauNmdaRise) &&
		positiveBrunelWang(s.TauNmdaDecay) &&
		positiveBrunelWang(s.TauGaba) &&
		nonnegativeBrunelWang(s.GAmpaExt) &&
		nonnegativeBrunelWang(s.GAmpaRec) &&
		nonnegativeBrunelWang(s.GNmda) &&
		nonnegativeBrunelWang(s.GGaba) &&
		finiteBrunelWang(s.VAmpa) &&
		finiteBrunelWang(s.VNmda) &&
		finiteBrunelWang(s.VGaba) &&
		positiveBrunelWang(s.CM) &&
		nonnegativeBrunelWang(s.MgConc) &&
		positiveBrunelWang(s.Dt) &&
		nonnegativeBrunelWang(s.RefRemaining)
}

func nmdaVoltageDepBrunelWang(s *BrunelWangNeuronState, v float64) (float64, error) {
	if !finiteBrunelWang(v) {
		return 0, errors.New("invalid Brunel-Wang voltage")
	}
	exponent := -0.062 * v
	if exponent > 700.0 {
		return 0.0, nil
	}
	factor := 1.0 / (1.0 + s.MgConc/3.57*math.Exp(exponent))
	if !finiteBrunelWang(factor) || factor < 0.0 || factor > 1.0 {
		return 0, errors.New("invalid Brunel-Wang NMDA voltage factor")
	}
	return factor, nil
}

// Step advances the neuron by one timestep
func (s *BrunelWangNeuronState) Step(iAmpaExt float64, syn ...float64) (int, error) {
	sAmpaRec := 0.0
	sNmdaRec := 0.0
	sGaba := 0.0
	if len(syn) > 0 {
		sAmpaRec = syn[0]
	}
	if len(syn) > 1 {
		sNmdaRec = syn[1]
	}
	if len(syn) > 2 {
		sGaba = syn[2]
	}
	if !ValidateBrunelWang(s) {
		return 0, errors.New("invalid Brunel-Wang runtime state")
	}
	if !nonnegativeBrunelWang(iAmpaExt) || !gateBrunelWang(sAmpaRec) || !gateBrunelWang(sNmdaRec) || !gateBrunelWang(sGaba) {
		return 0, errors.New("invalid Brunel-Wang synaptic input")
	}

	if s.RefRemaining > 0.0 {
		s.RefRemaining = math.Max(0.0, s.RefRemaining-s.Dt)
		return 0, nil
	}

	nmdaFactor, err := nmdaVoltageDepBrunelWang(s, s.V)
	if err != nil {
		return 0, err
	}
	iAmpa := -s.GAmpaExt*(s.V-s.VAmpa)*iAmpaExt - s.GAmpaRec*(s.V-s.VAmpa)*sAmpaRec
	iNmda := -s.GNmda * nmdaFactor * (s.V - s.VNmda) * sNmdaRec
	iGaba := -s.GGaba * (s.V - s.VGaba) * sGaba
	iLeak := -(s.V - s.VRest) / s.TauM
	dv := (iLeak + (iAmpa+iNmda+iGaba)/s.CM) * s.Dt
	nextV := s.V + dv
	if !finiteBrunelWang(iAmpa) || !finiteBrunelWang(iNmda) ||
		!finiteBrunelWang(iGaba) || !finiteBrunelWang(iLeak) ||
		!finiteBrunelWang(dv) || !finiteBrunelWang(nextV) {
		return 0, errors.New("invalid Brunel-Wang membrane candidate")
	}

	s.V = nextV
	if s.V >= s.VThreshold {
		s.V = s.VReset
		s.RefRemaining = s.TauRef
		return 1, nil
	}
	return 0, nil
}

// SimulateBrunelWangNeuron runs the neuron for n steps
func SimulateBrunelWangNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBrunelWangNeuron()
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
