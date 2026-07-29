// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel-Wang 2001 midpoint-RK2 Go service

package services

import (
	"errors"
	"math"
)

// BrunelWangNeuronState holds the complete source cell state and configuration.
type BrunelWangNeuronState struct {
	V, VRest, VReset, VThreshold float64
	TauM, TauRef                 float64
	GAmpaExt, GAmpaRec           float64
	GNmda, GGaba                 float64
	VAmpa, VNmda, VGaba          float64
	CM, MgConc, Dt, RefRemaining float64
}

// NewBrunelWangNeuron constructs Brunel and Wang's pyramidal-cell defaults.
func NewBrunelWangNeuron() *BrunelWangNeuronState {
	return &BrunelWangNeuronState{
		V: -70, VRest: -70, VReset: -55, VThreshold: -50,
		TauM: 20, TauRef: 2, GAmpaExt: 2.08, GAmpaRec: 0.104,
		GNmda: 0.327, GGaba: 1.25, VAmpa: 0, VNmda: 0, VGaba: -70,
		CM: 0.5, MgConc: 1, Dt: 0.1, RefRemaining: 0,
	}
}

func finiteBrunelWang(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func nonnegativeBrunelWang(value float64) bool { return finiteBrunelWang(value) && value >= 0 }

// ValidateBrunelWang checks every mutable state/configuration invariant.
func ValidateBrunelWang(s *BrunelWangNeuronState) bool {
	if s == nil {
		return false
	}
	values := []float64{s.V, s.VRest, s.VReset, s.VThreshold, s.TauM, s.TauRef,
		s.GAmpaExt, s.GAmpaRec, s.GNmda, s.GGaba, s.VAmpa, s.VNmda,
		s.VGaba, s.CM, s.MgConc, s.Dt, s.RefRemaining}
	for _, value := range values {
		if !finiteBrunelWang(value) {
			return false
		}
	}
	return s.TauM > 0 && s.TauRef > 0 && s.CM > 0 && s.Dt > 0 &&
		s.GAmpaExt >= 0 && s.GAmpaRec >= 0 && s.GNmda >= 0 && s.GGaba >= 0 &&
		s.MgConc >= 0 && s.RefRemaining >= 0
}

func (s *BrunelWangNeuronState) derivative(v, ext, ampa, nmda, gaba float64) float64 {
	block := 1 / (1 + s.MgConc/3.57*math.Exp(-0.062*v))
	iAmpa := -s.GAmpaExt*(v-s.VAmpa)*ext - s.GAmpaRec*(v-s.VAmpa)*ampa
	iNmda := -s.GNmda * block * (v - s.VNmda) * nmda
	iGaba := -s.GGaba * (v - s.VGaba) * gaba
	return -(v-s.VRest)/s.TauM + (iAmpa+iNmda+iGaba)/s.CM
}

// Step advances one atomic midpoint-RK2 step over four aggregate gates.
func (s *BrunelWangNeuronState) Step(ext float64, syn ...float64) (int, error) {
	ampa, nmda, gaba := 0.0, 0.0, 0.0
	if len(syn) > 0 {
		ampa = syn[0]
	}
	if len(syn) > 1 {
		nmda = syn[1]
	}
	if len(syn) > 2 {
		gaba = syn[2]
	}
	if !ValidateBrunelWang(s) || !nonnegativeBrunelWang(ext) || !nonnegativeBrunelWang(ampa) ||
		!nonnegativeBrunelWang(nmda) || !nonnegativeBrunelWang(gaba) {
		return 0, errors.New("invalid Brunel-Wang configuration or aggregate gate")
	}
	if s.RefRemaining > 0 {
		s.V = s.VReset
		s.RefRemaining = math.Max(0, s.RefRemaining-s.Dt)
		return 0, nil
	}
	v := s.V
	k1 := s.derivative(v, ext, ampa, nmda, gaba)
	midpoint := v + 0.5*s.Dt*k1
	k2 := s.derivative(midpoint, ext, ampa, nmda, gaba)
	candidate := v + s.Dt*k2
	if !finiteBrunelWang(k1) || !finiteBrunelWang(midpoint) || !finiteBrunelWang(k2) || !finiteBrunelWang(candidate) {
		return 0, errors.New("non-finite Brunel-Wang RK2 candidate")
	}
	s.V = candidate
	if candidate >= s.VThreshold {
		s.V = s.VReset
		s.RefRemaining = s.TauRef
		return 1, nil
	}
	return 0, nil
}

// Reset clears only dynamic state and preserves configuration.
func (s *BrunelWangNeuronState) Reset() { s.V, s.RefRemaining = s.VRest, 0 }

// SimulateBrunelWangNeuron retains the one-drive service compatibility path.
func SimulateBrunelWangNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBrunelWangNeuron()
	trace, spikes := make([]float64, nSteps), 0
	for index := range trace {
		event, err := s.Step(iExt)
		if err != nil {
			trace[index] = math.NaN()
			continue
		}
		trace[index], spikes = s.V, spikes+event
	}
	return trace, spikes
}
