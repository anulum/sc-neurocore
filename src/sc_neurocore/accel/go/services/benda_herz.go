// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Benda-Herz adaptation

package services

import "math"

// BendaHerzNeuronState implements the source rate and phase equations.
type BendaHerzNeuronState struct {
	A, Phase, OnsetGain, Rheobase, AdaptationSlope, TauA, Dt float64
}

func NewBendaHerzNeuron() *BendaHerzNeuronState {
	return &BendaHerzNeuronState{OnsetGain: 60.0, AdaptationSlope: 0.1, TauA: 100.0, Dt: 0.1}
}

func (s BendaHerzNeuronState) Valid() bool {
	return finite(s.A) && s.A >= 0 && finite(s.Phase) && s.Phase >= 0 && s.Phase < 1 &&
		finite(s.OnsetGain) && s.OnsetGain > 0 && finite(s.Rheobase) &&
		finite(s.AdaptationSlope) && s.AdaptationSlope >= 0 && finite(s.TauA) && s.TauA > 0 && finite(s.Dt) && s.Dt > 0
}

func (s BendaHerzNeuronState) rhs(a, current float64) (float64, float64) {
	rate := s.OnsetGain * math.Sqrt(math.Max(current-a-s.Rheobase, 0))
	return (s.AdaptationSlope*rate - a) / s.TauA, rate / 1000.0
}

func (s *BendaHerzNeuronState) Step(current float64) int {
	if !s.Valid() || !finite(current) {
		return -1
	}
	k1a, k1p := s.rhs(s.A, current)
	k2a, k2p := s.rhs(s.A+0.5*s.Dt*k1a, current)
	k3a, k3p := s.rhs(s.A+0.5*s.Dt*k2a, current)
	k4a, k4p := s.rhs(s.A+s.Dt*k3a, current)
	scale := s.Dt / 6.0
	nextA := s.A + scale*(k1a+2*k2a+2*k3a+k4a)
	nextPhase := s.Phase + scale*(k1p+2*k2p+2*k3p+k4p)
	if !finite(nextA) || nextA < 0 || !finite(nextPhase) || nextPhase < 0 || nextPhase >= 2 {
		return -1
	}
	s.A = nextA
	if nextPhase >= 1 {
		s.Phase = 0
		return 1
	}
	s.Phase = nextPhase
	return 0
}

func (s *BendaHerzNeuronState) Reset() { s.A, s.Phase = 0, 0 }
