// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go mirror of Bertram et al. 2000 phantom burster

package services

import "math"

type BertramPhantomState struct {
	V, N, S1, S2                          float64
	LambdaN, GCa, GK, GS1, GS2, GL        float64
	ECa, EK, EL, CM                       float64
	VM, SM, VN, SN, VS1, SS1, VS2, SS2    float64
	TauNBar, TauS1, TauS2, Dt, VThreshold float64
}

func NewBertramPhantom() *BertramPhantomState {
	return &BertramPhantomState{
		V: -43, N: 0.03, S1: 0.1, S2: 0.434, LambdaN: 1.1,
		GCa: 280, GK: 1300, GS1: 20, GS2: 32, GL: 25,
		ECa: 100, EK: -80, EL: -40, CM: 4524,
		VM: -22, SM: 7.5, VN: -9, SN: 10,
		VS1: -40, SS1: 0.5, VS2: -42, SS2: 0.4,
		TauNBar: 9.09, TauS1: 1000, TauS2: 120000, Dt: 0.5, VThreshold: -20,
	}
}

func bertramBoltz(v, midpoint, slope float64) float64 {
	return 1 / (1 + math.Exp((midpoint-v)/slope))
}

func (s *BertramPhantomState) derivatives(v, n, s1, s2, current float64) [4]float64 {
	mInf := bertramBoltz(v, s.VM, s.SM)
	nInf := bertramBoltz(v, s.VN, s.SN)
	s1Inf := bertramBoltz(v, s.VS1, s.SS1)
	s2Inf := bertramBoltz(v, s.VS2, s.SS2)
	tauN := s.TauNBar / (1 + math.Exp((v-s.VN)/s.SN))
	iCa := s.GCa * mInf * (v - s.ECa)
	iK := s.GK * n * (v - s.EK)
	iS1 := s.GS1 * s1 * (v - s.EK)
	iS2 := s.GS2 * s2 * (v - s.EK)
	iL := s.GL * (v - s.EL)
	return [4]float64{
		(-iCa - iK - iS1 - iS2 - iL + current) / s.CM,
		s.LambdaN * (nInf - n) / tauN,
		(s1Inf - s1) / s.TauS1,
		(s2Inf - s2) / s.TauS2,
	}
}

func shiftedBertram(state, derivative [4]float64, scale float64) [4]float64 {
	return [4]float64{
		state[0] + scale*derivative[0], state[1] + scale*derivative[1],
		state[2] + scale*derivative[2], state[3] + scale*derivative[3],
	}
}

func (s *BertramPhantomState) valid() bool {
	values := []float64{s.V, s.N, s.S1, s.S2, s.LambdaN, s.GCa, s.GK, s.GS1, s.GS2, s.GL,
		s.ECa, s.EK, s.EL, s.CM, s.VM, s.SM, s.VN, s.SN, s.VS1, s.SS1, s.VS2, s.SS2,
		s.TauNBar, s.TauS1, s.TauS2, s.Dt, s.VThreshold}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return s.V >= -250 && s.V <= 250 && s.N >= 0 && s.N <= 1 && s.S1 >= 0 && s.S1 <= 1 &&
		s.S2 >= 0 && s.S2 <= 1 && s.LambdaN > 0 && s.GCa >= 0 && s.GK >= 0 && s.GS1 >= 0 &&
		s.GS2 >= 0 && s.GL >= 0 && s.CM > 0 && s.SM > 0 && s.SN > 0 && s.SS1 > 0 && s.SS2 > 0 &&
		s.TauNBar > 0 && s.TauS1 > 0 && s.TauS2 > 0 && s.Dt > 0
}

func (s *BertramPhantomState) Step(current float64) int {
	if !s.valid() || math.IsNaN(current) || math.IsInf(current, 0) {
		return -1
	}
	previousV := s.V
	state := [4]float64{s.V, s.N, s.S1, s.S2}
	k1 := s.derivatives(state[0], state[1], state[2], state[3], current)
	k2s := shiftedBertram(state, k1, 0.5*s.Dt)
	k2 := s.derivatives(k2s[0], k2s[1], k2s[2], k2s[3], current)
	k3s := shiftedBertram(state, k2, 0.5*s.Dt)
	k3 := s.derivatives(k3s[0], k3s[1], k3s[2], k3s[3], current)
	k4s := shiftedBertram(state, k3, s.Dt)
	k4 := s.derivatives(k4s[0], k4s[1], k4s[2], k4s[3], current)
	for i := range state {
		state[i] += s.Dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
		if math.IsNaN(state[i]) || math.IsInf(state[i], 0) {
			return -1
		}
	}
	if state[0] < -250 || state[0] > 250 || state[1] < -1e-9 || state[1] > 1+1e-9 ||
		state[2] < -1e-9 || state[2] > 1+1e-9 || state[3] < -1e-9 || state[3] > 1+1e-9 {
		return -1
	}
	s.V, s.N, s.S1, s.S2 = state[0], max(0, min(1, state[1])), max(0, min(1, state[2])), max(0, min(1, state[3]))
	if s.V >= s.VThreshold && previousV < s.VThreshold {
		return 1
	}
	return 0
}

func SimulateBertramPhantom(steps int, current float64) ([]float64, int) {
	state := NewBertramPhantom()
	trace := make([]float64, steps)
	events := 0
	for index := range trace {
		if state.Step(current) == 1 {
			events++
		}
		trace[index] = state.V
	}
	return trace, events
}
