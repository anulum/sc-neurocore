// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import "math"

// BoothRinzelState stores the two-compartment Booth-Rinzel motoneuron state.
type BoothRinzelState struct {
	Vs         float64
	Vd         float64
	H          float64
	N          float64
	Q          float64
	Ca         float64
	GNa        float64
	GK         float64
	GCa        float64
	GKCa       float64
	GL         float64
	GC         float64
	P          float64
	CM         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	AlphaCa    float64
	KCa        float64
	FCa        float64
	Dt         float64
	VThreshold float64
}

// NewBoothRinzelState returns a validated default Booth-Rinzel motoneuron.
func NewBoothRinzelState() *BoothRinzelState {
	return &BoothRinzelState{
		Vs:         -60.0,
		Vd:         -60.0,
		H:          0.6,
		N:          0.1,
		Q:          0.1,
		Ca:         0.1,
		GNa:        120.0,
		GK:         36.0,
		GCa:        2.0,
		GKCa:       5.0,
		GL:         0.3,
		GC:         1.0,
		P:          0.5,
		CM:         1.0,
		ENa:        50.0,
		EK:         -77.0,
		ECa:        120.0,
		EL:         -54.4,
		AlphaCa:    0.002,
		KCa:        0.01,
		FCa:        0.01,
		Dt:         0.01,
		VThreshold: 0.0,
	}
}

func finiteBoothRinzel(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func gateBoothRinzel(x float64) bool { return finiteBoothRinzel(x) && x >= 0.0 && x <= 1.0 }

func clipBoothRinzel(x, lo, hi float64) float64 { return math.Min(math.Max(x, lo), hi) }

func safeExpBoothRinzel(x float64) float64 { return math.Exp(clipBoothRinzel(x, -100.0, 100.0)) }

func validateBoothRinzelConfig(s *BoothRinzelState) bool {
	if s == nil {
		return false
	}
	strictlyPositive := []float64{s.GNa, s.GK, s.GCa, s.GKCa, s.GL, s.GC, s.CM, s.AlphaCa, s.KCa, s.FCa, s.Dt}
	for _, value := range strictlyPositive {
		if !finiteBoothRinzel(value) || value <= 0.0 {
			return false
		}
	}
	if !finiteBoothRinzel(s.P) || s.P <= 0.0 || s.P >= 1.0 {
		return false
	}
	return finiteBoothRinzel(s.ENa) && finiteBoothRinzel(s.EK) && finiteBoothRinzel(s.ECa) && finiteBoothRinzel(s.EL) && finiteBoothRinzel(s.VThreshold)
}

func validateBoothRinzelState(vs, vd, h, n, q, ca float64) bool {
	if !finiteBoothRinzel(vs) || !finiteBoothRinzel(vd) || !finiteBoothRinzel(ca) || ca < 0.0 {
		return false
	}
	if !gateBoothRinzel(h) || !gateBoothRinzel(n) || !gateBoothRinzel(q) {
		return false
	}
	return vs >= -200.0 && vs <= 100.0 && vd >= -200.0 && vd <= 100.0
}

func boothRinzelSubstep(s *BoothRinzelState, vs, vd, h, n, q, ca, current, dt float64) (float64, float64, float64, float64, float64, float64, bool) {
	if !finiteBoothRinzel(current) || !finiteBoothRinzel(dt) || dt <= 0.0 {
		return vs, vd, h, n, q, ca, false
	}
	mInf := 1.0 / (1.0 + safeExpBoothRinzel(-(vs+30.0)/9.5))
	hInf := 1.0 / (1.0 + safeExpBoothRinzel((vs+53.0)/7.0))
	nInf := 1.0 / (1.0 + safeExpBoothRinzel(-(vs+30.0)/10.0))
	qInf := 1.0 / (1.0 + safeExpBoothRinzel(-(vd+25.0)/5.0))

	tauH := 1.0 + 7.0/(safeExpBoothRinzel((vs+40.0)/5.0)+safeExpBoothRinzel(-(vs+40.0)/5.0))
	tauN := 1.0 + 5.0/(safeExpBoothRinzel((vs+35.0)/10.0)+safeExpBoothRinzel(-(vs+35.0)/10.0))
	tauQ := 10.0

	h = clipBoothRinzel(h+dt*(hInf-h)/tauH, 0.0, 1.0)
	n = clipBoothRinzel(n+dt*(nInf-n)/tauN, 0.0, 1.0)
	q = clipBoothRinzel(q+dt*(qInf-q)/tauQ, 0.0, 1.0)

	iNa := s.GNa * math.Pow(mInf, 3.0) * h * (vs - s.ENa)
	iK := s.GK * math.Pow(n, 4.0) * (vs - s.EK)
	iL := s.GL * (vs - s.EL)
	iC := s.GC * (vs - vd)
	iCa := s.GCa * q * q * (vd - s.ECa)
	iKCa := s.GKCa * (ca / (ca + s.KCa)) * (vd - s.EK)

	dVs := (current - iNa - iK - iL - iC) / (s.CM * s.P)
	dVd := (-iCa - iKCa - iL + iC) / (s.CM * (1.0 - s.P))
	dCa := -s.AlphaCa*iCa - s.FCa*ca

	vs = clipBoothRinzel(vs+dt*dVs, -200.0, 100.0)
	vd = clipBoothRinzel(vd+dt*dVd, -200.0, 100.0)
	ca = math.Max(0.0, ca+dt*dCa)
	return vs, vd, h, n, q, ca, validateBoothRinzelState(vs, vd, h, n, q, ca)
}

// Step advances the state by one configured sample and returns 1 on threshold crossing, 0 otherwise, -1 on invalid input.
func (s *BoothRinzelState) Step(current float64) int {
	if !validateBoothRinzelConfig(s) || !finiteBoothRinzel(current) || !validateBoothRinzelState(s.Vs, s.Vd, s.H, s.N, s.Q, s.Ca) {
		return -1
	}
	oldVs := s.Vs
	vs, vd, h, n, q, ca := s.Vs, s.Vd, s.H, s.N, s.Q, s.Ca
	dt := s.Dt / 4.0
	for i := 0; i < 4; i++ {
		var ok bool
		vs, vd, h, n, q, ca, ok = boothRinzelSubstep(s, vs, vd, h, n, q, ca, current, dt)
		if !ok {
			return -1
		}
	}
	s.Vs, s.Vd, s.H, s.N, s.Q, s.Ca = vs, vd, h, n, q, ca
	if oldVs < s.VThreshold && s.Vs >= s.VThreshold {
		return 1
	}
	return 0
}

// Reset restores the canonical resting state while preserving model parameters.
func (s *BoothRinzelState) Reset() {
	s.Vs = -60.0
	s.Vd = -60.0
	s.H = 0.6
	s.N = 0.1
	s.Q = 0.1
	s.Ca = 0.1
}

// Simulate returns somatic voltage samples for a constant-current stimulus.
func (s *BoothRinzelState) Simulate(current float64, steps int) []float64 {
	if steps <= 0 {
		return []float64{}
	}
	trace := make([]float64, 0, steps)
	for i := 0; i < steps; i++ {
		s.Step(current)
		trace = append(trace, s.Vs)
	}
	return trace
}
