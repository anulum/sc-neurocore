// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for granule_cell

package services

import "math"

// GranuleCellState holds the D'Angelo-style cerebellar granule-cell state.
type GranuleCellState struct {
	V        float64
	M        float64
	H        float64
	N        float64
	A        float64
	B        float64
	MT       float64
	S        float64
	Ca       float64
	R        float64
	CM       float64
	GNa      float64
	GKdr     float64
	GKa      float64
	GT       float64
	GKca     float64
	GH       float64
	GL       float64
	GTonic   float64
	ENa      float64
	EK       float64
	ECa      float64
	EH       float64
	EL       float64
	EGaba    float64
	TauCa    float64
	KdKca    float64
	Dt       float64
	SubSteps int
	Gain     float64
}

// NewGranuleCell creates a new GranuleCell neuron with default parameters.
func NewGranuleCell() *GranuleCellState {
	return &GranuleCellState{
		V:        -70.0,
		M:        0.02,
		H:        0.85,
		N:        0.05,
		A:        0.1,
		B:        0.8,
		MT:       0.01,
		S:        0.95,
		Ca:       0.05,
		R:        0.1,
		CM:       1.0,
		GNa:      17.0,
		GKdr:     9.0,
		GKa:      1.0,
		GT:       0.5,
		GKca:     3.5,
		GH:       0.03,
		GL:       0.1,
		GTonic:   0.2,
		ENa:      87.4,
		EK:       -84.7,
		ECa:      129.3,
		EH:       -40.0,
		EL:       -58.0,
		EGaba:    -75.0,
		TauCa:    10.0,
		KdKca:    0.2,
		Dt:       0.5,
		SubSteps: 4,
		Gain:     1.0,
	}
}

func boltzGranule(v float64, vh float64, k float64) float64 {
	z := -(v - vh) / k
	if z > 60.0 {
		return 0.0
	}
	if z < -60.0 {
		return 1.0
	}
	return 1.0 / (1.0 + math.Exp(z))
}

func clamp01Granule(value float64) float64 {
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

func finiteGranule(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func gateRangeGranule(values ...float64) bool {
	for _, value := range values {
		if value < 0.0 || value > 1.0 {
			return false
		}
	}
	return true
}

func (s *GranuleCellState) valid() bool {
	return finiteGranule(
		s.V, s.M, s.H, s.N, s.A, s.B, s.MT, s.S, s.Ca, s.R,
		s.CM, s.GNa, s.GKdr, s.GKa, s.GT, s.GKca, s.GH, s.GL, s.GTonic,
		s.ENa, s.EK, s.ECa, s.EH, s.EL, s.EGaba, s.TauCa, s.KdKca, s.Dt, s.Gain,
	) &&
		gateRangeGranule(s.M, s.H, s.N, s.A, s.B, s.MT, s.S, s.R) &&
		s.Ca >= 0.0 &&
		s.GNa >= 0.0 && s.GKdr >= 0.0 && s.GKa >= 0.0 && s.GT >= 0.0 &&
		s.GKca >= 0.0 && s.GH >= 0.0 && s.GL >= 0.0 && s.GTonic >= 0.0 &&
		s.CM > 0.0 && s.TauCa > 0.0 && s.KdKca > 0.0 && s.Dt > 0.0 &&
		s.SubSteps > 0 && s.Gain >= 0.0
}

// Step advances the neuron by one timestep. Invalid input or state leaves the
// old state untouched and returns no spike.
func (s *GranuleCellState) Step(iExt float64) int {
	if !s.valid() || !finiteGranule(iExt) {
		return 0
	}

	input := s.Gain * iExt
	dtSub := s.Dt / float64(s.SubSteps)
	vPrev := s.V
	v := s.V
	m := s.M
	h := s.H
	n := s.N
	a := s.A
	b := s.B
	mT := s.MT
	gateS := s.S
	ca := s.Ca
	r := s.R

	for step := 0; step < s.SubSteps; step++ {
		mInf := boltzGranule(v, -30.0, 7.0)
		tauM := 0.1 + 0.3/math.Max(0.01, 1.0+math.Pow((v+30.0)/10.0, 2.0))
		m = clamp01Granule(m + dtSub*(mInf-m)/tauM)

		hInf := boltzGranule(v, -52.0, -6.0)
		tauH := 0.5 + 5.0/math.Max(0.01, 1.0+math.Pow((v+50.0)/15.0, 2.0))
		h = clamp01Granule(h + dtSub*(hInf-h)/tauH)

		nInf := boltzGranule(v, -35.0, 8.0)
		tauN := 1.0 + 5.0/math.Max(0.01, 1.0+math.Pow((v+35.0)/15.0, 2.0))
		n = clamp01Granule(n + dtSub*(nInf-n)/tauN)

		aInf := boltzGranule(v, -50.0, 20.0)
		a = clamp01Granule(a + dtSub*(aInf-a)/2.0)

		bInf := boltzGranule(v, -70.0, -6.0)
		b = clamp01Granule(b + dtSub*(bInf-b)/50.0)

		mtInf := boltzGranule(v, -52.0, 5.0)
		mT = clamp01Granule(mT + dtSub*(mtInf-mT))

		sInf := boltzGranule(v, -60.0, -6.5)
		tauS := 20.0 + 50.0/math.Max(0.01, 1.0+math.Pow((v+65.0)/10.0, 2.0))
		gateS = clamp01Granule(gateS + dtSub*(sInf-gateS)/tauS)

		rInf := boltzGranule(v, -80.0, -10.0)
		tauR := 50.0 + 200.0/math.Max(0.01, 1.0+math.Pow((v+80.0)/20.0, 2.0))
		r = clamp01Granule(r + dtSub*(rInf-r)/tauR)

		iCaT := s.GT * mT * mT * gateS * (v - s.ECa)
		caEntry := 0.0
		if iCaT < 0.0 {
			caEntry = -iCaT * 0.001
		}
		ca = math.Max(0.0, ca+dtSub*(-ca/s.TauCa+caEntry))

		kcaInf := ca * ca / (ca*ca + s.KdKca*s.KdKca)
		iNa := s.GNa * math.Pow(m, 3.0) * h * (v - s.ENa)
		iKdr := s.GKdr * math.Pow(n, 4.0) * (v - s.EK)
		iKa := s.GKa * math.Pow(a, 3.0) * b * (v - s.EK)
		iKca := s.GKca * kcaInf * (v - s.EK)
		iH := s.GH * r * (v - s.EH)
		iL := s.GL * (v - s.EL)
		iGaba := s.GTonic * (v - s.EGaba)

		dv := (-(iNa + iKdr + iKa + iCaT + iKca + iH + iL + iGaba) + input) / s.CM
		v = math.Max(-100.0, math.Min(60.0, v+dtSub*dv))

		if !finiteGranule(v, m, h, n, a, b, mT, gateS, ca, r) {
			return 0
		}
	}

	s.V = v
	s.M = m
	s.H = h
	s.N = n
	s.A = a
	s.B = b
	s.MT = mT
	s.S = gateS
	s.Ca = ca
	s.R = r

	if s.V >= 0.0 && vPrev < 0.0 {
		return 1
	}
	return 0
}

// SimulateGranuleCell runs the neuron for n steps.
func SimulateGranuleCell(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewGranuleCell()
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
