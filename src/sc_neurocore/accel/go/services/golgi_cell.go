// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for golgi_cell

package services

import (
	"math"
)

// GolgiCellState holds the neuron state
type GolgiCellState struct {
	V        float64
	M        float64
	H        float64
	PNa      float64
	N        float64
	A        float64
	B        float64
	W        float64
	MT       float64
	S        float64
	CN       float64
	R        float64
	Ca       float64
	GNaT     float64
	GNaP     float64
	GKdr     float64
	GKa      float64
	GKm      float64
	GCat     float64
	GCan     float64
	GBK      float64
	GSK      float64
	GH       float64
	GL       float64
	ENa      float64
	EK       float64
	ECa      float64
	EH       float64
	EL       float64
	CM       float64
	TauCa    float64
	KdBK     float64
	KdSK     float64
	Dt       float64
	SubSteps int
	Gain     float64
}

// NewGolgiCell creates a new GolgiCell neuron with default parameters
func NewGolgiCell() *GolgiCellState {
	return &GolgiCellState{
		V:        -60.0,
		M:        0.02,
		H:        0.85,
		PNa:      0.01,
		N:        0.05,
		A:        0.1,
		B:        0.8,
		W:        0.01,
		MT:       0.01,
		S:        0.9,
		CN:       0.01,
		R:        0.1,
		Ca:       0.05,
		GNaT:     48.0,
		GNaP:     0.2,
		GKdr:     16.0,
		GKa:      8.0,
		GKm:      1.0,
		GCat:     0.5,
		GCan:     1.0,
		GBK:      3.0,
		GSK:      1.0,
		GH:       0.1,
		GL:       0.05,
		ENa:      55.0,
		EK:       -90.0,
		ECa:      120.0,
		EH:       -40.0,
		EL:       -55.0,
		CM:       1.0,
		TauCa:    200.0,
		KdBK:     1.0,
		KdSK:     0.5,
		Dt:       0.5,
		SubSteps: 10,
		Gain:     1.0,
	}
}

func golgiFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func golgiVoltage(value float64) bool {
	return golgiFinite(value) && value >= -100.0 && value <= 60.0
}

func golgiProbability(value float64) bool {
	return golgiFinite(value) && value >= 0.0 && value <= 1.0
}

func golgiSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

func golgiBoltz(v, vh, k float64) float64 {
	x := (v - vh) / k
	if x >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	ex := math.Exp(x)
	return ex / (1.0 + ex)
}

func golgiGateAlphaBeta(previous, alpha, beta, phi, dt float64) (float64, bool) {
	total := phi * (alpha + beta)
	if !golgiFinite(previous, alpha, beta, total, dt) || total <= 0.0 {
		return previous, false
	}
	steady := alpha / (alpha + beta)
	return math.Max(0.0, math.Min(1.0, steady+(previous-steady)*math.Exp(-total*dt))), true
}

func golgiGateInf(previous, steady, tau, dt float64) (float64, bool) {
	if !golgiFinite(previous, steady, tau, dt) || tau <= 0.0 {
		return previous, false
	}
	return math.Max(0.0, math.Min(1.0, steady+(previous-steady)*math.Exp(-dt/tau))), true
}

func golgiCalcium(previous, entry, tau, dt float64) (float64, bool) {
	if !golgiFinite(previous, entry, tau, dt) || tau <= 0.0 || previous < 0.0 {
		return previous, false
	}
	steady := entry * tau
	value := steady + (previous-steady)*math.Exp(-dt/tau)
	if !golgiFinite(value) {
		return previous, false
	}
	return math.Max(0.0, value), true
}

func (s *GolgiCellState) valid() bool {
	gates := []float64{s.M, s.H, s.PNa, s.N, s.A, s.B, s.W, s.MT, s.S, s.CN, s.R}
	for _, gate := range gates {
		if !golgiProbability(gate) {
			return false
		}
	}
	conductances := []float64{s.GNaT, s.GNaP, s.GKdr, s.GKa, s.GKm, s.GCat, s.GCan, s.GBK, s.GSK, s.GH, s.GL}
	for _, conductance := range conductances {
		if !golgiFinite(conductance) || conductance < 0.0 {
			return false
		}
	}
	return golgiVoltage(s.V) &&
		golgiFinite(s.Ca, s.ENa, s.EK, s.ECa, s.EH, s.EL, s.CM, s.TauCa, s.KdBK, s.KdSK, s.Dt, s.Gain) &&
		s.Ca >= 0.0 &&
		s.CM > 0.0 &&
		s.TauCa > 0.0 &&
		s.KdBK > 0.0 &&
		s.KdSK > 0.0 &&
		s.Dt > 0.0 &&
		s.SubSteps > 0 &&
		s.Gain >= 0.0
}

// Step advances the neuron by one timestep
func (s *GolgiCellState) Step(iExt float64) int {
	if !golgiFinite(iExt) || !s.valid() {
		return 0
	}

	next := *s
	vPrev := s.V
	input := s.Gain * iExt
	dtSub := s.Dt / float64(s.SubSteps)
	for i := 0; i < s.SubSteps; i++ {
		v := next.V
		alphaM := golgiSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		m, ok := golgiGateAlphaBeta(next.M, alphaM, betaM, 5.0, dtSub)
		if !ok {
			return 0
		}
		h, ok := golgiGateAlphaBeta(next.H, alphaH, betaH, 5.0, dtSub)
		if !ok {
			return 0
		}
		tauPNa := 5.0 + 20.0/math.Max(0.01, 1.0+math.Pow((v+48.0)/10.0, 2.0))
		pNa, ok := golgiGateInf(next.PNa, golgiBoltz(v, -48.0, 5.0), tauPNa, dtSub)
		if !ok {
			return 0
		}
		alphaN := golgiSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		n, ok := golgiGateAlphaBeta(next.N, alphaN, betaN, 5.0, dtSub)
		if !ok {
			return 0
		}
		a, ok := golgiGateInf(next.A, golgiBoltz(v, -27.0, 16.0), 2.0, dtSub)
		if !ok {
			return 0
		}
		b, ok := golgiGateInf(next.B, golgiBoltz(v, -80.0, -6.0), 15.0, dtSub)
		if !ok {
			return 0
		}
		tauW := 100.0 / (3.3*math.Exp((v+35.0)/20.0) + math.Exp(-(v+35.0)/20.0))
		w, ok := golgiGateInf(next.W, golgiBoltz(v, -35.0, 10.0), tauW, dtSub)
		if !ok {
			return 0
		}
		mt, ok := golgiGateInf(next.MT, golgiBoltz(v, -52.0, 5.0), 1.0, dtSub)
		if !ok {
			return 0
		}
		tauS := 20.0 + 50.0/math.Max(0.01, 1.0+math.Pow((v+65.0)/10.0, 2.0))
		sGate, ok := golgiGateInf(next.S, golgiBoltz(v, -60.0, -6.5), tauS, dtSub)
		if !ok {
			return 0
		}
		tauCN := 2.0 + 10.0/math.Max(0.01, 1.0+math.Pow((v+20.0)/10.0, 2.0))
		cn, ok := golgiGateInf(next.CN, golgiBoltz(v, -20.0, 5.0), tauCN, dtSub)
		if !ok {
			return 0
		}
		tauR := 50.0 + 200.0/math.Max(0.01, 1.0+math.Pow((v+80.0)/20.0, 2.0))
		r, ok := golgiGateInf(next.R, golgiBoltz(v, -80.0, -10.0), tauR, dtSub)
		if !ok {
			return 0
		}

		gCat := s.GCat * math.Pow(mt, 2.0) * sGate
		gCan := s.GCan * math.Pow(cn, 2.0)
		iCa := gCat*(v-s.ECa) + gCan*(v-s.ECa)
		caEntry := 0.0
		if iCa < 0.0 {
			caEntry = -iCa * 0.001
		}
		ca, ok := golgiCalcium(next.Ca, caEntry, s.TauCa, dtSub)
		if !ok {
			return 0
		}
		ca2 := ca * ca
		bkV := golgiBoltz(v, 100.0-120.0*ca2/(ca2+math.Pow(s.KdBK, 2.0)), 15.0)
		skInf := ca2 / (ca2 + math.Pow(s.KdSK, 2.0))
		gNa := s.GNaT*math.Pow(m, 3.0)*h + s.GNaP*pNa
		gK := s.GKdr*math.Pow(n, 4.0) + s.GKa*math.Pow(a, 3.0)*b + s.GKm*w + s.GBK*bkV + s.GSK*skInf
		gCa := gCat + gCan
		gH := s.GH * r
		gTotal := gNa + gK + gCa + gH + s.GL
		if !golgiFinite(gTotal) || gTotal <= 0.0 {
			return 0
		}
		steadyV := (input + gNa*s.ENa + gK*s.EK + gCa*s.ECa + gH*s.EH + s.GL*s.EL) / gTotal
		vNext := steadyV + (v-steadyV)*math.Exp(-(gTotal/s.CM)*dtSub)
		if !golgiVoltage(vNext) || !golgiFinite(ca) || ca < 0.0 {
			return 0
		}

		next.V = vNext
		next.M = m
		next.H = h
		next.PNa = pNa
		next.N = n
		next.A = a
		next.B = b
		next.W = w
		next.MT = mt
		next.S = sGate
		next.CN = cn
		next.R = r
		next.Ca = ca
	}

	*s = next
	if s.V >= 0.0 && vPrev < 0.0 {
		return 1
	}
	return 0
}

// SimulateGolgiCell runs the neuron for n steps
func SimulateGolgiCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewGolgiCell()
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
