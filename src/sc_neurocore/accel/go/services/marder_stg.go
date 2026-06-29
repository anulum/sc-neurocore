// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for marder_stg (LGMA98 STG, RK4)

// Liu-Golowasch-Marder-Abbott 1998 stomatogastric ganglion neuron with a
// fourth-order Runge-Kutta integrator. Mirrors neurons/models/marder_stg.py:
// thirteen states, voltage-dependent time constants, Nernst calcium reversal.
// ModelDB 93321.
package services

import (
	"math"
)

// MarderSTGNeuronState holds the thirteen-state STG neuron and its parameters.
type MarderSTGNeuronState struct {
	V          float64
	MNa        float64
	HNa        float64
	MCat       float64
	HCat       float64
	MCas       float64
	HCas       float64
	MA         float64
	HA         float64
	MKca       float64
	MKd        float64
	MH         float64
	Ca         float64
	Cm         float64
	GNa        float64
	GCat       float64
	GCas       float64
	GA         float64
	GKca       float64
	GKd        float64
	GH         float64
	GL         float64
	ENa        float64
	EK         float64
	EH         float64
	EL         float64
	CaOut      float64
	CaRest     float64
	TauCa      float64
	FCa        float64
	Celsius    float64
	Dt         float64
	VThreshold float64
}

// NewMarderSTGNeuron creates a MarderSTGNeuron with the published defaults.
func NewMarderSTGNeuron() *MarderSTGNeuronState {
	return &MarderSTGNeuronState{
		V: -60.0, MNa: 0.0, HNa: 1.0, MCat: 0.0, HCat: 1.0, MCas: 0.0, HCas: 1.0,
		MA: 0.0, HA: 1.0, MKca: 0.0, MKd: 0.0, MH: 0.0, Ca: 0.05, Cm: 1.0,
		GNa: 200.0, GCat: 2.5, GCas: 4.0, GA: 50.0, GKca: 25.0, GKd: 75.0, GH: 0.01, GL: 0.01,
		ENa: 50.0, EK: -80.0, EH: -20.0, EL: -50.0,
		CaOut: 3000.0, CaRest: 0.05, TauCa: 20.0, FCa: 0.94, Celsius: 10.0,
		Dt: 0.05, VThreshold: -20.0,
	}
}

func msExp(x float64) float64 {
	if x > 700.0 {
		x = 700.0
	} else if x < -700.0 {
		x = -700.0
	}
	return math.Exp(x)
}

func msSig(v, vh, s float64) float64 {
	return 1.0 / (1.0 + msExp((vh-v)/s))
}

func (s *MarderSTGNeuronState) nernstECa(ca float64) float64 {
	rtZf := 1000.0 * 8.314462618 * (s.Celsius + 273.15) / (2.0 * 96485.33212)
	return rtZf * math.Log(s.CaOut/math.Max(ca, 1e-9))
}

// derivatives returns d/dt of (V, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, Ca).
func (s *MarderSTGNeuronState) derivatives(y [13]float64, current float64) [13]float64 {
	v := y[0]
	mNa, hNa := y[1], y[2]
	mCat, hCat := y[3], y[4]
	mCas, hCas := y[5], y[6]
	mA, hA := y[7], y[8]
	mKca, mKd, mH, ca := y[9], y[10], y[11], y[12]

	tauMNa := 1.32 - 1.26/(1.0+msExp(-(v+120.0)/25.0))
	tauHNa := (0.67 / (1.0 + msExp(-(v+62.9)/10.0))) * (1.5 + 1.0/(1.0+msExp((v+34.9)/3.6)))
	tauMCat := 21.7 - 21.3/(1.0+msExp(-(v+68.1)/20.5))
	tauHCat := 105.0 - 89.8/(1.0+msExp(-(v+55.0)/16.9))
	tauMCas := 1.4 + 7.0/(msExp((v+27.0)/10.0)+msExp(-(v+70.0)/13.0))
	tauHCas := 60.0 + 150.0/(msExp((v+55.0)/9.0)+msExp(-(v+65.0)/16.0))
	tauMA := 11.6 - 10.4/(1.0+msExp(-(v+32.9)/15.2))
	tauHA := 38.6 - 29.2/(1.0+msExp(-(v+38.9)/26.5))
	tauMKca := 90.3 - 75.1/(1.0+msExp(-(v+46.0)/22.7))
	tauMKd := 7.2 - 6.4/(1.0+msExp(-(v+28.3)/19.2))
	tauMH := 272.0 + 1499.0/(1.0+msExp(-(v+42.2)/8.73))

	mKcaInf := (ca / (ca + 3.0)) * msSig(v, -28.3, 12.6)
	eCa := s.nernstECa(ca)
	iNa := s.GNa * mNa * mNa * mNa * hNa * (v - s.ENa)
	iCat := s.GCat * mCat * mCat * mCat * hCat * (v - eCa)
	iCas := s.GCas * mCas * mCas * mCas * hCas * (v - eCa)
	iA := s.GA * mA * mA * mA * hA * (v - s.EK)
	iKca := s.GKca * mKca * mKca * mKca * mKca * (v - s.EK)
	iKd := s.GKd * mKd * mKd * mKd * mKd * (v - s.EK)
	iH := s.GH * mH * (v - s.EH)
	iL := s.GL * (v - s.EL)

	dv := (current - iNa - iCat - iCas - iA - iKca - iKd - iH - iL) / s.Cm
	dca := (-s.FCa*(iCat+iCas) - (ca - s.CaRest)) / s.TauCa
	return [13]float64{
		dv,
		(msSig(v, -25.5, 5.29) - mNa) / tauMNa,
		(msSig(v, -48.9, -5.18) - hNa) / tauHNa,
		(msSig(v, -27.1, 7.2) - mCat) / tauMCat,
		(msSig(v, -32.1, -5.5) - hCat) / tauHCat,
		(msSig(v, -33.0, 8.1) - mCas) / tauMCas,
		(msSig(v, -60.0, -6.2) - hCas) / tauHCas,
		(msSig(v, -27.2, 8.7) - mA) / tauMA,
		(msSig(v, -56.9, -4.9) - hA) / tauHA,
		(mKcaInf - mKca) / tauMKca,
		(msSig(v, -12.3, 11.8) - mKd) / tauMKd,
		(msSig(v, -70.0, -6.0) - mH) / tauMH,
		dca,
	}
}

func msAxpy(y, k [13]float64, f float64) [13]float64 {
	var out [13]float64
	for i := 0; i < 13; i++ {
		out[i] = y[i] + f*k[i]
	}
	return out
}

func clampUnit(value float64) float64 {
	return math.Min(math.Max(value, 0.0), 1.0)
}

// Step advances the neuron one RK4 timestep under injected current.
func (s *MarderSTGNeuronState) Step(current float64) int {
	if math.IsNaN(current) || math.IsInf(current, 0) {
		return -1
	}
	vPrev := s.V
	y := [13]float64{s.V, s.MNa, s.HNa, s.MCat, s.HCat, s.MCas, s.HCas, s.MA, s.HA, s.MKca, s.MKd, s.MH, s.Ca}
	dt := s.Dt
	k1 := s.derivatives(y, current)
	k2 := s.derivatives(msAxpy(y, k1, dt/2.0), current)
	k3 := s.derivatives(msAxpy(y, k2, dt/2.0), current)
	k4 := s.derivatives(msAxpy(y, k3, dt), current)
	var nxt [13]float64
	for i := 0; i < 13; i++ {
		nxt[i] = y[i] + (dt/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])
		if math.IsNaN(nxt[i]) || math.IsInf(nxt[i], 0) {
			return -1
		}
	}
	s.V = nxt[0]
	s.MNa = clampUnit(nxt[1])
	s.HNa = clampUnit(nxt[2])
	s.MCat = clampUnit(nxt[3])
	s.HCat = clampUnit(nxt[4])
	s.MCas = clampUnit(nxt[5])
	s.HCas = clampUnit(nxt[6])
	s.MA = clampUnit(nxt[7])
	s.HA = clampUnit(nxt[8])
	s.MKca = clampUnit(nxt[9])
	s.MKd = clampUnit(nxt[10])
	s.MH = clampUnit(nxt[11])
	s.Ca = math.Max(nxt[12], 0.0)
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateMarderSTGNeuron runs the neuron for n steps under constant drive.
func SimulateMarderSTGNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMarderSTGNeuron()
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
