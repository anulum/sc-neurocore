// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dcn_neuron

package services

import (
	"math"
)

// DCNNeuronState holds the neuron state
type DCNNeuronState struct {
	V          float64
	H          float64
	N          float64
	P          float64
	S          float64
	R          float64
	Ca         float64
	GNa        float64
	GNap       float64
	GK         float64
	GT         float64
	GAhp       float64
	GH         float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EH         float64
	EL         float64
	CM         float64
	Phi        float64
	TauCa      float64
	KdAhp      float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewDCNNeuron creates a new DCNNeuron neuron with default parameters
func NewDCNNeuron() *DCNNeuronState {
	return &DCNNeuronState{
		V:          -60.0,
		H:          0.6,
		N:          0.32,
		P:          0.01,
		S:          0.8,
		R:          0.1,
		Ca:         0.05,
		GNa:        35.0,
		GNap:       0.5,
		GK:         9.0,
		GT:         0.1,
		GAhp:       2.0,
		GH:         0.02,
		GL:         0.2,
		ENa:        55.0,
		EK:         -90.0,
		ECa:        120.0,
		EH:         -40.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		TauCa:      150.0,
		KdAhp:      0.5,
		Dt:         0.5,
		VThreshold: -20.0,
		Gain:       1.0,
		SubSteps:   20,
	}
}

// Step advances the neuron by one timestep
func (s *DCNNeuronState) Step(iExt float64) int {
	if !s.valid() || math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0
	}
	input := s.Gain * iExt
	subDt := s.Dt / float64(s.SubSteps)
	fired := 0
	v, h, n, p, gateS, r, ca := s.V, s.H, s.N, s.P, s.S, s.R, s.Ca
	for i := 0; i < s.SubSteps; i++ {
		alphaM := safeRateDCN(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := safeRateDCN(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		pInf := 1.0 / (1.0 + math.Exp(-(v+48.0)/5.0))
		tauP := 5.0 + 15.0/math.Max(0.01, 1.0+math.Pow((v+48.0)/10.0, 2))
		mtInf := 1.0 / (1.0 + math.Exp(-(v+52.0)/5.0))
		sInf := 1.0 / (1.0 + math.Exp((v+60.0)/6.5))
		tauS := 20.0 + 50.0/(1.0+math.Exp((v+65.0)/10.0))
		rInf := 1.0 / (1.0 + math.Exp((v+80.0)/10.0))
		tauR := 100.0 + 200.0/(1.0+math.Exp((v+70.0)/10.0))
		h = exactHHGateDCN(h, alphaH, betaH, s.Phi, subDt)
		n = exactHHGateDCN(n, alphaN, betaN, s.Phi, subDt)
		p = exactRelaxDCN(p, pInf, tauP, subDt)
		gateS = exactRelaxDCN(gateS, sInf, tauS, subDt)
		r = exactRelaxDCN(r, rInf, tauR, subDt)
		iT := s.GT * math.Pow(mtInf, 2) * gateS * (v - s.ECa)
		caEntry := 0.0
		if iT < 0.0 {
			caEntry = -iT * 0.001
		}
		ca = exactRelaxDCN(ca, caEntry*s.TauCa, s.TauCa, subDt)
		ca = math.Max(0.0, ca)
		ahpInf := math.Pow(ca, 2) / (math.Pow(ca, 2) + math.Pow(s.KdAhp, 2))
		gNaEff := s.GNa * math.Pow(mInf, 3) * h
		gNapEff := s.GNap * p
		gKEff := s.GK * math.Pow(n, 4)
		gTEff := s.GT * math.Pow(mtInf, 2) * gateS
		gAhpEff := s.GAhp * ahpInf
		gHEff := s.GH * r
		v = exactVoltageDCN(v, input, s.CM, subDt, [][2]float64{
			{gNaEff, s.ENa},
			{gNapEff, s.ENa},
			{gKEff, s.EK},
			{gTEff, s.ECa},
			{gAhpEff, s.EK},
			{gHEff, s.EH},
			{s.GL, s.EL},
		})
		if v >= s.VThreshold {
			fired = 1
			v = -60.0
			gateS *= 0.5
			ca += 0.5
		}
	}
	candidates := []float64{v, h, n, p, gateS, r, ca}
	for _, candidate := range candidates {
		if math.IsNaN(candidate) || math.IsInf(candidate, 0) {
			return 0
		}
	}
	s.V = math.Max(-100.0, math.Min(60.0, v))
	s.H = clamp01DCN(h)
	s.N = clamp01DCN(n)
	s.P = clamp01DCN(p)
	s.S = clamp01DCN(gateS)
	s.R = clamp01DCN(r)
	s.Ca = math.Max(0.0, ca)
	return fired
}

func safeRateDCN(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

func exactRelaxDCN(value, target, tau, dt float64) float64 {
	return target + (value-target)*math.Exp(-dt/tau)
}

func exactHHGateDCN(value, alpha, beta, phi, dt float64) float64 {
	rate := phi * (alpha + beta)
	target := alpha / (alpha + beta)
	return target + (value-target)*math.Exp(-rate*dt)
}

func exactVoltageDCN(v, inputCurrent, cM, dt float64, conductances [][2]float64) float64 {
	gTotal := 0.0
	reversalDrive := 0.0
	for _, pair := range conductances {
		gTotal += pair[0]
		reversalDrive += pair[0] * pair[1]
	}
	if gTotal <= 0.0 {
		return v + dt*inputCurrent/cM
	}
	vInf := (inputCurrent + reversalDrive) / gTotal
	return vInf + (v-vInf)*math.Exp(-dt*gTotal/cM)
}

func clamp01DCN(x float64) float64 {
	return math.Max(0.0, math.Min(1.0, x))
}

func (s *DCNNeuronState) valid() bool {
	values := []float64{
		s.V, s.H, s.N, s.P, s.S, s.R, s.Ca, s.GNa, s.GNap, s.GK, s.GT,
		s.GAhp, s.GH, s.GL, s.ENa, s.EK, s.ECa, s.EH, s.EL, s.CM,
		s.Phi, s.TauCa, s.KdAhp, s.Dt, s.VThreshold, s.Gain,
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	if s.V < -100.0 || s.V > 60.0 || s.Ca < 0.0 || s.CM <= 0.0 || s.Phi <= 0.0 || s.TauCa <= 0.0 ||
		s.KdAhp <= 0.0 || s.Dt <= 0.0 || s.Gain < 0.0 || s.SubSteps <= 0 {
		return false
	}
	for _, gate := range []float64{s.H, s.N, s.P, s.S, s.R} {
		if gate < 0.0 || gate > 1.0 {
			return false
		}
	}
	for _, conductance := range []float64{s.GNa, s.GNap, s.GK, s.GT, s.GAhp, s.GH, s.GL} {
		if conductance < 0.0 {
			return false
		}
	}
	return true
}

// SimulateDCNNeuron runs the neuron for n steps
func SimulateDCNNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewDCNNeuron()
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

var _ = math.Exp
