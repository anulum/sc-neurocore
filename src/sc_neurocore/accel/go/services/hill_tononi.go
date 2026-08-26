// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Hill-Tononi 2005 hybrid neuron

package services

import (
	"errors"
	"math"
)

// HillTononiNeuronState holds the source continuous state and cell parameters.
type HillTononiNeuronState struct {
	V, Theta, DK, MH, MT, HT, SpikeTimer float64
	GNaL, GKL, GNaP, GDK, GH, GT         float64
	ENa, EK, ENaP, EDK, EH, ET           float64
	NNaP, NT                             float64
	TauM, ThetaEq, TauTheta              float64
	GSpike, TSpike, TauSpike             float64
	TauD, DInfluxPeak, DThreshold        float64
	DSlope, DEq, DHalf, Dt               float64
}

// NewHillTononiNeuron returns the paper's cortical-excitatory waking profile.
func NewHillTononiNeuron() *HillTononiNeuronState {
	return &HillTononiNeuronState{
		V: -70.0, Theta: -51.0, DK: 0.001,
		MH: 0.2871859013825026, MT: 0.1450215950687922,
		HT: 0.03732688734412946, SpikeTimer: 0.0,
		GNaL: 0.2, GKL: 1.0, GNaP: 0.5, GDK: 0.5, GH: 0.0, GT: 0.0,
		ENa: 30.0, EK: -90.0, ENaP: 30.0, EDK: -90.0, EH: -40.0, ET: 0.0,
		NNaP: 3.0, NT: 2.0,
		TauM: 16.0, ThetaEq: -51.0, TauTheta: 2.0,
		GSpike: 1.0, TSpike: 2.0, TauSpike: 1.75,
		TauD: 1250.0, DInfluxPeak: 0.025, DThreshold: -10.0,
		DSlope: 5.0, DEq: 0.001, DHalf: 0.25, Dt: 0.25,
	}
}

func mHInf(v float64) float64 {
	return 1.0 / (1.0 + math.Exp((v+75.0)/5.5))
}

func tauMH(v float64) float64 {
	return 1.0 / (math.Exp(-14.59-0.086*v) + math.Exp(-1.87+0.0701*v))
}

func mTInf(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-(v+59.0)/6.2))
}

func tauMT(v float64) float64 {
	return 0.22/(math.Exp(-(v+132.0)/16.7)+math.Exp((v+16.8)/18.2)) + 0.13
}

func hTInf(v float64) float64 {
	return 1.0 / (1.0 + math.Exp((v+83.0)/4.0))
}

func tauHT(v float64) float64 {
	return 8.2 + (56.6+0.27*math.Exp((v+115.2)/5.0))/(1.0+math.Exp((v+86.0)/3.2))
}

func (s *HillTononiNeuronState) dKInf(v float64) float64 {
	influx := s.DInfluxPeak / (1.0 + math.Exp(-(v-s.DThreshold)/s.DSlope))
	return s.TauD*influx + s.DEq
}

func (s *HillTononiNeuronState) derivatives(state [6]float64, current float64, spikeActive bool) [6]float64 {
	v, theta, dK := state[0], state[1], state[2]
	mH, mT, hT := state[3], state[4], state[5]
	mNaP := 1.0 / (1.0 + math.Exp(-(v+55.7)/7.7))
	dActivation := 1.0 / (1.0 + math.Pow(s.DHalf/math.Max(dK, 1e-15), 3.5))
	iNaL := -s.GNaL * (v - s.ENa)
	iKL := -s.GKL * (v - s.EK)
	iNaP := -s.GNaP * math.Pow(mNaP, s.NNaP) * (v - s.ENaP)
	iDK := -s.GDK * dActivation * (v - s.EDK)
	iH := -s.GH * mH * (v - s.EH)
	iT := -s.GT * math.Pow(mT, s.NT) * hT * (v - s.ET)
	iSpike := 0.0
	if spikeActive {
		iSpike = -s.GSpike * (v - s.EK) / s.TauSpike
	}
	return [6]float64{
		(iNaL+iKL+iNaP+iDK+iH+iT+current)/s.TauM + iSpike,
		-(theta - s.ThetaEq) / s.TauTheta,
		(s.dKInf(v) - dK) / s.TauD,
		(mHInf(v) - mH) / tauMH(v),
		(mTInf(v) - mT) / tauMT(v),
		(hTInf(v) - hT) / tauHT(v),
	}
}

func shifted(state, slope [6]float64, scale float64) [6]float64 {
	var out [6]float64
	for i := range state {
		out[i] = state[i] + scale*slope[i]
	}
	return out
}

func (s *HillTononiNeuronState) candidate(state [6]float64, current float64, spikeActive bool) [6]float64 {
	k1 := s.derivatives(state, current, spikeActive)
	k2 := s.derivatives(shifted(state, k1, 0.5*s.Dt), current, spikeActive)
	k3 := s.derivatives(shifted(state, k2, 0.5*s.Dt), current, spikeActive)
	k4 := s.derivatives(shifted(state, k3, s.Dt), current, spikeActive)
	var out [6]float64
	for i := range state {
		out[i] = state[i] + s.Dt*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0
	}
	return out
}

func hillTononiFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *HillTononiNeuronState) configurationIsValid() bool {
	if !hillTononiFinite(
		s.V, s.Theta, s.DK, s.MH, s.MT, s.HT, s.SpikeTimer,
		s.GNaL, s.GKL, s.GNaP, s.GDK, s.GH, s.GT,
		s.ENa, s.EK, s.ENaP, s.EDK, s.EH, s.ET, s.NNaP, s.NT,
		s.TauM, s.ThetaEq, s.TauTheta, s.GSpike, s.TSpike, s.TauSpike,
		s.TauD, s.DInfluxPeak, s.DThreshold, s.DSlope, s.DEq, s.DHalf, s.Dt,
	) {
		return false
	}
	if s.DK < 0.0 || s.SpikeTimer < 0.0 {
		return false
	}
	for _, value := range []float64{s.GNaL, s.GKL, s.GNaP, s.GDK, s.GH, s.GT, s.GSpike, s.DInfluxPeak, s.DEq} {
		if value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.NNaP, s.NT, s.TauM, s.TauTheta, s.TSpike, s.TauSpike, s.TauD, s.DSlope, s.DHalf, s.Dt} {
		if value <= 0.0 {
			return false
		}
	}
	return true
}

// TryStep advances one source RK4 step and leaves state unchanged on error.
func (s *HillTononiNeuronState) TryStep(current float64) (int, error) {
	if !hillTononiFinite(current) || !s.configurationIsValid() {
		return 0, errors.New("Hill-Tononi configuration and current must be finite and physical")
	}
	refractory := s.SpikeTimer > 0.0
	state := [6]float64{s.V, s.Theta, s.DK, s.MH, s.MT, s.HT}
	next := s.candidate(state, current, refractory)
	if !hillTononiFinite(next[:]...) || next[2] < 0.0 {
		return 0, errors.New("Hill-Tononi candidate must be finite and physical")
	}
	timer := math.Max(0.0, s.SpikeTimer-s.Dt)
	spike := !refractory && next[0] >= next[1]
	if spike {
		next[0], next[1], timer = s.ENa, s.ENa, s.TSpike
	}
	s.V, s.Theta, s.DK = next[0], next[1], next[2]
	s.MH, s.MT, s.HT, s.SpikeTimer = next[3], next[4], next[5], timer
	if spike {
		return 1, nil
	}
	return 0, nil
}

// Step preserves the historical integer-return API and fails closed on error.
func (s *HillTononiNeuronState) Step(current float64) int {
	spike, err := s.TryStep(current)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores the source cortical-excitatory waking initial state.
func (s *HillTononiNeuronState) Reset() {
	s.V, s.Theta, s.DK = -70.0, -51.0, 0.001
	s.MH, s.MT, s.HT = mHInf(s.V), mTInf(s.V), hTInf(s.V)
	s.SpikeTimer = 0.0
}

// SimulateHillTononiNeuron runs the public step surface for n steps.
func SimulateHillTononiNeuron(nSteps int, current float64) ([]float64, int) {
	s := NewHillTononiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for step := 0; step < nSteps; step++ {
		spikes += s.Step(current)
		trace[step] = s.V
	}
	return trace, spikes
}
