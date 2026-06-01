// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for av_ron_cardiac

package services

import "math"

// AvRonCardiacNeuronState holds the Av-Ron cardiac ganglion state.
type AvRonCardiacNeuronState struct {
	V          float64
	H          float64
	N          float64
	S          float64
	GNa        float64
	GK         float64
	GS         float64
	GL         float64
	ENa        float64
	EK         float64
	ES         float64
	EL         float64
	Dt         float64
	VThreshold float64
}

// NewAvRonCardiacNeuron creates a new AvRonCardiacNeuron neuron with default parameters.
func NewAvRonCardiacNeuron() *AvRonCardiacNeuronState {
	return &AvRonCardiacNeuronState{
		V:          -60.0,
		H:          0.6,
		N:          0.3,
		S:          0.5,
		GNa:        80.0,
		GK:         40.0,
		GS:         20.0,
		GL:         0.1,
		ENa:        40.0,
		EK:         -80.0,
		ES:         -25.0,
		EL:         -60.0,
		Dt:         0.02,
		VThreshold: -20.0,
	}
}

func finiteAvRonValues(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func avRonGateInRange(value float64) bool {
	return value >= 0.0 && value <= 1.0
}

func avRonBoundedExp(value float64) float64 {
	return math.Exp(math.Max(math.Min(value, 709.0), -745.0))
}

func avRonSigmoidPos(value float64) float64 {
	return 1.0 / (1.0 + avRonBoundedExp(-value))
}

func avRonSigmoidNeg(value float64) float64 {
	return 1.0 / (1.0 + avRonBoundedExp(value))
}

func (s *AvRonCardiacNeuronState) validRuntime() bool {
	return finiteAvRonValues(s.V, s.H, s.N, s.S, s.GNa, s.GK, s.GS, s.GL, s.ENa, s.EK, s.ES, s.EL, s.Dt, s.VThreshold) &&
		s.Dt > 0.0 && s.GNa >= 0.0 && s.GK >= 0.0 && s.GS >= 0.0 && s.GL >= 0.0 &&
		avRonGateInRange(s.H) && avRonGateInRange(s.N) && avRonGateInRange(s.S)
}

func (s *AvRonCardiacNeuronState) rates(voltage float64) [7]float64 {
	return [7]float64{
		avRonSigmoidPos((voltage + 40.0) / 7.0),
		avRonSigmoidNeg((voltage + 45.0) / 5.0),
		avRonSigmoidPos((voltage + 40.0) / 15.0),
		avRonSigmoidNeg((voltage + 35.0) / 3.0),
		1.0 + 12.0*avRonSigmoidNeg((voltage+50.0)/8.0),
		1.0 + 8.0*avRonSigmoidNeg((voltage+35.0)/8.0),
		200.0 + 1000.0*avRonSigmoidNeg((voltage+30.0)/5.0),
	}
}

func (s *AvRonCardiacNeuronState) derivatives(state [4]float64, iExt float64) [4]float64 {
	voltage, hGate, nGate, sGate := state[0], state[1], state[2], state[3]
	if !finiteAvRonValues(voltage, hGate, nGate, sGate) || !avRonGateInRange(hGate) || !avRonGateInRange(nGate) || !avRonGateInRange(sGate) {
		return [4]float64{math.NaN(), math.NaN(), math.NaN(), math.NaN()}
	}
	rates := s.rates(voltage)
	iNa := s.GNa * math.Pow(rates[0], 3.0) * hGate * (voltage - s.ENa)
	iK := s.GK * math.Pow(nGate, 4.0) * (voltage - s.EK)
	iS := s.GS * sGate * (voltage - s.ES)
	iL := s.GL * (voltage - s.EL)
	return [4]float64{
		-iNa - iK - iS - iL + iExt,
		(rates[1] - hGate) / rates[4],
		(rates[2] - nGate) / rates[5],
		(rates[3] - sGate) / rates[6],
	}
}

func avRonAddScaled(state [4]float64, slope [4]float64, scale float64) [4]float64 {
	return [4]float64{
		state[0] + scale*slope[0],
		state[1] + scale*slope[1],
		state[2] + scale*slope[2],
		state[3] + scale*slope[3],
	}
}

func (s *AvRonCardiacNeuronState) rk4Candidate(iExt float64) ([4]float64, bool) {
	state := [4]float64{s.V, s.H, s.N, s.S}
	halfDt := 0.5 * s.Dt
	k1 := s.derivatives(state, iExt)
	k2 := s.derivatives(avRonAddScaled(state, k1, halfDt), iExt)
	k3 := s.derivatives(avRonAddScaled(state, k2, halfDt), iExt)
	k4 := s.derivatives(avRonAddScaled(state, k3, s.Dt), iExt)
	candidate := [4]float64{
		state[0] + s.Dt*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0,
		state[1] + s.Dt*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0,
		state[2] + s.Dt*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0,
		state[3] + s.Dt*(k1[3]+2.0*k2[3]+2.0*k3[3]+k4[3])/6.0,
	}
	ok := finiteAvRonValues(candidate[:]...) && avRonGateInRange(candidate[1]) && avRonGateInRange(candidate[2]) && avRonGateInRange(candidate[3])
	return candidate, ok
}

// Step advances the neuron by one timestep.
func (s *AvRonCardiacNeuronState) Step(iExt float64) int {
	if math.IsNaN(iExt) || math.IsInf(iExt, 0) || !s.validRuntime() {
		return 0
	}
	vPrev := s.V
	candidate, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V, s.H, s.N, s.S = candidate[0], candidate[1], candidate[2], candidate[3]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// Reset restores the dynamic state.
func (s *AvRonCardiacNeuronState) Reset() {
	s.V = -60.0
	s.H = 0.6
	s.N = 0.3
	s.S = 0.5
}

// SimulateAvRonCardiacNeuron runs the neuron for n steps.
func SimulateAvRonCardiacNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAvRonCardiacNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for step := 0; step < nSteps; step++ {
		result := s.Step(iExt)
		trace[step] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
