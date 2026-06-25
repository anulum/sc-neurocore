// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for upper_motor_neuron

package services

import (
	"math"
)

// UpperMotorNeuronState holds the neuron state
type UpperMotorNeuronState struct {
	V          float64
	M          float64
	H          float64
	N          float64
	P          float64
	S          float64
	GNa        float64
	GK         float64
	GM         float64
	GCa        float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	CM         float64
	Dt         float64
	VThreshold float64
}

// NewUpperMotorNeuron creates a new UpperMotorNeuron neuron with default parameters
func NewUpperMotorNeuron() *UpperMotorNeuronState {
	return &UpperMotorNeuronState{
		V:          -70.0,
		M:          0.05,
		H:          0.6,
		N:          0.3,
		P:          0.0,
		S:          0.0,
		GNa:        50.0,
		GK:         5.0,
		GM:         0.07,
		GCa:        0.3,
		GL:         0.1,
		ENa:        50.0,
		EK:         -90.0,
		ECa:        120.0,
		EL:         -70.0,
		CM:         1.0,
		Dt:         0.025,
		VThreshold: -20.0,
	}
}

func upperMotorNeuronFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func upperMotorNeuronRateExp(value float64) float64 {
	return math.Exp(math.Max(-60.0, math.Min(60.0, value)))
}

func upperMotorNeuronGate(previous float64, alpha float64, beta float64, dt float64) (float64, bool) {
	total := alpha + beta
	if total <= 0.0 || !upperMotorNeuronFinite(total) {
		return previous, false
	}
	steady := alpha / total
	next := steady + (previous-steady)*upperMotorNeuronRateExp(-total*dt)
	return math.Min(1.0, math.Max(0.0, next)), upperMotorNeuronFinite(next)
}

func upperMotorNeuronGateInf(previous float64, steady float64, tau float64, dt float64) (float64, bool) {
	if tau <= 0.0 || !upperMotorNeuronFinite(tau) {
		return previous, false
	}
	next := steady + (previous-steady)*upperMotorNeuronRateExp(-dt/tau)
	return math.Min(1.0, math.Max(0.0, next)), upperMotorNeuronFinite(next)
}

func (s *UpperMotorNeuronState) validConfiguration() bool {
	if s == nil {
		return false
	}
	if !upperMotorNeuronFinite(
		s.GNa, s.GK, s.GM, s.GCa, s.GL,
		s.ENa, s.EK, s.ECa, s.EL,
		s.CM, s.Dt, s.VThreshold,
	) {
		return false
	}
	return s.GNa >= 0.0 &&
		s.GK >= 0.0 &&
		s.GM >= 0.0 &&
		s.GCa >= 0.0 &&
		s.GL >= 0.0 &&
		s.CM > 0.0 &&
		s.Dt > 0.0
}

func (s *UpperMotorNeuronState) validState() bool {
	if !upperMotorNeuronFinite(s.V, s.M, s.H, s.N, s.P, s.S) {
		return false
	}
	if s.V < -150.0 || s.V > 100.0 {
		return false
	}
	return s.M >= 0.0 && s.M <= 1.0 &&
		s.H >= 0.0 && s.H <= 1.0 &&
		s.N >= 0.0 && s.N <= 1.0 &&
		s.P >= 0.0 && s.P <= 1.0 &&
		s.S >= 0.0 && s.S <= 1.0
}

func (s *UpperMotorNeuronState) stepCandidate(v float64, m float64, h float64, n float64, p float64, caGate float64, current float64) (float64, float64, float64, float64, float64, float64, bool) {
	dv := v - (-56.2)
	xM := dv - 13.0
	alphaM := 0.32 * 4.0
	if math.Abs(xM) >= 1e-6 {
		alphaM = -0.32 * xM / (upperMotorNeuronRateExp(-xM/4.0) - 1.0)
	}
	// Published Pospischil beta_m numerator is V - V_T - 40 (an earlier -17 offset,
	// shared with alpha_h, drove the cell into depolarisation block).
	xBm := dv - 40.0
	betaM := 0.28 * 5.0
	if math.Abs(xBm) >= 1e-6 {
		betaM = 0.28 * xBm / (upperMotorNeuronRateExp(xBm/5.0) - 1.0)
	}
	alphaH := 0.128 * upperMotorNeuronRateExp(-(dv-17.0)/18.0)
	betaH := 4.0 / (1.0 + upperMotorNeuronRateExp(-(dv-40.0)/5.0))
	xN := dv - 15.0
	alphaN := 0.032 * 5.0
	if math.Abs(xN) >= 1e-6 {
		alphaN = -0.032 * xN / (upperMotorNeuronRateExp(-xN/5.0) - 1.0)
	}
	betaN := 0.5 * upperMotorNeuronRateExp(-(dv-10.0)/40.0)
	var ok bool
	if m, ok = upperMotorNeuronGate(m, alphaM, betaM, s.Dt); !ok {
		return v, m, h, n, p, caGate, false
	}
	if h, ok = upperMotorNeuronGate(h, alphaH, betaH, s.Dt); !ok {
		return v, m, h, n, p, caGate, false
	}
	if n, ok = upperMotorNeuronGate(n, alphaN, betaN, s.Dt); !ok {
		return v, m, h, n, p, caGate, false
	}
	pInf := 1.0 / (1.0 + upperMotorNeuronRateExp(-(v+35.0)/10.0))
	tauP := 400.0 / (3.3*upperMotorNeuronRateExp((v+35.0)/20.0) + upperMotorNeuronRateExp(-(v+35.0)/20.0))
	if p, ok = upperMotorNeuronGateInf(p, pInf, tauP, s.Dt); !ok {
		return v, m, h, n, p, caGate, false
	}
	sInf := 1.0 / (1.0 + upperMotorNeuronRateExp(-(v+20.0)/5.0))
	if caGate, ok = upperMotorNeuronGateInf(caGate, sInf, 10.0, s.Dt); !ok {
		return v, m, h, n, p, caGate, false
	}
	gNa := s.GNa * m * m * m * h
	gK := s.GK * math.Pow(n, 4)
	gM := s.GM * p
	gCa := s.GCa * caGate * caGate
	gTotal := gNa + gK + gM + gCa + s.GL
	if gTotal <= 0.0 || !upperMotorNeuronFinite(gTotal) {
		return v, m, h, n, p, caGate, false
	}
	steadyV := (current + gNa*s.ENa + gK*s.EK + gM*s.EK + gCa*s.ECa + s.GL*s.EL) / gTotal
	v = steadyV + (v-steadyV)*upperMotorNeuronRateExp(-(gTotal/s.CM)*s.Dt)
	v = math.Max(-150.0, math.Min(100.0, v))
	return v, m, h, n, p, caGate, upperMotorNeuronFinite(v, m, h, n, p, caGate)
}

// Step advances the neuron by one timestep
func (s *UpperMotorNeuronState) Step(iExt float64) int {
	if !s.validConfiguration() || !s.validState() || !upperMotorNeuronFinite(iExt) {
		return 0
	}
	vPrev := s.V
	v, m, h, n, p, caGate := s.V, s.M, s.H, s.N, s.P, s.S
	ok := true
	for step := 0; step < 4; step++ {
		v, m, h, n, p, caGate, ok = s.stepCandidate(v, m, h, n, p, caGate, iExt)
		if !ok {
			return 0
		}
	}
	s.V, s.M, s.H, s.N, s.P, s.S = v, m, h, n, p, caGate
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateUpperMotorNeuron runs the neuron for n steps
func SimulateUpperMotorNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewUpperMotorNeuron()
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
