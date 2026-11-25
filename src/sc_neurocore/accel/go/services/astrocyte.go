// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for astrocyte

package services

import (
	"errors"
	"math"
)

// AstrocyteModelState holds the neuron state
type AstrocyteModelState struct {
	Ca       float64
	H        float64
	Ip3      float64
	VEr      float64
	KEr      float64
	VSerca   float64
	D1       float64
	D2       float64
	D3       float64
	D5       float64
	A2       float64
	C0       float64
	C1       float64
	Leak     float64
	Ip3Prod  float64
	Ip3Decay float64
	Dt       float64
}

// NewAstrocyteModel creates a new AstrocyteModel neuron with default parameters
func NewAstrocyteModel() *AstrocyteModelState {
	return &AstrocyteModelState{
		Ca:       0.05,
		H:        0.8,
		Ip3:      0.5,
		VEr:      0.9,
		KEr:      0.15,
		VSerca:   0.4,
		D1:       0.13,
		D2:       1.049,
		D3:       0.9434,
		D5:       0.08234,
		A2:       0.2,
		C0:       2.0,
		C1:       0.185,
		Leak:     0.01,
		Ip3Prod:  0.0,
		Ip3Decay: 0.14,
		Dt:       0.01,
	}
}

// Step advances the neuron by one timestep
// astrocyteFinite reports whether value is neither NaN nor infinite.
func astrocyteFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validRuntime reports whether every parameter and the state are finite and in range; calcium may
// equal C0, which a step can reach.
func (s *AstrocyteModelState) validRuntime() bool {
	for _, value := range []float64{s.VEr, s.KEr, s.VSerca, s.D1, s.D2, s.D3, s.D5, s.A2, s.C0, s.C1, s.Dt} {
		if !astrocyteFinite(value) || value <= 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.Leak, s.Ip3Prod, s.Ip3Decay} {
		if !astrocyteFinite(value) || value < 0.0 {
			return false
		}
	}
	return astrocyteFinite(s.Ca) && s.Ca >= 0.0 && s.Ca <= s.C0 &&
		astrocyteFinite(s.H) && s.H >= 0.0 && s.H <= 1.0 &&
		astrocyteFinite(s.Ip3) && s.Ip3 >= 0.0
}

// Step advances one forward-Euler step of the Li-Rinzel recurrence with iExt, the input IP3
// production (uM/s), and returns the new cytosolic calcium. It mirrors the Python reference
// term for term: a negative or non-finite input, an invalid parameter or state, or a calcium
// candidate that is not finite or exceeds C0 returns an error and leaves the state unchanged.
// float64(...) keeps each product from fusing into an FMA, so the lane rounds as the others do.
func (s *AstrocyteModelState) Step(iExt float64) (float64, error) {
	if !astrocyteFinite(iExt) || iExt < 0.0 {
		return s.Ca, errors.New("current must be finite and non-negative")
	}
	if !s.validRuntime() {
		return s.Ca, errors.New("astrocyte parameters and state must be finite and in range")
	}
	mInf := s.Ip3 / (s.Ip3 + s.D1)
	nInf := s.Ca / (s.Ca + s.D5)
	caEr := (s.C0 - s.Ca) / s.C1
	jChannel := float64(s.VEr*math.Pow(mInf*nInf*s.H, 3.0)) * (caEr - s.Ca)
	jSerca := float64(s.VSerca*math.Pow(s.Ca, 2.0)) / (math.Pow(s.Ca, 2.0) + math.Pow(s.KEr, 2.0))
	jLeak := float64(s.Leak * (caEr - s.Ca))
	dCa := jChannel - jSerca + jLeak
	q2 := float64(s.D2*(s.Ip3+s.D1)) / (s.Ip3 + s.D3)
	hInf := q2 / (q2 + s.Ca)
	tauH := 1.0 / float64(s.A2*(q2+s.Ca))
	dH := (hInf - s.H) / math.Max(tauH, 1e-6)
	dIp3 := iExt + s.Ip3Prod - float64(s.Ip3Decay*s.Ip3)
	caNext := s.Ca + float64(dCa*s.Dt)
	if !astrocyteFinite(caNext) || caNext > s.C0 {
		return s.Ca, errors.New("calcium update must remain finite and within the total calcium pool")
	}
	s.Ca = math.Max(0.0, caNext)
	s.H = math.Min(math.Max(s.H+float64(dH*s.Dt), 0.0), 1.0)
	s.Ip3 = math.Max(0.0, s.Ip3+float64(dIp3*s.Dt))
	return s.Ca, nil
}

// SimulateAstrocyteModel runs a default cell for nSteps steps at a constant input and returns
// the calcium trace and the number of refused steps.
func SimulateAstrocyteModel(nSteps int, iExt float64) ([]float64, int) {
	s := NewAstrocyteModel()
	trace := make([]float64, nSteps)
	refused := 0
	for t := 0; t < nSteps; t++ {
		if _, err := s.Step(iExt); err != nil {
			refused++
		}
		trace[t] = s.Ca
	}
	return trace, refused
}
