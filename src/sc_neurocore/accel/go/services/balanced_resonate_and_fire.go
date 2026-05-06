// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for balanced_resonate_and_fire

package services

import (
	"errors"
	"math"
)

type BalancedResonateAndFireNeuronState struct {
	X         float64
	Y         float64
	Q         float64
	Omega     float64
	BOffset   float64
	Threshold float64
	Gamma     float64
	Dt        float64
}

func NewBalancedResonateAndFireNeuron() *BalancedResonateAndFireNeuronState {
	return &BalancedResonateAndFireNeuronState{
		X:         0.0,
		Y:         0.0,
		Q:         0.0,
		Omega:     10.0,
		BOffset:   1.0,
		Threshold: 1.0,
		Gamma:     0.9,
		Dt:        0.01,
	}
}

func BalancedRFSustainOscillationBoundary(omega float64, dt float64) (float64, error) {
	if dt <= 0.0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return 0.0, errors.New("dt must be finite and positive")
	}
	if omega <= 0.0 || math.IsNaN(omega) || math.IsInf(omega, 0) {
		return 0.0, errors.New("omega must be finite and positive")
	}
	scaled := dt * omega
	if scaled > 1.0 {
		return 0.0, errors.New("dt * omega must be <= 1")
	}
	return (-1.0 + math.Sqrt(math.Max(0.0, 1.0-scaled*scaled))) / dt, nil
}

func (s *BalancedResonateAndFireNeuronState) Validate() error {
	if _, err := BalancedRFSustainOscillationBoundary(s.Omega, s.Dt); err != nil {
		return err
	}
	if s.BOffset <= 0.0 || math.IsNaN(s.BOffset) || math.IsInf(s.BOffset, 0) {
		return errors.New("b_offset must be finite and positive")
	}
	if s.Threshold <= 0.0 || math.IsNaN(s.Threshold) || math.IsInf(s.Threshold, 0) {
		return errors.New("threshold must be finite and positive")
	}
	if s.Gamma < 0.0 || s.Gamma >= 1.0 || math.IsNaN(s.Gamma) || math.IsInf(s.Gamma, 0) {
		return errors.New("gamma must satisfy 0 <= gamma < 1")
	}
	if math.IsNaN(s.X) || math.IsNaN(s.Y) || math.IsNaN(s.Q) ||
		math.IsInf(s.X, 0) || math.IsInf(s.Y, 0) || math.IsInf(s.Q, 0) {
		return errors.New("state variables must be finite")
	}
	return nil
}

func (s *BalancedResonateAndFireNeuronState) Damping() (float64, error) {
	if err := s.Validate(); err != nil {
		return 0.0, err
	}
	pOmega, _ := BalancedRFSustainOscillationBoundary(s.Omega, s.Dt)
	return pOmega - s.BOffset - s.Q, nil
}

func (s *BalancedResonateAndFireNeuronState) DynamicThreshold() float64 {
	return s.Threshold + s.Q
}

func (s *BalancedResonateAndFireNeuronState) Step(current float64) (int, error) {
	if err := s.Validate(); err != nil {
		return 0, err
	}
	bT, _ := s.Damping()
	thetaT := s.DynamicThreshold()
	xPrev := s.X
	yPrev := s.Y

	s.X = xPrev + s.Dt*(bT*xPrev-s.Omega*yPrev+current)
	s.Y = yPrev + s.Dt*(s.Omega*xPrev+bT*yPrev)

	spike := 0
	if s.X >= thetaT {
		spike = 1
	}
	s.Q = s.Gamma*s.Q + float64(spike)
	return spike, nil
}

func SimulateBalancedResonateAndFireNeuron(nSteps int, current float64) ([]float64, int, error) {
	s := NewBalancedResonateAndFireNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spike, err := s.Step(current)
		if err != nil {
			return trace, spikes, err
		}
		trace[t] = s.X
		spikes += spike
	}
	return trace, spikes, nil
}
