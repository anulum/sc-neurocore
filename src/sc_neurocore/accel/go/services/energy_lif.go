// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Fardet-Levina eLIF

package services

import "math"

// EnergyLIFNeuronState holds the complete author-Brian eLIF state.
type EnergyLIFNeuronState struct {
	V, Epsilon                       float64
	Capacitance, GLeak               float64
	E0, EU, ED, EF                   float64
	VThreshold, VReset               float64
	Alpha, Epsilon0, EpsilonC, Delta float64
	TauE, Dt                         float64
}

// NewEnergyLIFNeuron constructs the pinned Fardet-Levina Brian profile.
func NewEnergyLIFNeuron() *EnergyLIFNeuronState {
	return &EnergyLIFNeuronState{-61, .32, 100, 9, -62.5, -58.5, -40, -62, -59, -62, 1, .5, .18, .01, 200, .1}
}

func (s *EnergyLIFNeuronState) rhs(v, epsilon, current float64) (float64, float64) {
	leak := s.E0 + (s.EU-s.E0)*(1-epsilon/s.Epsilon0)
	dv := (s.GLeak*(leak-v) + current) / s.Capacitance
	production := math.Pow(1-epsilon/(s.Alpha*s.Epsilon0), 3)
	cost := (v - s.EF) / (s.ED - s.EF)
	return dv, (production - cost) / s.TauE
}

func (s *EnergyLIFNeuronState) rk4Candidate(current float64) (float64, float64) {
	dt := s.Dt
	k1v, k1e := s.rhs(s.V, s.Epsilon, current)
	k2v, k2e := s.rhs(s.V+dt*k1v/2, s.Epsilon+dt*k1e/2, current)
	k3v, k3e := s.rhs(s.V+dt*k2v/2, s.Epsilon+dt*k2e/2, current)
	k4v, k4e := s.rhs(s.V+dt*k3v, s.Epsilon+dt*k3e, current)
	return s.V + dt*(k1v+2*k2v+2*k3v+k4v)/6,
		s.Epsilon + dt*(k1e+2*k2e+2*k3e+k4e)/6
}

// Step advances one coupled RK4 sample, returning -1 on invalid input/state.
func (s *EnergyLIFNeuronState) Step(current float64) int {
	if !s.Valid() || !isFiniteEnergyLIF(current) {
		return -1
	}
	v, epsilon := s.rk4Candidate(current)
	if !isFiniteEnergyLIF(v) || v < -200 || v > 100 || !isFiniteEnergyLIF(epsilon) || epsilon < 0 || epsilon > 5 {
		return -1
	}
	if v > s.VThreshold && epsilon > s.EpsilonC {
		after := epsilon - s.Delta
		if after < 0 || after > 5 {
			return -1
		}
		s.V, s.Epsilon = s.VReset, after
		return 1
	}
	s.V, s.Epsilon = v, epsilon
	return 0
}

// Valid reports whether the complete source state is inside its envelope.
func (s *EnergyLIFNeuronState) Valid() bool {
	finite := []float64{s.V, s.Epsilon, s.Capacitance, s.GLeak, s.E0, s.EU, s.ED, s.EF, s.VThreshold, s.VReset, s.Alpha, s.Epsilon0, s.EpsilonC, s.Delta, s.TauE, s.Dt}
	for _, value := range finite {
		if !isFiniteEnergyLIF(value) {
			return false
		}
	}
	return s.V >= -200 && s.V <= 100 && s.VReset >= -200 && s.VReset <= 100 &&
		s.Epsilon >= 0 && s.Epsilon <= 5 && s.Capacitance > 0 && s.GLeak > 0 &&
		s.Alpha > 0 && s.Epsilon0 > 0 && s.EpsilonC >= 0 && s.Delta >= 0 &&
		s.TauE > 0 && s.Dt > 0 && s.Dt <= 1 && s.Dt <= s.TauE && s.ED != s.EF && s.VThreshold > s.VReset
}

func isFiniteEnergyLIF(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// SimulateEnergyLIFNeuron runs the source neuron for n samples.
func SimulateEnergyLIFNeuron(n int, current float64) ([]float64, int) {
	state := NewEnergyLIFNeuron()
	trace := make([]float64, n)
	events := 0
	for i := range trace {
		events += state.Step(current)
		trace[i] = state.V
	}
	return trace, events
}
