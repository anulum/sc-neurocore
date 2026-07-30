// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go source MAT* service

package services

import (
	"errors"
	"math"
)

const matVMin = -200.0
const matVMax = 200.0
const matThetaMax = 1.0e9

// MATNeuronState contains the complete non-resetting MAT* state and profile.
// Voltage is relative to rest. Units are millivolts, milliseconds, nanoamps,
// and megaohms. Defaults select the paper's regular-spiking example.
type MATNeuronState struct {
	V                   float64
	Theta1              float64
	Theta2              float64
	RefractoryRemaining float64
	Omega               float64
	TauM                float64
	Tau1                float64
	Tau2                float64
	Alpha1              float64
	Alpha2              float64
	Resistance          float64
	RefractoryPeriod    float64
	Dt                  float64
}

// NewMATNeuron returns the Kobayashi-Tsubo-Shinomoto regular-spiking profile.
func NewMATNeuron() *MATNeuronState {
	return &MATNeuronState{
		V: 0.0, Theta1: 0.0, Theta2: 0.0, RefractoryRemaining: 0.0,
		Omega: 19.0, TauM: 5.0, Tau1: 10.0, Tau2: 200.0,
		Alpha1: 37.0, Alpha2: 2.0, Resistance: 50.0,
		RefractoryPeriod: 2.0, Dt: 0.001,
	}
}

func matFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func (s *MATNeuronState) validate() error {
	for _, value := range []float64{s.V, s.Theta1, s.Theta2, s.RefractoryRemaining, s.Omega, s.TauM, s.Tau1, s.Tau2, s.Alpha1, s.Alpha2, s.Resistance, s.RefractoryPeriod, s.Dt} {
		if !matFinite(value) {
			return errors.New("MAT state and configuration must be finite")
		}
	}
	if s.V < matVMin || s.V > matVMax || math.Abs(s.Omega) > matThetaMax {
		return errors.New("MAT voltage or baseline threshold outside safety envelope")
	}
	for _, value := range []float64{s.Theta1, s.Theta2, s.Alpha1, s.Alpha2} {
		if value < 0.0 || value > matThetaMax {
			return errors.New("MAT threshold history outside safety envelope")
		}
	}
	if s.TauM <= 0.0 || s.Tau1 <= 0.0 || s.Tau2 <= 0.0 || s.Resistance <= 0.0 || s.RefractoryPeriod < 0.0 || s.Dt <= 0.0 {
		return errors.New("MAT time, resistance, or refractory parameter invalid")
	}
	if s.RefractoryRemaining < 0.0 || s.RefractoryRemaining > s.RefractoryPeriod {
		return errors.New("MAT refractory state outside configured interval")
	}
	return nil
}

// Step advances one atomic source MAT* step and returns 1 on an event.
// Voltage uses forward Euler and is never reset; threshold memories use exact
// exponential decay. Invalid input/state returns -1 without mutation.
func (s *MATNeuronState) Step(current float64) int {
	if !matFinite(current) || s.validate() != nil {
		return -1
	}
	v := s.V + s.Dt*(-s.V+s.Resistance*current)/s.TauM
	theta1 := s.Theta1 * math.Exp(-s.Dt/s.Tau1)
	theta2 := s.Theta2 * math.Exp(-s.Dt/s.Tau2)
	refractory := math.Max(0.0, s.RefractoryRemaining-s.Dt)
	if !matFinite(v) || !matFinite(theta1) || !matFinite(theta2) || !matFinite(refractory) || v < matVMin || v > matVMax || theta1 < 0.0 || theta1 > matThetaMax || theta2 < 0.0 || theta2 > matThetaMax {
		return -1
	}
	spike := refractory == 0.0 && v >= s.Omega+theta1+theta2
	if spike {
		theta1 += s.Alpha1
		theta2 += s.Alpha2
		refractory = s.RefractoryPeriod
		if theta1 > matThetaMax || theta2 > matThetaMax {
			return -1
		}
	}
	s.V, s.Theta1, s.Theta2, s.RefractoryRemaining = v, theta1, theta2, refractory
	if spike {
		return 1
	}
	return 0
}

// Reset clears dynamic state while preserving the configured profile.
func (s *MATNeuronState) Reset() {
	s.V, s.Theta1, s.Theta2, s.RefractoryRemaining = 0.0, 0.0, 0.0, 0.0
}

// SimulateMATNeuron runs a constant-current source MAT* trace.
func SimulateMATNeuron(nSteps int, current float64) ([]float64, int) {
	state := NewMATNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for index := range trace {
		result := state.Step(current)
		trace[index] = state.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
