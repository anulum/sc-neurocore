// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for unipolar_brush_cell

package services

import (
	"math"
)

// UnipolarBrushCellState holds the neuron state
type UnipolarBrushCellState struct {
	V              float64
	Persistent     float64
	VRest          float64
	VReset         float64
	VThreshold     float64
	TauM           float64
	TauPersistent  float64
	PersistentGain float64
	Gain           float64
	Dt             float64
}

// NewUnipolarBrushCell creates a new UnipolarBrushCell neuron with default parameters
func NewUnipolarBrushCell() *UnipolarBrushCellState {
	return &UnipolarBrushCellState{
		V:              -65.0,
		Persistent:     0.0,
		VRest:          -65.0,
		VReset:         -70.0,
		VThreshold:     -50.0,
		TauM:           8.0,
		TauPersistent:  200.0,
		PersistentGain: 0.5,
		Gain:           2.5,
		Dt:             0.5,
	}
}

func unipolarBrushCellFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *UnipolarBrushCellState) validConfiguration() bool {
	if s == nil {
		return false
	}
	return unipolarBrushCellFinite(
		s.VRest,
		s.VReset,
		s.VThreshold,
		s.TauM,
		s.TauPersistent,
		s.PersistentGain,
		s.Gain,
		s.Dt,
	) &&
		s.TauM > 0.0 &&
		s.TauPersistent > 0.0 &&
		s.PersistentGain >= 0.0 &&
		s.Gain >= 0.0 &&
		s.Dt > 0.0 &&
		s.VReset < s.VThreshold
}

func (s *UnipolarBrushCellState) validState() bool {
	return unipolarBrushCellFinite(s.V, s.Persistent) &&
		s.V >= -100.0 &&
		s.V <= 60.0 &&
		s.Persistent >= 0.0
}

func unipolarBrushCellFirstOrderRelaxation(previous float64, steadyState float64, dt float64, tau float64) float64 {
	return previous + (steadyState-previous)*(-math.Expm1(-dt/tau))
}

// Step advances the neuron by one timestep
func (s *UnipolarBrushCellState) Step(iExt float64) int {
	if !s.validConfiguration() || !s.validState() || math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0
	}
	inputDrive := s.Gain * math.Max(0.0, iExt)
	if math.IsNaN(inputDrive) || math.IsInf(inputDrive, 0) {
		return 0
	}
	nextPersistent := unipolarBrushCellFirstOrderRelaxation(
		s.Persistent,
		s.PersistentGain*inputDrive,
		s.Dt,
		s.TauPersistent,
	)
	nextPersistent = math.Max(0.0, nextPersistent)
	nextV := unipolarBrushCellFirstOrderRelaxation(
		s.V,
		s.VRest+inputDrive+nextPersistent,
		s.Dt,
		s.TauM,
	)
	if math.IsNaN(nextPersistent) || math.IsInf(nextPersistent, 0) || math.IsNaN(nextV) || math.IsInf(nextV, 0) {
		return 0
	}
	s.Persistent = nextPersistent
	if nextV >= s.VThreshold {
		s.V = s.VReset
		return 1
	}
	s.V = math.Max(-100.0, math.Min(60.0, nextV))
	return 0
}

// SimulateUnipolarBrushCell runs the neuron for n steps
func SimulateUnipolarBrushCell(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewUnipolarBrushCell()
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
