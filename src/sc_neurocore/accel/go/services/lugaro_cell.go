// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lugaro_cell

package services

import "math"

// LugaroCellState holds the LIF-adaptation Lugaro-cell state.
type LugaroCellState struct {
	V          float64
	Adapt      float64
	VRest      float64
	VReset     float64
	VThreshold float64
	TauM       float64
	TauAdapt   float64
	AAdapt     float64
	Gain       float64
	Serotonin  float64
	Dt         float64
}

// NewLugaroCell creates a new LugaroCell neuron with default parameters.
func NewLugaroCell() *LugaroCellState {
	return &LugaroCellState{
		V:          -55.0,
		Adapt:      0.0,
		VRest:      -55.0,
		VReset:     -65.0,
		VThreshold: -48.0,
		TauM:       10.0,
		TauAdapt:   150.0,
		AAdapt:     0.05,
		Gain:       2.0,
		Serotonin:  0.0,
		Dt:         0.5,
	}
}

func finiteLugaro(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *LugaroCellState) valid() bool {
	return finiteLugaro(
		s.V, s.Adapt, s.VRest, s.VReset, s.VThreshold, s.TauM, s.TauAdapt,
		s.AAdapt, s.Gain, s.Serotonin, s.Dt,
	) &&
		s.TauM > 0.0 && s.TauAdapt > 0.0 && s.Dt > 0.0 &&
		s.AAdapt >= 0.0 && s.Gain >= 0.0 && s.Serotonin >= 0.0 && s.Serotonin <= 1.0 &&
		s.Adapt >= 0.0 && s.VThreshold > s.VReset && s.VThreshold > s.VRest
}

// Step advances the neuron by one timestep. Invalid input or state leaves the
// old state untouched and returns no spike.
func (s *LugaroCellState) Step(iExt float64) int {
	if !s.valid() || !finiteLugaro(iExt) {
		return 0
	}

	effectiveGain := s.Gain * (1.0 + 0.5*s.Serotonin)
	input := effectiveGain * iExt
	vInf := s.VRest + input - s.Adapt
	vNext := vInf + (s.V-vInf)*math.Exp(-s.Dt/s.TauM)
	adaptInf := math.Max(0.0, s.AAdapt*math.Max(0.0, vNext-s.VRest))
	adaptNext := adaptInf + (s.Adapt-adaptInf)*math.Exp(-s.Dt/s.TauAdapt)
	adaptNext = math.Max(0.0, adaptNext)
	if !finiteLugaro(vNext, adaptNext) {
		return 0
	}

	if vNext >= s.VThreshold {
		s.V = s.VReset
		s.Adapt = adaptNext + 1.0
		return 1
	}

	s.V = math.Max(-100.0, math.Min(60.0, vNext))
	s.Adapt = adaptNext
	return 0
}

// SimulateLugaroCell runs the neuron for n steps.
func SimulateLugaroCell(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewLugaroCell()
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
