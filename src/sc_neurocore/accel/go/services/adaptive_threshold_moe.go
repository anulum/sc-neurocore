// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for adaptive_threshold_moe

package services

import (
	"errors"
	"math"
)

var (
	ErrAdaptiveThresholdMoEInvalidInput    = errors.New("adaptive threshold moe input must be finite")
	ErrAdaptiveThresholdMoEInvalidState    = errors.New("adaptive threshold moe state must be finite with positive k, threshold, and EMA alpha in (0,1]")
	ErrAdaptiveThresholdMoENonFiniteOutput = errors.New("adaptive threshold moe adaptive threshold or residual must remain finite")
)

// AdaptiveThresholdMoENeuronState holds the neuron state
type AdaptiveThresholdMoENeuronState struct {
	K        float64
	EmaAlpha float64
	V        float64
	VTh      float64
	MeanAbsX float64
}

// NewAdaptiveThresholdMoENeuron creates a new AdaptiveThresholdMoENeuron neuron with default parameters
func NewAdaptiveThresholdMoENeuron() *AdaptiveThresholdMoENeuronState {
	return &AdaptiveThresholdMoENeuronState{
		K:        4.0,
		EmaAlpha: 0.1,
		V:        0.0,
		VTh:      1.0,
		MeanAbsX: 0.0,
	}
}

func adaptiveThresholdMoEFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *AdaptiveThresholdMoENeuronState) Valid() bool {
	return adaptiveThresholdMoEFinite(s.K, s.EmaAlpha, s.V, s.VTh, s.MeanAbsX) &&
		s.K > 0.0 &&
		s.EmaAlpha > 0.0 &&
		s.EmaAlpha <= 1.0 &&
		s.VTh > 0.0 &&
		s.MeanAbsX >= 0.0
}

func adaptiveThresholdMoEThreshold(meanAbsX float64, k float64) (float64, error) {
	if !adaptiveThresholdMoEFinite(meanAbsX, k) || meanAbsX < 0.0 || k <= 0.0 {
		return 0.0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	vTh := 1.0
	if meanAbsX > 1e-12 {
		vTh = meanAbsX / k
	}
	if !adaptiveThresholdMoEFinite(vTh) || vTh <= 0.0 {
		return 0.0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	return vTh, nil
}

// Step advances the neuron by one timestep
func (s *AdaptiveThresholdMoENeuronState) Step(iExt float64) (int, error) {
	if !adaptiveThresholdMoEFinite(iExt) {
		return 0, ErrAdaptiveThresholdMoEInvalidInput
	}
	if !s.Valid() {
		return 0, ErrAdaptiveThresholdMoEInvalidState
	}

	nextMeanAbsX := (1.0-s.EmaAlpha)*s.MeanAbsX + s.EmaAlpha*math.Abs(iExt)
	nextVTh, err := adaptiveThresholdMoEThreshold(nextMeanAbsX, s.K)
	if err != nil {
		return 0, err
	}
	nextV := s.V + iExt
	if !adaptiveThresholdMoEFinite(nextV) {
		return 0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	ratio := nextV / nextVTh
	if !adaptiveThresholdMoEFinite(ratio) {
		return 0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	spikes := int(math.RoundToEven(ratio))
	if spikes < 0 {
		spikes = 0
	}
	residual := nextV
	if spikes != 0 {
		residual = nextV - nextVTh*float64(spikes)
	}
	if !adaptiveThresholdMoEFinite(residual) {
		return 0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	s.MeanAbsX = nextMeanAbsX
	s.VTh = nextVTh
	s.V = residual
	return spikes, nil
}

func (s *AdaptiveThresholdMoENeuronState) StepCollapsed(activation float64) (int, error) {
	if !adaptiveThresholdMoEFinite(activation) {
		return 0, ErrAdaptiveThresholdMoEInvalidInput
	}
	if !s.Valid() {
		return 0, ErrAdaptiveThresholdMoEInvalidState
	}
	nextMeanAbsX := (1.0-s.EmaAlpha)*s.MeanAbsX + s.EmaAlpha*math.Abs(activation)
	nextVTh, err := adaptiveThresholdMoEThreshold(nextMeanAbsX, s.K)
	if err != nil {
		return 0, err
	}
	ratio := activation / nextVTh
	if !adaptiveThresholdMoEFinite(ratio) {
		return 0, ErrAdaptiveThresholdMoENonFiniteOutput
	}
	spikes := int(math.RoundToEven(ratio))
	if spikes < 0 {
		spikes = 0
	}
	s.MeanAbsX = nextMeanAbsX
	s.VTh = nextVTh
	return spikes, nil
}

// SimulateAdaptiveThresholdMoENeuron runs the neuron for n steps
func SimulateAdaptiveThresholdMoENeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAdaptiveThresholdMoENeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
