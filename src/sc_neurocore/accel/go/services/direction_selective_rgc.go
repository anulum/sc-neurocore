// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for direction_selective_rgc

package services

import (
	"errors"
	"math"
)

// DirectionSelectiveRGCState holds the direction-selective RGC state.
type DirectionSelectiveRGCState struct {
	Tau           float64
	Theta         float64
	IsOnCentre    float64
	WCentre       float64
	WSurround     float64
	DirectionPref float64
	Dt            float64
	V             float64
	PrevIntensity float64
	Surround      float64
}

// NewDirectionSelectiveRGC creates a new On-centre DirectionSelectiveRGC.
func NewDirectionSelectiveRGC() *DirectionSelectiveRGCState {
	return &DirectionSelectiveRGCState{
		Tau: 10.0, Theta: 0.5, IsOnCentre: 1.0, WCentre: 1.0, WSurround: 0.3,
		DirectionPref: 0.0, Dt: 1.0, V: 0.0, PrevIntensity: 0.0, Surround: 0.0,
	}
}

func rgcFinite(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func (s *DirectionSelectiveRGCState) validRuntime() bool {
	return rgcFinite(s.Tau, s.Theta, s.IsOnCentre, s.WCentre, s.WSurround, s.DirectionPref, s.Dt, s.V, s.PrevIntensity, s.Surround) &&
		s.Tau > 0.0 && s.Theta > 0.0 && s.Dt > 0.0 && s.WCentre >= 0.0 && s.WSurround >= 0.0 && s.PrevIntensity >= 0.0 && s.Surround >= 0.0 &&
		(s.IsOnCentre == 0.0 || s.IsOnCentre == 1.0)
}

// StepRF advances the cell with centre intensity and surround mean input.
func (s *DirectionSelectiveRGCState) StepRF(intensity, surroundMean float64) (int, error) {
	if !rgcFinite(intensity, surroundMean) || intensity < 0.0 || surroundMean < 0.0 || !s.validRuntime() {
		return 0, errors.New("invalid DirectionSelectiveRGC state or optical drive")
	}
	temporalDiff := intensity - s.PrevIntensity
	centreResponse := s.WCentre * temporalDiff
	if s.IsOnCentre == 0.0 {
		centreResponse = -centreResponse
	}
	nextSurround := 0.9*s.Surround + 0.1*surroundMean
	drive := centreResponse - s.WSurround*nextSurround
	decay := math.Exp(-s.Dt / s.Tau)
	nextV := drive + (s.V-drive)*decay
	if !rgcFinite(nextSurround, drive, decay, nextV) || nextSurround < 0.0 {
		return 0, errors.New("invalid DirectionSelectiveRGC candidate")
	}
	s.PrevIntensity = intensity
	s.Surround = nextSurround
	if nextV >= s.Theta {
		s.V = 0.0
		return 1, nil
	}
	s.V = nextV
	return 0, nil
}

// Step advances the neuron by one timestep with no surround input.
func (s *DirectionSelectiveRGCState) Step(iExt float64) (int, error) {
	return s.StepRF(iExt, 0.0)
}

// SimulateDirectionSelectiveRGC runs the neuron for n steps.
func SimulateDirectionSelectiveRGC(nSteps int, iExt float64) ([]float64, int) {
	s := NewDirectionSelectiveRGC()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = s.V
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
