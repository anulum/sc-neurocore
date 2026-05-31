// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for cochlear_hair_cell

package services

import (
	"math"
)

// CochlearHairCellState holds the neuron state
type CochlearHairCellState struct {
	GMax             float64
	EMet             float64
	GL               float64
	EL               float64
	Cap              float64
	X0               float64
	Delta            float64
	Dt               float64
	V                float64
	GlutamateRelease float64
}

// NewCochlearHairCell creates a new CochlearHairCell neuron with default parameters
func NewCochlearHairCell() *CochlearHairCellState {
	return &CochlearHairCellState{
		GMax:             10.0,
		EMet:             0.0,
		GL:               1.0,
		EL:               -60.0,
		Cap:              10.0,
		X0:               0.0,
		Delta:            0.1,
		Dt:               0.01,
		V:                -60.0,
		GlutamateRelease: 0.0,
	}
}

func cochlearFinite(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func (s *CochlearHairCellState) validRuntime() bool {
	return cochlearFinite(s.GMax, s.EMet, s.GL, s.EL, s.Cap, s.X0, s.Delta, s.Dt, s.V, s.GlutamateRelease) &&
		s.GMax >= 0 && s.GL > 0 && s.Cap > 0 && s.Delta > 0 && s.Dt > 0 && s.GlutamateRelease >= 0
}

func (s *CochlearHairCellState) pOpen(displacement float64) (float64, bool) {
	if !cochlearFinite(displacement, s.X0, s.Delta) || s.Delta <= 0 {
		return 0, false
	}
	z := (displacement - s.X0) / s.Delta
	if z >= 0 {
		po := 1.0 / (1.0 + math.Exp(-z))
		return po, cochlearFinite(po)
	}
	ez := math.Exp(z)
	po := ez / (1.0 + ez)
	return po, cochlearFinite(po)
}

// Step advances the neuron by one timestep
func (s *CochlearHairCellState) Step(iExt float64) int {
	if !s.validRuntime() || !cochlearFinite(iExt) {
		return -1
	}
	po, ok := s.pOpen(iExt)
	if !ok {
		return -1
	}
	gMET := s.GMax * po
	gTotal := s.GL + gMET
	if !cochlearFinite(gTotal) || gTotal <= 0 {
		return -1
	}
	vInf := (s.GL*s.EL + gMET*s.EMet) / gTotal
	candidateV := vInf + (s.V-vInf)*math.Exp(-(gTotal/s.Cap)*s.Dt)
	candidateRelease := math.Max(candidateV+60.0, 0.0) / 40.0
	if !cochlearFinite(candidateV, candidateRelease) {
		return -1
	}
	s.V = candidateV
	s.GlutamateRelease = candidateRelease
	if s.GlutamateRelease > 0.5 {
		return 1
	}
	return 0
}

// SimulateCochlearHairCell runs the neuron for n steps
func SimulateCochlearHairCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewCochlearHairCell()
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

var _ = math.Exp
