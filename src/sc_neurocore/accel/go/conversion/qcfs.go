// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go QCFS quantisation and surrogate derivatives

package conversion

import (
	"errors"
	"math"
)

// ErrQCFSParameters reports an invalid quantisation grid or threshold.
var ErrQCFSParameters = errors.New("QCFS requires positive steps and a finite positive threshold")

// QCFSActivation stores the quantisation grid and current learned threshold.
// Explicit float64 conversions round every intermediate, so no multiply-add
// is fused and results match the Python operation order bit for bit.
type QCFSActivation struct {
	Steps uint32
	Theta float64
}

// NewQCFSActivation constructs the same eight-step default as Python.
func NewQCFSActivation() QCFSActivation {
	return QCFSActivation{Steps: 8, Theta: 1.0}
}

// Valid reports a positive grid and a finite positive threshold.
func (s QCFSActivation) Valid() bool {
	return s.Steps > 0 && !math.IsNaN(s.Theta) && !math.IsInf(s.Theta, 0) && s.Theta > 0
}

// clip bounds the shifted coordinate by comparison so a NaN passes through with
// its own bits, as NumPy and PyTorch propagate it; math.Min and math.Max would
// substitute a canonical NaN.
func clip(value, upper float64) float64 {
	if value < 0 {
		return 0
	}
	if value > upper {
		return upper
	}
	return value
}

// Forward quantises a scalar, saturating infinities and propagating NaNs.
func (s QCFSActivation) Forward(x float64) (float64, error) {
	if !s.Valid() {
		return 0, ErrQCFSParameters
	}
	steps := float64(s.Steps)
	shifted := float64(float64(x*steps)/s.Theta) + 0.5
	clipped := clip(shifted, steps)
	return float64(math.Floor(clipped)*s.Theta) / steps, nil
}

// Backward returns one element's straight-through input derivative, upstream on
// the open interior 0 < s < T and zero elsewhere, and the threshold derivative a
// one-element batch receives (Bu et al., 2022, Eq. 17).
func (s QCFSActivation) Backward(x, upstream float64) (float64, float64, error) {
	if !s.Valid() {
		return 0, 0, ErrQCFSParameters
	}
	steps := float64(s.Steps)
	theta := s.Theta
	shifted := float64(float64(x*steps)/theta) + 0.5
	interior := shifted > 0 && shifted < steps
	lattice := math.Floor(clip(shifted, steps))
	outputGradient := upstream / steps
	carried, retained, inputGradient := 0.0, 0.0, 0.0
	if interior {
		carried = float64(outputGradient * theta)
		retained = x
		inputGradient = float64(carried/theta) * steps
	}
	scaledInput := float64(retained * steps)
	latticeTerm := float64(outputGradient * lattice)
	quotient := float64(float64(scaledInput/theta) / theta)
	threshold := float64(0+latticeTerm) + float64(0+float64(-carried*quotient))
	return inputGradient, threshold, nil
}
