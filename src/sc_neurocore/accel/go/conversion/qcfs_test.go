// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go QCFS quantisation and surrogate derivative tests

package conversion

import (
	"math"
	"testing"
)

// TestQCFSQuantisation resolves rounding boundaries and both saturation tails.
func TestQCFSQuantisation(t *testing.T) {
	s := NewQCFSActivation()
	for _, sample := range []struct{ x, expected float64 }{
		{math.Inf(-1), 0}, {-1, 0}, {0, 0},
		{0.06249, 0}, {0.06251, 0.125}, {0.5, 0.5},
		{0.93749, 0.875}, {0.93751, 1}, {math.Inf(1), 1},
	} {
		actual, err := s.Forward(sample.x)
		if err != nil || actual != sample.expected {
			t.Fatalf("x=%g: got %g, error %v; want %g", sample.x, actual, err, sample.expected)
		}
	}
	actual, err := s.Forward(math.NaN())
	if err != nil || !math.IsNaN(actual) {
		t.Fatal("NaN input must propagate")
	}
}

// TestQCFSInvalidMutableParameters refuses grids or learned thresholds with no rate meaning.
func TestQCFSInvalidMutableParameters(t *testing.T) {
	s := NewQCFSActivation()
	s.Steps = 0
	if _, err := s.Forward(0.5); err != ErrQCFSParameters {
		t.Fatal("zero-step grid accepted")
	}
	for _, theta := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		s = NewQCFSActivation()
		s.Theta = theta
		if _, err := s.Forward(0.5); err != ErrQCFSParameters {
			t.Fatalf("invalid theta %g accepted", theta)
		}
	}
}

// TestQCFSSurrogateDerivatives checks interior, saturated, infinite and NaN elements.
func TestQCFSSurrogateDerivatives(t *testing.T) {
	s := QCFSActivation{Steps: 4, Theta: 2}
	for _, sample := range []struct{ x, upstream, input, threshold float64 }{
		{-0.125, 1, 1, 0.0625}, {0.625, 1, 1, -0.0625},
		{math.Inf(1), 1, 0, 1}, {math.Inf(-1), 1, 0, 0},
	} {
		input, threshold, err := s.Backward(sample.x, sample.upstream)
		if err != nil || input != sample.input || threshold != sample.threshold {
			t.Fatalf("x=%g: got %g %g %v", sample.x, input, threshold, err)
		}
	}
	input, threshold, err := s.Backward(-1, -1)
	if err != nil || input != 0 || math.Signbit(threshold) || threshold != 0 {
		t.Fatalf("lower saturation must give +0 threshold derivative, got %g %g", input, threshold)
	}
	input, threshold, err = s.Backward(math.NaN(), 1)
	if err != nil || input != 0 || !math.IsNaN(threshold) {
		t.Fatal("NaN input must carry a NaN threshold derivative")
	}
	s.Theta = math.NaN()
	if _, _, err := s.Backward(0.5, 1); err != ErrQCFSParameters {
		t.Fatal("invalid threshold accepted")
	}
}

// TestQCFSNaNKeepsItsBits propagates the input NaN payload and sign unchanged.
func TestQCFSNaNKeepsItsBits(t *testing.T) {
	s := NewQCFSActivation()
	negative := math.Float64frombits(0xfff8000000000000)
	result, err := s.Forward(negative)
	if err != nil || math.Float64bits(result) != 0xfff8000000000000 {
		t.Fatalf("NaN bits changed to %#x", math.Float64bits(result))
	}
	_, threshold, err := s.Backward(negative, 1)
	if err != nil || math.Float64bits(threshold) != 0xfff8000000000000 {
		t.Fatalf("NaN threshold bits changed to %#x", math.Float64bits(threshold))
	}
}
