// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed photonic crosstalk service kernel

// Package photonic_emitter implements the numeric coupled-waveguide hot path.
// Text emission, FDTD, Meep, and filesystem work remain in the Python layer.
package photonic_emitter

import (
	"fmt"
	"math"
)

const (
	isolationCeilingDB  = 300.0
	isolationRatioFloor = 1.0e-15
)

// PairSpec identifies one arbitrary pair of waveguides.
type PairSpec struct {
	IndexA           int
	IndexB           int
	GapNM            float64
	CouplingLengthUM float64
}

// PairMetrics contains the complete coupled-mode result for one pair.
type PairMetrics struct {
	CouplingCoefficientPerUM float64
	CouplingRatio            float64
	IsolationDB              float64
}

// BankMetrics contains aggregate results for a uniform waveguide bank.
type BankMetrics struct {
	NumWaveguides     int
	NumNearPairs      int
	NumFarPairs       int
	Adjacent          PairMetrics
	NextNearest       PairMetrics
	WorstIsolationDB  float64
	MeanCouplingRatio float64
	MaxCouplingRatio  float64
	CrosstalkSafe     bool
}

func requireNonNegative(value float64, name string) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0.0 {
		return fmt.Errorf("%s must be finite and non-negative", name)
	}
	return nil
}

func validateMaterial(wavelengthNM, coreIndex, claddingIndex float64) error {
	if math.IsNaN(wavelengthNM) || math.IsInf(wavelengthNM, 0) || wavelengthNM <= 0.0 {
		return fmt.Errorf("wavelength_nm must be finite and positive")
	}
	if math.IsNaN(coreIndex) || math.IsInf(coreIndex, 0) || coreIndex <= 0.0 {
		return fmt.Errorf("core_index must be finite and positive")
	}
	if math.IsNaN(claddingIndex) || math.IsInf(claddingIndex, 0) || claddingIndex <= 0.0 {
		return fmt.Errorf("cladding_index must be finite and positive")
	}
	if coreIndex <= claddingIndex {
		return fmt.Errorf("core_index must be greater than cladding_index")
	}
	return nil
}

// AnalyzePair evaluates the Marcatili-form coupled-mode contract.
func AnalyzePair(gapNM, couplingLengthUM, wavelengthNM, coreIndex, claddingIndex float64) (PairMetrics, error) {
	if err := requireNonNegative(gapNM, "gap_nm"); err != nil {
		return PairMetrics{}, err
	}
	if err := requireNonNegative(couplingLengthUM, "coupling_length_um"); err != nil {
		return PairMetrics{}, err
	}
	if err := validateMaterial(wavelengthNM, coreIndex, claddingIndex); err != nil {
		return PairMetrics{}, err
	}

	indexContrast := math.Sqrt(coreIndex*coreIndex - claddingIndex*claddingIndex)
	decayLengthNM := wavelengthNM / (2.0 * math.Pi * indexContrast)
	effectiveIndexDifference := 0.1 * math.Exp(-gapNM/decayLengthNM)
	couplingCoefficient := math.Pi * effectiveIndexDifference / (wavelengthNM * 1.0e-3)
	phase := couplingCoefficient * couplingLengthUM
	ratio := math.Pow(math.Sin(phase), 2)
	isolation := isolationCeilingDB
	if ratio >= isolationRatioFloor {
		isolation = -10.0 * math.Log10(ratio)
	}
	return PairMetrics{couplingCoefficient, ratio, isolation}, nil
}

// AnalyzeBank evaluates adjacent and next-nearest coupling in a uniform bank.
func AnalyzeBank(numWaveguides int, gapNM, couplingLengthUM, wavelengthNM, coreIndex, claddingIndex float64) (BankMetrics, error) {
	if numWaveguides < 1 {
		return BankMetrics{}, fmt.Errorf("num_waveguides must be at least one")
	}
	adjacent, err := AnalyzePair(gapNM, couplingLengthUM, wavelengthNM, coreIndex, claddingIndex)
	if err != nil {
		return BankMetrics{}, err
	}
	nextNearest, err := AnalyzePair(2.0*gapNM, couplingLengthUM, wavelengthNM, coreIndex, claddingIndex)
	if err != nil {
		return BankMetrics{}, err
	}
	numNear := numWaveguides - 1
	numFar := numWaveguides - 2
	if numFar < 0 {
		numFar = 0
	}
	pairCount := numNear + numFar
	worst := math.Inf(1)
	meanRatio := 0.0
	maxRatio := 0.0
	if pairCount > 0 {
		worst = math.Min(adjacent.IsolationDB, nextNearest.IsolationDB)
		meanRatio = (float64(numNear)*adjacent.CouplingRatio + float64(numFar)*nextNearest.CouplingRatio) / float64(pairCount)
		maxRatio = math.Max(adjacent.CouplingRatio, nextNearest.CouplingRatio)
	}
	return BankMetrics{
		NumWaveguides:     numWaveguides,
		NumNearPairs:      numNear,
		NumFarPairs:       numFar,
		Adjacent:          adjacent,
		NextNearest:       nextNearest,
		WorstIsolationDB:  worst,
		MeanCouplingRatio: meanRatio,
		MaxCouplingRatio:  maxRatio,
		CrosstalkSafe:     worst > 20.0,
	}, nil
}

// AnalyzePairs evaluates an arbitrary pair batch atomically.
func AnalyzePairs(pairs []PairSpec, wavelengthNM, coreIndex, claddingIndex float64) ([]PairMetrics, error) {
	if err := validateMaterial(wavelengthNM, coreIndex, claddingIndex); err != nil {
		return nil, err
	}
	for _, pair := range pairs {
		if pair.IndexA < 0 || pair.IndexB < 0 || pair.IndexA == pair.IndexB {
			return nil, fmt.Errorf("each pair must name two distinct non-negative waveguides")
		}
		if err := requireNonNegative(pair.GapNM, "gap_nm"); err != nil {
			return nil, err
		}
		if err := requireNonNegative(pair.CouplingLengthUM, "coupling_length_um"); err != nil {
			return nil, err
		}
	}
	results := make([]PairMetrics, len(pairs))
	for index, pair := range pairs {
		metrics, err := AnalyzePair(pair.GapNM, pair.CouplingLengthUM, wavelengthNM, coreIndex, claddingIndex)
		if err != nil {
			return nil, err
		}
		results[index] = metrics
	}
	return results, nil
}
