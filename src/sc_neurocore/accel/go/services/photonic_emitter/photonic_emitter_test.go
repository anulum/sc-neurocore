// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Photonic crosstalk service tests and benchmark

package photonic_emitter

import (
	"math"
	"testing"
)

func closeEnough(actual, expected float64) bool {
	return math.Abs(actual-expected) < 1.0e-15
}

func TestAnalyzePairGoldenParity(t *testing.T) {
	metrics, err := AnalyzePair(200.0, 50.0, 1550.0, 3.48, 1.45)
	if err != nil {
		t.Fatal(err)
	}
	if !closeEnough(metrics.CouplingCoefficientPerUM, 0.015593714868342372) ||
		!closeEnough(metrics.CouplingRatio, 0.49428770428966934) ||
		!closeEnough(metrics.IsolationDB, 3.0602019274692) {
		t.Fatalf("unexpected pair metrics: %#v", metrics)
	}
}

func TestAnalyzeBankAndPairs(t *testing.T) {
	bank, err := AnalyzeBank(5, 200.0, 50.0, 1550.0, 3.48, 1.45)
	if err != nil {
		t.Fatal(err)
	}
	if bank.NumNearPairs != 4 || bank.NumFarPairs != 3 {
		t.Fatalf("unexpected bank counts: %#v", bank)
	}
	pairs, err := AnalyzePairs([]PairSpec{{0, 1, 200.0, 50.0}, {1, 2, 400.0, 50.0}}, 1550.0, 3.48, 1.45)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 2 || pairs[0].CouplingRatio < pairs[1].CouplingRatio {
		t.Fatalf("unexpected pair ordering: %#v", pairs)
	}
}

func TestFailClosedInputsAndIsolationCeiling(t *testing.T) {
	zero, err := AnalyzePair(200.0, 0.0, 1550.0, 3.48, 1.45)
	if err != nil || zero.IsolationDB != 300.0 {
		t.Fatalf("zero-length ceiling mismatch: %#v, %v", zero, err)
	}
	invalid := []struct {
		gap, length, wavelength, core, cladding float64
	}{
		{math.NaN(), 10.0, 1550.0, 3.48, 1.45},
		{200.0, -1.0, 1550.0, 3.48, 1.45},
		{200.0, 10.0, 0.0, 3.48, 1.45},
		{200.0, 10.0, 1550.0, 1.45, 1.45},
	}
	for _, value := range invalid {
		if _, err := AnalyzePair(value.gap, value.length, value.wavelength, value.core, value.cladding); err == nil {
			t.Fatalf("invalid input accepted: %#v", value)
		}
	}
	if _, err := AnalyzeBank(0, 200.0, 10.0, 1550.0, 3.48, 1.45); err == nil {
		t.Fatal("zero-waveguide bank accepted")
	}
	if _, err := AnalyzePairs([]PairSpec{{1, 1, 200.0, 10.0}}, 1550.0, 3.48, 1.45); err == nil {
		t.Fatal("self-pair accepted")
	}
}

func BenchmarkAnalyzePairs(b *testing.B) {
	pairs := make([]PairSpec, 4096)
	for index := range pairs {
		pairs[index] = PairSpec{
			index,
			index + 1,
			180.0 + float64(index%64),
			8.0 + float64(index%17),
		}
	}
	first, err := AnalyzePair(180.0, 8.0, 1550.0, 3.48, 1.45)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf(
		"GO_FIRST %.17g %.17g %.17g",
		first.CouplingCoefficientPerUM,
		first.CouplingRatio,
		first.IsolationDB,
	)
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := AnalyzePairs(pairs, 1550.0, 3.48, 1.45); err != nil {
			b.Fatal(err)
		}
	}
}
