// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go public dense IF ownership and admission
// SC-NeuroCore — Go public replay budget and domain admission

package conversion_test

import (
	c "github.com/anulum/sc-neurocore/accel/conversion"
	"math"
	"testing"
)

func TestPublicReplayResourceAndDomainRefusals(t *testing.T) {
	layer, _ := c.NewDenseLayer(1, 1, []float64{1}, nil, 1, 0, 1024)
	model, _ := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Spikes, 1024)
	for _, pair := range []struct {
		trace      bool
		pass, fail int
	}{{false, 112, 111}, {true, 128, 127}} {
		options := c.ReplayOptions{Trace: pair.trace, BinaryInputs: true, MaxWorkingBytes: pair.pass}
		if _, err := model.Replay([]float64{1}, 1, 1, options); err != nil {
			t.Fatal(err)
		}
		options.MaxWorkingBytes = pair.fail
		if _, err := model.Replay([]float64{1}, 1, 1, options); err != c.ErrResourceLimit {
			t.Fatal(err)
		}
	}
	options := c.ReplayOptions{BinaryInputs: true, MaxWorkingBytes: 1024}
	for _, frames := range [][]float64{{math.NaN()}, {math.Inf(1)}, {-1}, {1.5}, {0.5}} {
		if r, err := model.Replay(frames, 1, 1, options); err != c.ErrInvalidInput || r != nil {
			t.Fatal(r, err)
		}
	}
	if _, err := model.Replay([]float64{1}, -1, 1, options); err != c.ErrInvalidInput {
		t.Fatal(err)
	}
	if _, err := model.Replay([]float64{1}, 2, 1, options); err != c.ErrInvalidInput {
		t.Fatal(err)
	}
	for _, states := range [][][]float64{{}, {{0, 0}}, {{math.NaN()}}} {
		options.InitialState = states
		if _, err := model.Replay([]float64{1}, 1, 1, options); err != c.ErrInvalidInput {
			t.Fatal(err)
		}
	}
	options.InitialState = nil
	options.MaxWorkingBytes = 0
	if _, err := model.Replay([]float64{1}, 1, 1, options); err != c.ErrInvalidInput {
		t.Fatal(err)
	}
	if _, err := model.Replay(nil, 0, 1<<30, c.ReplayOptions{MaxWorkingBytes: 256 << 20}); err != c.ErrResourceLimit {
		t.Fatal(err)
	}
	if r, err := model.Replay(nil, 1<<53, 0, c.ReplayOptions{Trace: true, MaxWorkingBytes: 16}); err != nil || len(r.Output) != 0 {
		t.Fatal(r, err)
	}
	var empty c.ConvertedSNN
	if _, err := empty.Replay(nil, 0, 0, options); err != c.ErrInvalidInput {
		t.Fatal(err)
	}
	var absent *c.ConvertedSNN
	if _, err := absent.Replay(nil, 0, 0, options); err != c.ErrInvalidInput {
		t.Fatal(err)
	}
}
