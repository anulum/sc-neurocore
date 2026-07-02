// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go waveform codec validation tests

package waveform_codec

import (
	"math"
	"testing"
)

func TestDefaultConfigIsValid(t *testing.T) {
	if err := ValidateConfig(DefaultConfig()); err != nil {
		t.Fatalf("default config should be valid: %v", err)
	}
}

func TestValidateConfigRejectsHeaderUnsafeRanges(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(Config) Config
	}{
		{"threshold_sigma", func(c Config) Config { c.ThresholdSigma = math.NaN(); return c }},
		{"snippet_samples", func(c Config) Config { c.SnippetSamples = WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES + 1; return c }},
		{"max_templates", func(c Config) Config { c.MaxTemplates = WAVEFORM_CODEC_MAX_TEMPLATES + 1; return c }},
		{"template_threshold", func(c Config) Config { c.TemplateThreshold = 1.01; return c }},
		{"quantize_bits", func(c Config) Config { c.QuantizeBits = WAVEFORM_CODEC_MAX_QUANTIZE_BITS + 1; return c }},
		{"mode", func(c Config) Config { c.Mode = "turbo"; return c }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateConfig(tc.mutate(DefaultConfig())); err == nil {
				t.Fatalf("expected %s validation to fail", tc.name)
			}
		})
	}
}
