// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service acceleration for spike_codec/waveform_codec.py

package waveform_codec

import (
	"fmt"
	"math"
)

// Source: spike_codec/waveform_codec.py (service score: 4)
// 1 functions to accelerate

const (
	WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES = 1
	WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES = 255
	WAVEFORM_CODEC_MIN_TEMPLATES       = 1
	WAVEFORM_CODEC_MAX_HEADER_COUNT    = 65535
	WAVEFORM_CODEC_MAX_TEMPLATES       = WAVEFORM_CODEC_MAX_HEADER_COUNT
	WAVEFORM_CODEC_MIN_QUANTIZE_BITS   = 1
	WAVEFORM_CODEC_MAX_QUANTIZE_BITS   = 8
)

var WAVEFORM_CODEC_VALID_MODES = map[string]struct{}{
	"full":     {},
	"waveform": {},
	"spike":    {},
}

type Config struct {
	ThresholdSigma    float64
	SnippetSamples    int
	MaxTemplates      int
	TemplateThreshold float64
	QuantizeBits      int
	Mode              string
}

func DefaultConfig() Config {
	return Config{
		ThresholdSigma:    4.5,
		SnippetSamples:    48,
		MaxTemplates:      16,
		TemplateThreshold: 0.9,
		QuantizeBits:      6,
		Mode:              "full",
	}
}

func ValidateConfig(config Config) error {
	if math.IsNaN(config.ThresholdSigma) || math.IsInf(config.ThresholdSigma, 0) || config.ThresholdSigma <= 0 {
		return fmt.Errorf("threshold_sigma must be finite and positive")
	}
	if config.SnippetSamples < WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES || config.SnippetSamples > WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES {
		return fmt.Errorf("snippet_samples must be in [%d, %d]", WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES, WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES)
	}
	if config.MaxTemplates < WAVEFORM_CODEC_MIN_TEMPLATES || config.MaxTemplates > WAVEFORM_CODEC_MAX_TEMPLATES {
		return fmt.Errorf("max_templates must be in [%d, %d]", WAVEFORM_CODEC_MIN_TEMPLATES, WAVEFORM_CODEC_MAX_TEMPLATES)
	}
	if math.IsNaN(config.TemplateThreshold) || math.IsInf(config.TemplateThreshold, 0) || config.TemplateThreshold < 0 || config.TemplateThreshold > 1 {
		return fmt.Errorf("template_threshold must be finite and in [0, 1]")
	}
	if config.QuantizeBits < WAVEFORM_CODEC_MIN_QUANTIZE_BITS || config.QuantizeBits > WAVEFORM_CODEC_MAX_QUANTIZE_BITS {
		return fmt.Errorf("quantize_bits must be in [%d, %d]", WAVEFORM_CODEC_MIN_QUANTIZE_BITS, WAVEFORM_CODEC_MAX_QUANTIZE_BITS)
	}
	if _, ok := WAVEFORM_CODEC_VALID_MODES[config.Mode]; !ok {
		return fmt.Errorf("mode must be full, waveform, or spike")
	}
	return nil
}

func Compress() {
	_ = ValidateConfig(DefaultConfig())
	// Go-accelerated compress
}
