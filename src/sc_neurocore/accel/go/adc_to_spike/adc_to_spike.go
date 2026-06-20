// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go ADC-to-spike decimating rate-code encoder

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libadc_to_spike.so`) that the Python
// dispatcher loads via ctypes.
//
// Parity contract: `adc_to_spike_windows_c` produces bit-identical per-window
// values, spike counts and polarities as the Rust, Julia, Mojo and Python
// references. The per-window quantise/average/rate-code arithmetic is exact
// integer (Go `/` truncates toward zero, matching the reference), so the parity
// tolerance is zero.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

func absI64(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}

func quantiseAdc(sample int64, adcWidth, qInt, qFrac, signedInput int, qMin, qMax int64) int64 {
	qTotal := qInt + qFrac
	var centred int64
	if signedInput != 0 {
		signBit := int64(1) << (adcWidth - 1)
		mask := (int64(1) << adcWidth) - 1
		masked := sample & mask
		if masked&signBit != 0 {
			centred = masked - (int64(1) << adcWidth)
		} else {
			centred = masked
		}
	} else {
		centred = sample - (int64(1) << (adcWidth - 1))
	}

	var rounded int64
	if qTotal > adcWidth {
		rounded = centred << (qTotal - adcWidth)
	} else if adcWidth > qTotal {
		shift := adcWidth - qTotal
		half := int64(1) << (shift - 1)
		if centred >= 0 {
			rounded = (centred + half) >> shift
		} else {
			rounded = (centred - half) >> shift
		}
	} else {
		rounded = centred
	}
	if rounded < qMin {
		return qMin
	}
	if rounded > qMax {
		return qMax
	}
	return rounded
}

func averageWindow(total int64, decimation int, qMin, qMax int64) int64 {
	half := int64(decimation / 2)
	var adjusted int64
	if total >= 0 {
		adjusted = total + half
	} else {
		adjusted = total - half
	}
	averaged := adjusted / int64(decimation) // truncates toward zero
	if averaged < qMin {
		return qMin
	}
	if averaged > qMax {
		return qMax
	}
	return averaged
}

// adc_to_spike_windows_c — C-ABI entry point.
//
// Returns 0 on success, 1 on an invalid config or non-positive window count.
//
//export adc_to_spike_windows_c
func adc_to_spike_windows_c(
	nWindows, adcWidth, qInt, qFrac, decimation, signedInput C.int,
	thresholdQ C.longlong,
	samplesPtr, windowValuesPtr, spikeCountsPtr, polaritiesPtr unsafe.Pointer,
) C.int {
	nw := int(nWindows)
	aw := int(adcWidth)
	qi := int(qInt)
	qf := int(qFrac)
	decim := int(decimation)
	signed := int(signedInput)
	thr := int64(thresholdQ)
	if nw <= 0 || aw <= 1 || qi <= 0 || decim <= 0 || thr <= 0 {
		return 1
	}
	samples := unsafe.Slice((*C.int64_t)(samplesPtr), nw*decim)
	windowValues := unsafe.Slice((*C.int32_t)(windowValuesPtr), nw)
	spikeCounts := unsafe.Slice((*C.int32_t)(spikeCountsPtr), nw)
	polarities := unsafe.Slice((*C.uint8_t)(polaritiesPtr), nw)

	qTotal := qi + qf
	halfQ := int64(1) << (qTotal - 1)
	qMin := -halfQ
	qMax := halfQ - 1
	for w := 0; w < nw; w++ {
		base := w * decim
		var total int64
		for k := 0; k < decim; k++ {
			total += quantiseAdc(int64(samples[base+k]), aw, qi, qf, signed, qMin, qMax)
		}
		wq := averageWindow(total, decim, qMin, qMax)
		windowValues[w] = C.int32_t(wq)
		spikeCounts[w] = C.int32_t(absI64(wq) / thr)
		if wq < 0 {
			polarities[w] = 1
		} else {
			polarities[w] = 0
		}
	}
	return 0
}

func main() {}
