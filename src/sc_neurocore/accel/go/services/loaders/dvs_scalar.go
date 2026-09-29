// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders

import (
	"encoding/binary"
	"math"
	"math/big"
)

type dvsDtype struct {
	kind      byte
	width     int
	bigEndian bool
}

func (dtype dvsDtype) decode(raw []byte) float64 {
	var normalized [16]byte
	if dtype.bigEndian {
		for i, value := range raw {
			normalized[len(raw)-1-i] = value
		}
	} else {
		copy(normalized[:], raw)
	}
	bytes := normalized[:]
	if dtype.kind == 'b' {
		if bytes[0] == 0 {
			return 0
		}
		return 1
	}
	bits := uint64(0)
	for i := 0; i < dtype.width && i < 8; i++ {
		bits |= uint64(bytes[i]) << uint(i*8)
	}
	if dtype.kind == 'u' {
		return float64(bits)
	}
	if dtype.kind == 'i' {
		shift := uint(64 - dtype.width*8)
		return float64(int64(bits<<shift) >> shift)
	}
	switch dtype.width {
	case 2:
		exponent := int(bits >> 10 & 31)
		fraction := bits & 1023
		value := math.Ldexp(float64(fraction), -24)
		if exponent == 31 {
			value = math.Inf(1)
			if fraction != 0 {
				value = math.NaN()
			}
		} else if exponent != 0 {
			value = math.Ldexp(float64(1024+fraction), exponent-25)
		}
		if bits&32768 != 0 {
			value = math.Copysign(value, -1)
		}
		return value
	case 4:
		return float64(math.Float32frombits(uint32(bits)))
	case 8:
		return math.Float64frombits(bits)
	default:
		signExponent := binary.LittleEndian.Uint16(bytes[8:10])
		exponent := int(signExponent & 32767)
		negative := signExponent&32768 != 0
		if exponent == 32767 {
			if bits != uint64(1)<<63 {
				return math.NaN()
			}
			if negative {
				return math.Inf(-1)
			}
			return math.Inf(1)
		}
		if exponent != 0 && bits>>63 == 0 {
			return math.NaN()
		}
		if bits == 0 {
			if negative {
				return math.Copysign(0, -1)
			}
			return 0
		}
		if exponent == 0 {
			exponent = 1
		}
		value := new(big.Float).SetPrec(64).SetMode(big.ToNearestEven).SetUint64(bits)
		value.SetMantExp(value, exponent-16383-63)
		if negative {
			value.Neg(value)
		}
		output, _ := value.Float64()
		return output
	}
}
