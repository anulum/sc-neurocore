// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

/* Host long-double conversion; the Rust caller supplies exactly 16 live bytes. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

size_t sc_dvs_extended_width(void) { return sizeof(long double); }
size_t sc_dvs_long_width(void) { return sizeof(long); }
double sc_dvs_extended(const unsigned char *source, int big_endian) {
    long double value = 0;
    unsigned char *bytes = (unsigned char *)&value;
    const uint16_t one = 1;
    const int host_big = ((const unsigned char *)&one)[0] == 0;
    if (sizeof(value) != 16) return 0; /* Rust refuses this ABI before calling. */
    for (size_t i = 0; i < sizeof(value); ++i)
        bytes[i] = source[host_big == big_endian ? i : sizeof(value) - 1 - i];
    return (double)value;
}
