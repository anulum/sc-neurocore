// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Indexed SHD HDF5 recordings

#ifndef SC_SHD_HDF5_H
#define SC_SHD_HDF5_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    double *times;
    double *units;
    size_t count;
    int64_t label;
} sc_shd_sample;
/* Read one paired VL row; maximum bounds the eventual four-double event matrix. */
int sc_shd_read(const char *path, size_t index, size_t maximum, sc_shd_sample *sample);
/* Release both caller-transferred vectors, including on failed reads. */
void sc_shd_free(sc_shd_sample *sample);
#endif
