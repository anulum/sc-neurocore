// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Indexed SHD HDF5 recordings

//go:build hdf5

#include "shd_hdf5.h"
#include <hdf5.h>
#include <stdlib.h>
#include <string.h>

static hid_t select_row(hid_t dataset, size_t index, hsize_t *length) {
    hid_t space = H5Dget_space(dataset);
    hsize_t dimensions[1], start[1] = {index}, count[1] = {1};
    if (space < 0) return -1;
    if (H5Sget_simple_extent_ndims(space) != 1 ||
        H5Sget_simple_extent_dims(space, dimensions, NULL) < 0 ||
        index >= dimensions[0] ||
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start, NULL, count, NULL) < 0) {
        H5Sclose(space);
        return -1;
    }
    *length = dimensions[0];
    return space;
}

static int numeric_vlen(hid_t dataset) {
    hid_t type = H5Dget_type(dataset), base = -1;
    int valid = 0;
    if (type < 0) return 0;
    if (H5Tget_class(type) == H5T_VLEN) base = H5Tget_super(type);
    if (base >= 0) {
        H5T_class_t category = H5Tget_class(base);
        valid = category == H5T_FLOAT || category == H5T_INTEGER;
        H5Tclose(base);
    }
    H5Tclose(type);
    return valid;
}

static int read_vector(hid_t dataset, hid_t file_space, hid_t memory_space,
                       hid_t type, double **output, size_t *count) {
    hvl_t value = {0, NULL};
    int result = -1;
    if (H5Dread(dataset, type, memory_space, file_space, H5P_DEFAULT, &value) >= 0) {
        if (value.len <= SIZE_MAX / sizeof(double)) {
            *count = value.len;
            if (value.len == 0) result = 0;
            else if ((*output = malloc(value.len * sizeof(double))) != NULL) {
                memcpy(*output, value.p, value.len * sizeof(double));
                result = 0;
            }
        }
    }
    if (H5Dvlen_reclaim(type, memory_space, H5P_DEFAULT, &value) < 0) result = -1;
    return result;
}

void sc_shd_free(sc_shd_sample *sample) {
    free(sample->times);
    free(sample->units);
    memset(sample, 0, sizeof(*sample));
}

int sc_shd_read(const char *path, size_t index, size_t maximum, sc_shd_sample *sample) {
    hid_t file = -1, times = -1, units = -1, labels = -1;
    hid_t ts = -1, us = -1, ls = -1, memory = -1, type = -1, label_type = -1;
    hsize_t tl = 0, ul = 0, ll = 0, one = 1, tb = 0, ub = 0;
    size_t time_count = 0, unit_count = 0;
    int result = -1;
    memset(sample, 0, sizeof(*sample));
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) goto cleanup;
    times = H5Dopen2(file, "spikes/times", H5P_DEFAULT);
    units = H5Dopen2(file, "spikes/units", H5P_DEFAULT);
    labels = H5Dopen2(file, "labels", H5P_DEFAULT);
    if (times < 0 || units < 0 || labels < 0) goto cleanup;
    if (!numeric_vlen(times) || !numeric_vlen(units)) goto cleanup;
    ts = select_row(times, index, &tl);
    us = select_row(units, index, &ul);
    ls = select_row(labels, index, &ll);
    if (ts < 0 || us < 0 || ls < 0 || tl != ul || tl != ll) goto cleanup;
    memory = H5Screate_simple(1, &one, NULL);
    type = H5Tvlen_create(H5T_NATIVE_DOUBLE);
    label_type = H5Dget_type(labels);
    if (memory < 0 || type < 0 || label_type < 0 || H5Tget_class(label_type) != H5T_INTEGER)
        goto cleanup;
    if (H5Dvlen_get_buf_size(times, type, ts, &tb) < 0 ||
        H5Dvlen_get_buf_size(units, type, us, &ub) < 0) goto cleanup;
    if (tb > maximum / 2 || ub > maximum / 2 - tb) { result = -2; goto cleanup; }
    if (read_vector(times, ts, memory, type, &sample->times, &time_count) < 0 ||
        read_vector(units, us, memory, type, &sample->units, &unit_count) < 0 ||
        time_count != unit_count || time_count > maximum / (4 * sizeof(double))) goto cleanup;
    if (H5Tget_size(label_type) > sizeof(int64_t)) goto cleanup;
    if (H5Tget_sign(label_type) == H5T_SGN_NONE) {
        uint64_t label = 0;
        if (H5Dread(labels, H5T_NATIVE_UINT64, memory, ls, H5P_DEFAULT, &label) < 0 ||
            label > INT64_MAX) goto cleanup;
        sample->label = (int64_t)label;
    } else if (H5Dread(labels, H5T_NATIVE_INT64, memory, ls, H5P_DEFAULT, &sample->label) < 0) {
        goto cleanup;
    }
    sample->count = time_count;
    result = 0;
cleanup:
    if (label_type >= 0) H5Tclose(label_type);
    if (type >= 0) H5Tclose(type);
    if (memory >= 0) H5Sclose(memory);
    if (ls >= 0) H5Sclose(ls);
    if (us >= 0) H5Sclose(us);
    if (ts >= 0) H5Sclose(ts);
    if (labels >= 0) H5Dclose(labels);
    if (units >= 0) H5Dclose(units);
    if (times >= 0) H5Dclose(times);
    if (file >= 0) H5Fclose(file);
    if (result != 0) sc_shd_free(sample);
    return result;
}
