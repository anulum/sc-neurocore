// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HDF5 ABI declarations and identifier ownership

//! HDF5 C ABI declarations matching H5public/H5Ipublic/H5Tpublic headers.

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type Hid = i64;
pub type Hsize = u64;

#[repr(C)]
pub struct Vlen {
    pub len: usize,
    pub data: *mut c_void,
}

extern "C" {
    pub fn H5open() -> c_int;
    pub fn H5Fopen(name: *const c_char, flags: c_uint, properties: Hid) -> Hid;
    pub fn H5Fclose(file: Hid) -> c_int;
    pub fn H5Dopen2(file: Hid, name: *const c_char, properties: Hid) -> Hid;
    pub fn H5Dclose(dataset: Hid) -> c_int;
    pub fn H5Dget_space(dataset: Hid) -> Hid;
    pub fn H5Dget_type(dataset: Hid) -> Hid;
    pub fn H5Dread(
        dataset: Hid,
        datatype: Hid,
        memory: Hid,
        file: Hid,
        properties: Hid,
        output: *mut c_void,
    ) -> c_int;
    pub fn H5Dvlen_get_buf_size(dataset: Hid, datatype: Hid, space: Hid, size: *mut Hsize)
        -> c_int;
    pub fn H5Dvlen_reclaim(datatype: Hid, space: Hid, properties: Hid, value: *mut c_void)
        -> c_int;
    pub fn H5Sget_simple_extent_ndims(space: Hid) -> c_int;
    pub fn H5Sget_simple_extent_dims(
        space: Hid,
        dimensions: *mut Hsize,
        maximum: *mut Hsize,
    ) -> c_int;
    pub fn H5Sselect_hyperslab(
        space: Hid,
        operation: c_int,
        start: *const Hsize,
        stride: *const Hsize,
        count: *const Hsize,
        block: *const Hsize,
    ) -> c_int;
    pub fn H5Screate_simple(rank: c_int, dimensions: *const Hsize, maximum: *const Hsize) -> Hid;
    pub fn H5Sclose(space: Hid) -> c_int;
    pub fn H5Tget_class(datatype: Hid) -> c_int;
    pub fn H5Tget_super(datatype: Hid) -> Hid;
    pub fn H5Tget_size(datatype: Hid) -> usize;
    pub fn H5Tget_sign(datatype: Hid) -> c_int;
    pub fn H5Tvlen_create(base: Hid) -> Hid;
    pub fn H5Tclose(datatype: Hid) -> c_int;
    pub static H5T_NATIVE_DOUBLE_g: Hid;
    pub static H5T_NATIVE_INT64_g: Hid;
    pub static H5T_NATIVE_UINT64_g: Hid;
}

pub struct Id {
    pub raw: Hid,
    close: unsafe extern "C" fn(Hid) -> c_int,
}

impl Id {
    pub fn new(raw: Hid, close: unsafe extern "C" fn(Hid) -> c_int) -> std::io::Result<Self> {
        if raw < 0 {
            return Err(invalid("HDF5 object is unavailable"));
        }
        Ok(Self { raw, close })
    }
}

impl Drop for Id {
    fn drop(&mut self) {
        // Each id owns exactly one successful HDF5 open/create result.
        unsafe {
            (self.close)(self.raw);
        }
    }
}

pub fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}
