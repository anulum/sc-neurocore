// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust indexed HDF5 recordings

//! Indexed HDF5 row selection, typed reads and allocation ownership.

use crate::{hdf5::*, Recording};
use std::{ffi::CStr, io, ptr, sync::Mutex};

static HDF5: Mutex<()> = Mutex::new(());

fn select_row(dataset: &Id, index: usize) -> io::Result<(Id, Hsize)> {
    let space = Id::new(unsafe { H5Dget_space(dataset.raw) }, H5Sclose)?;
    let mut length = 0;
    let start = index as Hsize;
    let one = 1;
    if unsafe { H5Sget_simple_extent_ndims(space.raw) } != 1
        || unsafe { H5Sget_simple_extent_dims(space.raw, &mut length, ptr::null_mut()) } < 0
        || start >= length
        || unsafe { H5Sselect_hyperslab(space.raw, 0, &start, ptr::null(), &one, ptr::null()) } < 0
    {
        return Err(invalid("SHD dataset shape or index is incompatible"));
    }
    Ok((space, length))
}

fn numeric_vlen(dataset: &Id) -> io::Result<()> {
    let datatype = Id::new(unsafe { H5Dget_type(dataset.raw) }, H5Tclose)?;
    if unsafe { H5Tget_class(datatype.raw) } != 9 {
        return Err(invalid("SHD events must be variable-length vectors"));
    }
    let base = Id::new(unsafe { H5Tget_super(datatype.raw) }, H5Tclose)?;
    if !matches!(unsafe { H5Tget_class(base.raw) }, 0 | 1) {
        return Err(invalid("SHD events must be numeric"));
    }
    Ok(())
}

struct Vector<'a> {
    value: Vlen,
    datatype: &'a Id,
    memory: &'a Id,
    reclaimed: bool,
}

impl Vector<'_> {
    fn reclaim(&mut self) -> io::Result<()> {
        self.reclaimed = true;
        if unsafe {
            H5Dvlen_reclaim(
                self.datatype.raw,
                self.memory.raw,
                0,
                (&mut self.value as *mut Vlen).cast(),
            )
        } < 0
        {
            return Err(invalid("HDF5 vector reclaim failed"));
        }
        Ok(())
    }
}

impl Drop for Vector<'_> {
    fn drop(&mut self) {
        if !self.reclaimed {
            let _ = self.reclaim();
        }
    }
}

fn read_vector(
    dataset: &Id,
    space: &Id,
    memory: &Id,
    datatype: &Id,
    maximum: usize,
) -> io::Result<Vec<f64>> {
    let mut vector = Vector {
        value: Vlen {
            len: 0,
            data: ptr::null_mut(),
        },
        datatype,
        memory,
        reclaimed: false,
    };
    if unsafe {
        H5Dread(
            dataset.raw,
            datatype.raw,
            memory.raw,
            space.raw,
            0,
            (&mut vector.value as *mut Vlen).cast(),
        )
    } < 0
    {
        return Err(invalid("HDF5 vector read failed"));
    }
    if vector.value.len > maximum / 32 || (vector.value.len != 0 && vector.value.data.is_null()) {
        return Err(invalid("SHD recording exceeds its event budget"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(vector.value.len)
        .map_err(io::Error::other)?;
    if vector.value.len != 0 {
        // H5Dread owns a native-double buffer until its matching reclaim.
        values.extend_from_slice(unsafe {
            std::slice::from_raw_parts(vector.value.data.cast::<f64>(), vector.value.len)
        });
    }
    vector.reclaim()?;
    Ok(values)
}

fn read_label(dataset: &Id, memory: &Id, space: &Id) -> io::Result<i64> {
    let datatype = Id::new(unsafe { H5Dget_type(dataset.raw) }, H5Tclose)?;
    if unsafe { H5Tget_class(datatype.raw) } != 0 || unsafe { H5Tget_size(datatype.raw) } > 8 {
        return Err(invalid("SHD labels must be representable integers"));
    }
    if unsafe { H5Tget_sign(datatype.raw) } == 0 {
        let mut label: u64 = 0;
        if unsafe {
            H5Dread(
                dataset.raw,
                H5T_NATIVE_UINT64_g,
                memory.raw,
                space.raw,
                0,
                (&mut label as *mut u64).cast(),
            )
        } < 0
        {
            return Err(invalid("HDF5 label read failed"));
        }
        i64::try_from(label).map_err(|_| invalid("SHD label exceeds int64"))
    } else {
        let mut label: i64 = 0;
        if unsafe {
            H5Dread(
                dataset.raw,
                H5T_NATIVE_INT64_g,
                memory.raw,
                space.raw,
                0,
                (&mut label as *mut i64).cast(),
            )
        } < 0
        {
            return Err(invalid("HDF5 label read failed"));
        }
        Ok(label)
    }
}

pub fn read(path: &CStr, index: usize, maximum: usize) -> io::Result<Recording> {
    let _guard = HDF5
        .lock()
        .map_err(|_| io::Error::other("HDF5 lock is poisoned"))?;
    if maximum > isize::MAX as usize || unsafe { H5open() } < 0 {
        return Err(invalid("SHD budget or HDF5 initialisation is invalid"));
    }
    let file = Id::new(unsafe { H5Fopen(path.as_ptr(), 0, 0) }, H5Fclose)?;
    let times = Id::new(
        unsafe { H5Dopen2(file.raw, c"spikes/times".as_ptr(), 0) },
        H5Dclose,
    )?;
    let units = Id::new(
        unsafe { H5Dopen2(file.raw, c"spikes/units".as_ptr(), 0) },
        H5Dclose,
    )?;
    let labels = Id::new(
        unsafe { H5Dopen2(file.raw, c"labels".as_ptr(), 0) },
        H5Dclose,
    )?;
    numeric_vlen(&times)?;
    numeric_vlen(&units)?;
    let (ts, tl) = select_row(&times, index)?;
    let (us, ul) = select_row(&units, index)?;
    let (ls, ll) = select_row(&labels, index)?;
    if tl != ul || tl != ll {
        return Err(invalid("SHD recording counts differ"));
    }
    let one = 1;
    let memory = Id::new(unsafe { H5Screate_simple(1, &one, ptr::null()) }, H5Sclose)?;
    let datatype = Id::new(unsafe { H5Tvlen_create(H5T_NATIVE_DOUBLE_g) }, H5Tclose)?;
    let mut tb = 0;
    let mut ub = 0;
    if unsafe { H5Dvlen_get_buf_size(times.raw, datatype.raw, ts.raw, &mut tb) } < 0
        || unsafe { H5Dvlen_get_buf_size(units.raw, datatype.raw, us.raw, &mut ub) } < 0
    {
        return Err(invalid("HDF5 vector allocation estimate failed"));
    }
    if tb > maximum as u64 / 2 || ub > maximum as u64 / 2 - tb {
        return Err(invalid("SHD recording exceeds its event budget"));
    }
    let time = read_vector(&times, &ts, &memory, &datatype, maximum)?;
    let channels = read_vector(&units, &us, &memory, &datatype, maximum)?;
    if time.len() != channels.len() {
        return Err(invalid("SHD event vectors differ"));
    }
    let label = read_label(&labels, &memory, &ls)?;
    let mut events = Vec::new();
    events
        .try_reserve_exact(time.len() * 4)
        .map_err(io::Error::other)?;
    for (channel, seconds) in channels.iter().zip(time) {
        events.extend_from_slice(&[*channel, 0.0, 0.0, seconds * 1000.0]);
    }
    Ok(Recording { events, label })
}
