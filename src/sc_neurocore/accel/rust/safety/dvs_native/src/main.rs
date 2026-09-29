// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Bounded DVS binary transport owned by its invoking Linux process.
use std::{
    io::{self, Write},
    path::PathBuf,
};
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn prctl(option: std::ffi::c_int, ...) -> std::ffi::c_int;
    fn getppid() -> std::ffi::c_int;
    fn alarm(seconds: std::ffi::c_uint) -> std::ffi::c_uint;
}
/// Arm Linux parent death and a deadline covering input and output blocking.
#[cfg(target_os = "linux")]
fn guard(expected: Option<i32>) -> io::Result<()> {
    // SAFETY: getppid has no arguments and does not dereference pointers.
    let parent = unsafe { getppid() };
    let expected = expected.unwrap_or(parent);
    if expected <= 0 || parent != expected {
        return Err(io::Error::other("DVS parent is absent"));
    }
    // SAFETY: PR_SET_PDEATHSIG accepts integer SIGKILL; remaining variadic slots are zero.
    let status = unsafe {
        prctl(
            1,
            9 as std::ffi::c_ulong,
            0 as std::ffi::c_ulong,
            0 as std::ffi::c_ulong,
            0 as std::ffi::c_ulong,
        )
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: no pointers; compare after prctl closes the startup race. alarm sets process SIGALRM.
    unsafe {
        if getppid() != expected {
            return Err(io::Error::other("DVS parent changed"));
        }
        alarm(30);
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn guard(_: Option<i32>) -> io::Result<()> {
    Err(io::Error::other(
        "DVS command requires Linux lifetime guards",
    ))
}
/// Validate native arguments, read one recording and emit an exact little-endian frame.
fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(io::Error::other("path budget [expected-parent] required"));
    }
    let budget = args[1]
        .to_str()
        .ok_or_else(|| io::Error::other("invalid budget encoding"))?
        .parse::<usize>()
        .map_err(io::Error::other)?;
    let expected = if args.len() == 3 {
        Some(
            args[2]
                .to_str()
                .ok_or_else(|| io::Error::other("invalid parent encoding"))?
                .parse::<i32>()
                .map_err(io::Error::other)?,
        )
    } else {
        None
    };
    guard(expected)?;
    let events = sc_neurocore_dvs::read_dvs_recording(PathBuf::from(&args[0]), budget)?;
    let count = u64::try_from(events.len()).map_err(io::Error::other)?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    output.write_all(b"DVS1")?;
    output.write_all(&count.to_le_bytes())?;
    for value in events {
        output.write_all(&value.to_le_bytes())?
    }
    output.flush()
}
fn main() {
    if let Err(error) = run() {
        eprintln!("DVS recording refused: {error}");
        std::process::exit(1)
    }
}
