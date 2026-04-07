// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU buffer allocation and upload utilities.

use wgpu;

/// Create a storage buffer with initial data.
pub fn storage_buffer_init(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    read_only: bool,
) -> wgpu::Buffer {
    let usage = if read_only {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    };
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len() as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Create a storage buffer with a given size (no initial data).
pub fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    read_only: bool,
) -> wgpu::Buffer {
    let usage = if read_only {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    };
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Create a uniform buffer with initial data.
pub fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create a staging (MAP_READ) buffer for downloading results.
pub fn staging_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
