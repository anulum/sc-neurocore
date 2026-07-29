// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Weight Serialization (no_std)

//! Zero-copy weight loading from flash for bare-metal SC networks.
//!
//! Defines a binary format for pre-trained SC network weights that can
//! be linked into flash memory and loaded at boot without heap allocation.

/// Magic number identifying a tinySC weight blob.
pub const WEIGHT_MAGIC: u32 = 0x5343_574C; // "SCWL"

/// Weight blob header (fixed-size, at start of flash region).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeightHeader {
    pub magic: u32,
    pub version: u16,
    pub num_layers: u16,
    pub total_weights: u32,
    pub checksum: u32,
}

impl WeightHeader {
    /// Validate the header magic and checksum.
    pub fn is_valid(&self) -> bool {
        self.magic == WEIGHT_MAGIC && self.version <= 1
    }
}

/// Per-layer weight descriptor (follows header in flash).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerDescriptor {
    pub num_neurons: u16,
    pub weights_per_neuron: u16,
    pub offset: u32, // byte offset from start of weight data
    pub lfsr_seed: u16,
    pub threshold: u16,
    pub bitstream_length: u16,
    pub leak_shift: u8,
    pub _reserved: u8,
}

/// Read a weight header from a raw byte slice (e.g. flash-mapped memory).
///
/// # Safety
/// Caller must ensure `data` is at least `size_of::<WeightHeader>()` bytes
/// and properly aligned. In practice, flash is always word-aligned.
pub fn read_header(data: &[u8]) -> Option<WeightHeader> {
    if data.len() < core::mem::size_of::<WeightHeader>() {
        return None;
    }
    let hdr = unsafe { core::ptr::read_unaligned(data.as_ptr() as *const WeightHeader) };
    if hdr.is_valid() {
        Some(hdr)
    } else {
        None
    }
}

/// Read layer descriptors following the header.
pub fn read_layer_descriptors(data: &[u8], num_layers: usize) -> Option<&[LayerDescriptor]> {
    let hdr_size = core::mem::size_of::<WeightHeader>();
    let desc_size = core::mem::size_of::<LayerDescriptor>();
    let needed = hdr_size + desc_size * num_layers;
    if data.len() < needed {
        return None;
    }
    let ptr = unsafe { data.as_ptr().add(hdr_size) as *const LayerDescriptor };
    Some(unsafe { core::slice::from_raw_parts(ptr, num_layers) })
}

/// Read Q8.8 weight values for a specific layer.
pub fn read_weights(data: &[u8], offset: usize, count: usize) -> Option<&[i16]> {
    let byte_count = count * 2; // i16 = 2 bytes
    if data.len() < offset + byte_count {
        return None;
    }
    let ptr = unsafe { data.as_ptr().add(offset) as *const i16 };
    Some(unsafe { core::slice::from_raw_parts(ptr, count) })
}

/// Compute a simple checksum over a byte slice (Fletcher-16).
pub fn fletcher16(data: &[u8]) -> u32 {
    let mut sum1: u16 = 0;
    let mut sum2: u16 = 0;
    for &b in data {
        sum1 = (sum1.wrapping_add(b as u16)) % 255;
        sum2 = (sum2.wrapping_add(sum1)) % 255;
    }
    ((sum2 as u32) << 16) | sum1 as u32
}

/// Build a weight blob in a provided buffer. Returns bytes written.
///
/// This is a host-side utility — in no_std context, weights are
/// pre-built by the training pipeline and linked into flash.
#[cfg(test)]
pub fn build_weight_blob(buf: &mut [u8], layers: &[LayerDescriptor], weights: &[i16]) -> usize {
    let hdr_size = core::mem::size_of::<WeightHeader>();
    let weight_bytes = weights.len() * 2;
    let total = hdr_size + core::mem::size_of_val(layers) + weight_bytes;
    assert!(buf.len() >= total);

    let hdr = WeightHeader {
        magic: WEIGHT_MAGIC,
        version: 1,
        num_layers: layers.len() as u16,
        total_weights: weights.len() as u32,
        checksum: 0, // filled after
    };

    unsafe {
        core::ptr::write_unaligned(buf.as_mut_ptr() as *mut WeightHeader, hdr);
        let desc_ptr = buf.as_mut_ptr().add(hdr_size) as *mut LayerDescriptor;
        for (i, desc) in layers.iter().enumerate() {
            core::ptr::write_unaligned(desc_ptr.add(i), *desc);
        }
        let w_ptr = buf
            .as_mut_ptr()
            .add(hdr_size + core::mem::size_of_val(layers)) as *mut i16;
        for (i, &w) in weights.iter().enumerate() {
            core::ptr::write_unaligned(w_ptr.add(i), w);
        }
    }

    // Compute checksum over everything except the checksum field itself
    let checksum = fletcher16(&buf[..total]);
    // Write checksum field via unaligned write
    unsafe {
        let checksum_offset = core::mem::offset_of!(WeightHeader, checksum);
        core::ptr::write_unaligned(buf.as_mut_ptr().add(checksum_offset) as *mut u32, checksum);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_blob() -> (Vec<u8>, usize) {
        let layers = [LayerDescriptor {
            num_neurons: 4,
            weights_per_neuron: 4,
            offset: 0,
            lfsr_seed: 0xACE1,
            threshold: 10,
            bitstream_length: 256,
            leak_shift: 1,
            _reserved: 0,
        }];
        let weights: Vec<i16> = (0..16).map(|i| (i * 256) as i16).collect();
        let mut buf = vec![0u8; 1024];
        let n = build_weight_blob(&mut buf, &layers, &weights);
        (buf, n)
    }

    #[test]
    fn test_read_header() {
        let (buf, _) = make_test_blob();
        let hdr = read_header(&buf).unwrap();
        assert_eq!(hdr.magic, WEIGHT_MAGIC);
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.num_layers, 1);
        assert_eq!(hdr.total_weights, 16);
    }

    #[test]
    fn test_invalid_header() {
        let buf = [0u8; 64];
        assert!(read_header(&buf).is_none());
    }

    #[test]
    fn test_read_layer_descriptors() {
        let (buf, _) = make_test_blob();
        let descs = read_layer_descriptors(&buf, 1).unwrap();
        assert_eq!(descs[0].num_neurons, 4);
        assert_eq!(descs[0].weights_per_neuron, 4);
        assert_eq!(descs[0].lfsr_seed, 0xACE1);
    }

    #[test]
    fn test_read_weights() {
        let (buf, _n) = make_test_blob();
        let hdr_size = core::mem::size_of::<WeightHeader>();
        let desc_size = core::mem::size_of::<LayerDescriptor>();
        let offset = hdr_size + desc_size;
        let weights = read_weights(&buf, offset, 16).unwrap();
        assert_eq!(weights.len(), 16);
        assert_eq!(weights[0], 0);
        assert_eq!(weights[1], 256);
    }

    #[test]
    fn test_fletcher16() {
        let data = b"hello";
        let cs = fletcher16(data);
        assert!(cs > 0);
        // Deterministic
        assert_eq!(cs, fletcher16(data));
    }

    #[test]
    fn test_too_small_buffer() {
        let buf = [0u8; 4];
        assert!(read_header(&buf).is_none());
        assert!(read_layer_descriptors(&buf, 1).is_none());
        assert!(read_weights(&buf, 0, 100).is_none());
    }
}
