// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — ISI packet framing

//! Bounded reads and unsigned LEB128 for the existing ISI wire formats.

pub(super) struct PacketReader<'a> {
    remaining: &'a [u8],
}

impl<'a> PacketReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    pub(super) fn tail(&self) -> &'a [u8] {
        self.remaining
    }

    pub(super) fn finish(self) -> Result<(), &'static str> {
        if self.remaining.is_empty() {
            Ok(())
        } else {
            Err("unexpected trailing ISI packet data")
        }
    }

    pub(super) fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let (head, tail) = self
            .remaining
            .split_at_checked(len)
            .ok_or("truncated ISI packet")?;
        self.remaining = tail;
        Ok(head)
    }

    pub(super) fn u16(&mut self) -> Result<u16, &'static str> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub(super) fn u32(&mut self) -> Result<u32, &'static str> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(super) fn section(&mut self) -> Result<PacketReader<'a>, &'static str> {
        let len = self.u32()? as usize;
        Ok(Self::new(self.take(len)?))
    }

    pub(super) fn varint(&mut self) -> Result<u32, &'static str> {
        let mut value = 0;
        for shift in (0..=28).step_by(7) {
            let byte = self.take(1)?[0];
            if shift == 28 && byte > 15 {
                return Err("ISI varint exceeds uint32");
            }
            value |= u32::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                if byte == 0 && shift != 0 {
                    return Err("nonminimal ISI varint");
                }
                return Ok(value);
            }
        }
        Err("ISI varint exceeds uint32")
    }
}

pub(super) fn write_varint(mut value: u32, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

pub(super) fn write_section(bytes: &[u8], output: &mut Vec<u8>) -> Result<(), &'static str> {
    let len = u32::try_from(bytes.len()).map_err(|_| "ISI section exceeds uint32")?;
    output.extend_from_slice(&len.to_be_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}
