// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Autonomous-learning state codec

//! Bounds-checked, endian-stable state transport for native rule layers.

const MAGIC: &[u8; 4] = b"SCAL";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 12;
const RECORD_HEADER_BYTES: usize = 8;

#[derive(Debug, PartialEq)]
pub(crate) struct RuleState {
    pub(crate) rule_id: u32,
    pub(crate) values: Vec<f32>,
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(input: &[u8], offset: &mut usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let bytes: [u8; 4] = input.get(*offset..end)?.try_into().ok()?;
    *offset = end;
    Some(u32::from_le_bytes(bytes))
}

pub(crate) fn encode(records: &[RuleState]) -> Option<Vec<u8>> {
    let count = u32::try_from(records.len()).ok()?;
    let payload_bytes = records.iter().try_fold(0usize, |total, record| {
        let values_bytes = record.values.len().checked_mul(4)?;
        total
            .checked_add(RECORD_HEADER_BYTES)?
            .checked_add(values_bytes)
    })?;
    let capacity = HEADER_BYTES.checked_add(payload_bytes)?;
    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(MAGIC);
    push_u32(&mut output, VERSION);
    push_u32(&mut output, count);
    for record in records {
        if record.values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        push_u32(&mut output, record.rule_id);
        push_u32(&mut output, u32::try_from(record.values.len()).ok()?);
        for value in &record.values {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    debug_assert_eq!(output.len(), capacity);
    Some(output)
}

pub(crate) fn decode(input: &[u8], expected: &[(u32, usize)]) -> Option<Vec<RuleState>> {
    if input.get(..4)? != MAGIC {
        return None;
    }
    let mut offset = 4;
    if read_u32(input, &mut offset)? != VERSION {
        return None;
    }
    if usize::try_from(read_u32(input, &mut offset)?).ok()? != expected.len() {
        return None;
    }
    let mut records = Vec::with_capacity(expected.len());
    for &(expected_id, expected_values) in expected {
        let rule_id = read_u32(input, &mut offset)?;
        let value_count = usize::try_from(read_u32(input, &mut offset)?).ok()?;
        if rule_id != expected_id || value_count != expected_values {
            return None;
        }
        let byte_count = value_count.checked_mul(4)?;
        let end = offset.checked_add(byte_count)?;
        let payload = input.get(offset..end)?;
        let mut values = Vec::with_capacity(value_count);
        for bytes in payload.chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().ok()?);
            if !value.is_finite() {
                return None;
            }
            values.push(value);
        }
        records.push(RuleState { rule_id, values });
        offset = end;
    }
    (offset == input.len()).then_some(records)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, RuleState};

    fn records() -> Vec<RuleState> {
        vec![RuleState {
            rule_id: 1,
            values: vec![0.5, 0.25, -0.125],
        }]
    }

    #[test]
    fn round_trip_is_exact() {
        let encoded = encode(&records()).expect("valid state must encode");
        assert_eq!(decode(&encoded, &[(1, 3)]), Some(records()));
    }

    #[test]
    fn rejects_every_truncation() {
        let encoded = encode(&records()).expect("valid state must encode");
        for end in 0..encoded.len() {
            assert_eq!(decode(&encoded[..end], &[(1, 3)]), None);
        }
    }

    #[test]
    fn rejects_trailing_bytes_and_wrong_layout() {
        let mut encoded = encode(&records()).expect("valid state must encode");
        encoded.push(0);
        assert_eq!(decode(&encoded, &[(1, 3)]), None);
        assert_eq!(decode(&encoded[..encoded.len() - 1], &[(2, 3)]), None);
        assert_eq!(decode(&encoded[..encoded.len() - 1], &[(1, 4)]), None);
    }

    #[test]
    fn rejects_non_finite_values() {
        let record = RuleState {
            rule_id: 1,
            values: vec![f32::NAN],
        };
        assert_eq!(encode(&[record]), None);
    }
}
