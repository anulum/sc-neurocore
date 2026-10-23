// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source and graph identity

//! Non-executable source provenance bound to exact canonical graph text.
//!
//! SHA-256 detects stale or damaged metadata; it is not a signature or proof
//! that an untrusted author's description correctly explains the operations.

#![warn(missing_docs)]

use super::graph::ScGraph;
use sha2::{Digest, Sha256};

/// Source bytes and their binding to one exact graph, preserved by SCIR text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    /// UTF-8 provenance, such as the equation frontend's canonical JSON contract.
    pub payload: String,
    /// Lowercase SHA-256 of the exact UTF-8 source payload.
    pub source_sha256: String,
    /// Lowercase SHA-256 of canonical graph text with source metadata omitted.
    pub graph_sha256: String,
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}

fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

impl SourceIdentity {
    /// Bind nonempty provenance to the current complete graph.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty payload without changing the graph.
    pub fn new(payload: String, graph: &ScGraph) -> Result<Self, String> {
        if payload.is_empty() {
            return Err("source payload must not be empty".into());
        }
        Ok(Self {
            source_sha256: digest(payload.as_bytes()),
            graph_sha256: digest(super::printer::print_without_source(graph).as_bytes()),
            payload,
        })
    }

    /// Check both the source checksum and its exact current graph binding.
    ///
    /// # Errors
    ///
    /// Refuses malformed digests, changed source bytes or changed graph operations/name.
    pub fn verify(&self, graph: &ScGraph) -> Result<(), String> {
        self.verify_payload()?;
        if self.graph_sha256 != digest(super::printer::print_without_source(graph).as_bytes()) {
            return Err("source identity graph checksum mismatch".into());
        }
        Ok(())
    }

    fn verify_payload(&self) -> Result<(), String> {
        if self.payload.is_empty() {
            return Err("source payload must not be empty".into());
        }
        for hash in [&self.source_sha256, &self.graph_sha256] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("source identity requires lowercase SHA-256 digests".into());
            }
        }
        if self.source_sha256 != digest(self.payload.as_bytes()) {
            return Err("source identity payload checksum mismatch".into());
        }
        Ok(())
    }

    pub(crate) fn to_line(&self) -> String {
        let payload = hex(self.payload.as_bytes());
        format!(
            "sc.source v1 source={} graph={} payload={payload}",
            self.source_sha256, self.graph_sha256
        )
    }
}

fn nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err("source payload requires lowercase hexadecimal UTF-8 bytes".into()),
    }
}

pub(crate) fn parse_line(line: &str) -> Result<SourceIdentity, String> {
    let rest = line
        .strip_prefix("sc.source v1 ")
        .ok_or("unsupported source identity version")?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let [source, graph, payload] = fields.as_slice() else {
        return Err("source identity requires exactly source, graph and payload fields".into());
    };
    let source_sha256 = source
        .strip_prefix("source=")
        .ok_or("missing source checksum")?
        .to_string();
    let graph_sha256 = graph
        .strip_prefix("graph=")
        .ok_or("missing graph checksum")?
        .to_string();
    let payload = payload
        .strip_prefix("payload=")
        .ok_or("missing source payload")?;
    if !payload.len().is_multiple_of(2) {
        return Err("source payload has an odd hexadecimal length".into());
    }
    let bytes: Result<Vec<u8>, String> = payload
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(nibble(pair[0])? * 16 + nibble(pair[1])?))
        .collect();
    let result = SourceIdentity {
        payload: String::from_utf8(bytes?).map_err(|_| "source payload is not UTF-8")?,
        source_sha256,
        graph_sha256,
    };
    result.verify_payload()?;
    Ok(result)
}
