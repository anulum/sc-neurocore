// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SCPN Module

pub mod dcls;
pub mod kuramoto;
pub mod metrics;

pub use dcls::{
    dcls_max_forward_q88, tent_gate_q88, DclsError, DclsForwardResult, DclsLayerConfig,
};
pub use kuramoto::KuramotoSolver;
pub use metrics::SCPNMetrics;
