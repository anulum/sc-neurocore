// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Map-neuron PyO3 binding composition

use pyo3::prelude::*;

mod aihara_map;
#[path = "../cazelles_map.rs"]
mod cazelles_map;
#[path = "../chialvo_map.rs"]
mod chialvo_map;
#[path = "../courage_nekorkin_map.rs"]
mod courage_nekorkin_map;
#[path = "../ermentrout_kopell_map.rs"]
mod ermentrout_kopell_map;
#[path = "../ibarz_tanaka_map.rs"]
mod ibarz_tanaka_map;
mod kilinc_bhatt_map;
#[path = "../medvedev_map.rs"]
mod medvedev_map;
mod nagumo_sato_map;
#[path = "../rulkov_map.rs"]
mod rulkov_map;
mod sc_adaptive_threshold_map;
mod sc_chaotic_map;
#[path = "../sc_upward_crossing_rulkov_map.rs"]
mod sc_upward_crossing_rulkov_map;

pub use aihara_map::PyAiharaMapNeuron;
pub use cazelles_map::PyCazellesMapNeuron;
pub use chialvo_map::PyChialvoMapNeuron;
pub use courage_nekorkin_map::PyCourageNekorkinMapNeuron;
pub use ermentrout_kopell_map::PyErmentroutKopellMapNeuron;
pub use ibarz_tanaka_map::PyIbarzTanakaMapNeuron;
pub use kilinc_bhatt_map::PyKilincBhattMapNeuron;
pub use medvedev_map::PyMedvedevMapNeuron;
pub use nagumo_sato_map::PyNagumoSatoMapNeuron;
pub use rulkov_map::PyRulkovMapNeuron;
pub use sc_adaptive_threshold_map::PySCAdaptiveThresholdMapNeuron;
pub use sc_chaotic_map::PySCChaoticMapNeuron;
pub use sc_upward_crossing_rulkov_map::PySCUpwardCrossingRulkovMapNeuron;

/// Register the model-owned map-neuron bindings in stable class order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    chialvo_map::register(module)?;
    rulkov_map::register(module)?;
    sc_upward_crossing_rulkov_map::register(module)?;
    ibarz_tanaka_map::register(module)?;
    medvedev_map::register(module)?;
    cazelles_map::register(module)?;
    courage_nekorkin_map::register(module)?;
    aihara_map::register(module)?;
    sc_chaotic_map::register(module)?;
    nagumo_sato_map::register(module)?;
    sc_adaptive_threshold_map::register(module)?;
    kilinc_bhatt_map::register(module)?;
    ermentrout_kopell_map::register(module)?;
    Ok(())
}
