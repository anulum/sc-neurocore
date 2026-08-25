// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Biophysical-neuron PyO3 binding composition

use pyo3::prelude::*;

mod av_ron_cardiac;
mod bertram_phantom_burster;
mod connor_stevens;
mod de_schutter_purkinje;
mod destexhe_thalamic;
mod durstewitz_dopamine;
mod gif_population;
#[path = "../glif.rs"]
mod glif;
mod golomb_fs;
mod hill_tononi;
mod hodgkin_huxley;
mod huber_braun;
mod mainen_sejnowski;
#[path = "../mihalas_niebur.rs"]
mod mihalas_niebur;
mod plant_r15;
mod pospischil;
mod prescott;
mod sc_three_state_phantom_burster;
mod traub_miles;
mod wang_buzsaki;
mod yamada;

pub use av_ron_cardiac::PyAvRonCardiacNeuron;
pub use bertram_phantom_burster::PyBertramPhantomBurster;
pub use connor_stevens::PyConnorStevensNeuron;
pub use de_schutter_purkinje::PyDeSchutterPurkinjeNeuron;
pub use destexhe_thalamic::PyDestexheThalamicNeuron;
pub use durstewitz_dopamine::PyDurstewitzDopamineNeuron;
pub use gif_population::PyGIFPopulationNeuron;
pub use glif::PyGLIFNeuron;
pub use golomb_fs::PyGolombFSNeuron;
pub use hill_tononi::PyHillTononiNeuron;
pub use hodgkin_huxley::PyHodgkinHuxleyNeuron;
pub use huber_braun::PyHuberBraunNeuron;
pub use mainen_sejnowski::PyMainenSejnowskiNeuron;
pub use mihalas_niebur::PyMihalasNieburNeuron;
pub use plant_r15::PyPlantR15Neuron;
pub use pospischil::PyPospischilNeuron;
pub use prescott::PyPrescottNeuron;
pub use sc_three_state_phantom_burster::PySCThreeStatePhantomBurster;
pub use traub_miles::PyTraubMilesNeuron;
pub use wang_buzsaki::PyWangBuzsakiNeuron;
pub use yamada::PyYamadaNeuron;

/// Register the twenty model-owned biophysical neuron bindings in stable class order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    hodgkin_huxley::register(module)?;
    traub_miles::register(module)?;
    wang_buzsaki::register(module)?;
    connor_stevens::register(module)?;
    destexhe_thalamic::register(module)?;
    huber_braun::register(module)?;
    golomb_fs::register(module)?;
    pospischil::register(module)?;
    mainen_sejnowski::register(module)?;
    de_schutter_purkinje::register(module)?;
    plant_r15::register(module)?;
    prescott::register(module)?;
    mihalas_niebur::register(module)?;
    glif::register(module)?;
    gif_population::register(module)?;
    av_ron_cardiac::register(module)?;
    durstewitz_dopamine::register(module)?;
    hill_tononi::register(module)?;
    bertram_phantom_burster::register(module)?;
    sc_three_state_phantom_burster::register(module)?;
    yamada::register(module)?;
    Ok(())
}
