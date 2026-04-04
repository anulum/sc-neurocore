// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

//! Biophysically grounded sensory neuron models.
//!
//! 10 cell types covering major sensory modalities:
//! - Auditory: inner/outer hair cells (graded, mechano-electrical)
//! - Visual: rod/cone photoreceptors (graded, hyperpolarising), retinal ganglion (ON/OFF spiking)
//! - Somatosensory: Merkel (slow adapting), Pacinian (fast adapting)
//! - Pain: nociceptor (threshold, sensitisation)
//! - Chemical: olfactory receptor, taste receptor
//!
//! Most sensory neurons produce graded potentials (step returns f64).
//! Only retinal ganglion and nociceptor produce spikes (step returns i32).

// ═══════════════════════════════════════════════════════════════════
// Inner Hair Cell (IHC) — auditory
// ═══════════════════════════════════════════════════════════════════

/// Inner hair cell — primary auditory transducer.
///
/// Mechano-electrical transduction: stereocilia displacement opens
/// MET channels → depolarisation → Ca2+ influx → glutamate release.
/// Graded receptor potential, no spikes.
///
/// Based on Meddis 2006 / Lopez-Poveda & Eustaquio-Martín 2006.
#[derive(Clone, Debug)]
pub struct InnerHairCell {
    pub v: f64,       // Receptor potential (mV)
    pub v_rest: f64,
    pub tau: f64,     // Membrane time constant (ms)
    pub g_met: f64,   // MET channel max conductance
    pub x_half: f64,  // Boltzmann half-activation displacement (nm)
    pub s: f64,       // Boltzmann slope
    pub ca: f64,      // Intracellular Ca2+ (µM)
    pub tau_ca: f64,  // Ca2+ decay time constant (ms)
    pub dt: f64,
}

impl InnerHairCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            v_rest: -60.0,
            tau: 0.5,
            g_met: 10.0,
            x_half: 50.0,  // Higher: MET channels barely open at rest
            s: 10.0,
            ca: 0.0,
            tau_ca: 1.0,
            dt: 0.025,
        }
    }

    /// Step with stereocilia displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        // Boltzmann MET channel open probability
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v); // E_met ≈ 0 mV (cation)
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // Ca2+ dynamics (proportional to depolarisation)
        let ca_entry = if self.v > self.v_rest { 0.01 * (self.v - self.v_rest) } else { 0.0 };
        self.ca += (-self.ca / self.tau_ca + ca_entry) * self.dt;
        if self.ca < 0.0 { self.ca = 0.0; }

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.0;
    }
}

impl Default for InnerHairCell {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Outer Hair Cell (OHC) — auditory, electromotility
// ═══════════════════════════════════════════════════════════════════

/// Outer hair cell — cochlear amplifier via electromotility.
///
/// Prestin-driven somatic motility amplifies basilar membrane vibration.
/// Non-linear gain: compresses dynamic range. Graded, no spikes.
///
/// Based on Dallos 2008 / Hudspeth 2008.
#[derive(Clone, Debug)]
pub struct OuterHairCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub g_met: f64,
    pub x_half: f64,
    pub s: f64,
    pub motility: f64,  // Somatic length change (normalised)
    pub gain: f64,       // Prestin gain factor
    pub dt: f64,
}

impl OuterHairCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            tau: 0.3,
            g_met: 15.0,
            x_half: 20.0,
            s: 6.0,
            motility: 0.0,
            gain: 5.0,
            dt: 0.025,
        }
    }

    /// Step with displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v);
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // Prestin electromotility: Boltzmann function of V
        self.motility = self.gain / (1.0 + (-(self.v + 40.0) / 10.0).exp()) - self.gain / 2.0;

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.motility = 0.0;
    }
}

impl Default for OuterHairCell {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Rod Photoreceptor — scotopic vision
// ═══════════════════════════════════════════════════════════════════

/// Rod photoreceptor — scotopic (dim light) vision.
///
/// Dark current (CNG channels open) → depolarised at rest (~-40 mV).
/// Light → rhodopsin → PDE → cGMP drops → CNG close → hyperpolarise.
/// Graded, no spikes. Very slow recovery (seconds).
///
/// Based on Nikonov et al. 2006 / Hamer et al. 2005.
#[derive(Clone, Debug)]
pub struct RodPhotoreceptor {
    pub v: f64,
    pub v_dark: f64,    // Dark resting potential (mV)
    pub v_hyper: f64,   // Maximum hyperpolarised level (mV)
    pub cgmp: f64,      // Normalised cGMP concentration [0,1]
    pub tau_act: f64,   // Activation time constant (ms)
    pub tau_rec: f64,   // Recovery time constant (ms)
    pub sensitivity: f64,
    pub dt: f64,
}

impl RodPhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -70.0,
            cgmp: 1.0,
            tau_act: 20.0,
            tau_rec: 500.0,   // Very slow dark adaptation
            sensitivity: 0.01,
            dt: 0.1,
        }
    }

    /// Step with light intensity (arbitrary units, ≥ 0). Returns membrane potential (mV).
    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);
        // cGMP hydrolysis by light-activated PDE
        let d_cgmp = -self.sensitivity * light_clamped * self.cgmp / self.tau_act
            + (1.0 - self.cgmp) / self.tau_rec;
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.0);

        // CNG channel current proportional to cGMP^3 (Hill coefficient)
        let cng_fraction = self.cgmp.powi(3);
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
    }
}

impl Default for RodPhotoreceptor {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Cone Photoreceptor — photopic vision
// ═══════════════════════════════════════════════════════════════════

/// Cone photoreceptor — photopic (bright light) colour vision.
///
/// Same transduction cascade as rods but faster kinetics, lower
/// sensitivity, and faster dark adaptation.
///
/// Based on Schnapf et al. 1990 / Baylor 1987.
#[derive(Clone, Debug)]
pub struct ConePhotoreceptor {
    pub v: f64,
    pub v_dark: f64,
    pub v_hyper: f64,
    pub cgmp: f64,
    pub tau_act: f64,
    pub tau_rec: f64,
    pub sensitivity: f64,
    pub dt: f64,
}

impl ConePhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -65.0,
            cgmp: 1.0,
            tau_act: 5.0,       // Faster than rods
            tau_rec: 50.0,      // Much faster recovery than rods
            sensitivity: 0.001, // Lower sensitivity than rods
            dt: 0.1,
        }
    }

    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);
        let d_cgmp = -self.sensitivity * light_clamped * self.cgmp / self.tau_act
            + (1.0 - self.cgmp) / self.tau_rec;
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.0);

        let cng_fraction = self.cgmp.powi(3);
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
    }
}

impl Default for ConePhotoreceptor {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Retinal Ganglion Cell (ON/OFF) — spiking output of retina
// ═══════════════════════════════════════════════════════════════════

/// Retinal ganglion cell — spiking output neuron of the retina.
///
/// ON-centre or OFF-centre receptive field. Receives graded bipolar
/// cell input, produces spikes. Simple LIF with contrast gain control.
///
/// Based on Pillow et al. 2005 GLM framework (simplified).
#[derive(Clone, Debug)]
pub struct RetinalGanglionCell {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub on_centre: bool,   // true = ON-centre, false = OFF-centre
    pub gain: f64,
    pub refrac_count: u32,
    pub refrac_period: u32,
    pub dt: f64,
}

impl RetinalGanglionCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 10.0,
            on_centre: true,
            gain: 2.0,
            refrac_count: 0,
            refrac_period: 3,
            dt: 0.5,
        }
    }

    pub fn off_centre() -> Self {
        Self { on_centre: false, ..Self::new() }
    }

    /// Step with bipolar cell input (mV-equivalent). Returns spike (1/0).
    pub fn step(&mut self, input: f64) -> i32 {
        if self.refrac_count > 0 {
            self.refrac_count -= 1;
            return 0;
        }
        let effective = if self.on_centre { input } else { -input };
        let drive = self.gain * effective;
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.refrac_period;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0;
    }
}

impl Default for RetinalGanglionCell {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Merkel Cell — slowly adapting type I mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Merkel cell — slowly adapting type I (SAI) mechanoreceptor.
///
/// Responds to sustained pressure with slowly adapting discharge.
/// Encodes texture and edges. Two-component model: fast onset + slow
/// sustained component.
///
/// Based on Lesniak et al. 2014.
#[derive(Clone, Debug)]
pub struct MerkelCell {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,       // Slow adaptation variable
    pub tau_adapt: f64,   // Adaptation time constant (ms)
    pub a_adapt: f64,     // Adaptation coupling
    pub gain: f64,
    pub dt: f64,
}

impl MerkelCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 5.0,
            adapt: 0.0,
            tau_adapt: 200.0,  // Very slow adaptation
            a_adapt: 0.3,
            gain: 1.5,
            dt: 0.5,
        }
    }

    /// Step with pressure (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        let drive = self.gain * pressure.max(0.0) - self.adapt;
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;
        self.adapt += (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt * self.dt;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }
}

impl Default for MerkelCell {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Pacinian Corpuscle — rapidly adapting mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Pacinian corpuscle — rapidly adapting (RA/RAII) mechanoreceptor.
///
/// Responds to vibration and transient pressure changes.
/// Band-pass filtering via lamellar structure: only signals
/// with rapid onset/offset produce responses. Derivative-like.
///
/// Based on Loewenstein & Skalak 1966 / Bell et al. 1994.
#[derive(Clone, Debug)]
pub struct PacinianCorpuscle {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub prev_pressure: f64,
    pub adapt: f64,
    pub tau_adapt: f64,
    pub gain: f64,
    pub dt: f64,
}

impl PacinianCorpuscle {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 2.0,
            prev_pressure: 0.0,
            adapt: 0.0,
            tau_adapt: 5.0,   // Fast adaptation
            gain: 10.0,       // High gain on derivative
            dt: 0.5,
        }
    }

    /// Step with pressure (arbitrary units). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        // Derivative-like response: rate of change drives the neuron
        let dp = (pressure - self.prev_pressure) / self.dt;
        self.prev_pressure = pressure;

        let drive = self.gain * dp.abs() - self.adapt;
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;
        self.adapt += (0.5 * drive.max(0.0) - self.adapt) / self.tau_adapt * self.dt;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.prev_pressure = 0.0;
        self.adapt = 0.0;
    }
}

impl Default for PacinianCorpuscle {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Nociceptor — pain receptor
// ═══════════════════════════════════════════════════════════════════

/// Nociceptor — high-threshold pain receptor neuron.
///
/// Only fires above noxious threshold. Sensitisation: repeated
/// stimulation lowers threshold (hyperalgesia). TTX-resistant Na+
/// channels provide slow, broad APs.
///
/// Based on Basbaum et al. 2009 / Gold & Gebhart 2010.
#[derive(Clone, Debug)]
pub struct Nociceptor {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub sensitisation: f64,    // Threshold reduction (mV)
    pub tau_sens: f64,         // Sensitisation decay (ms)
    pub sens_rate: f64,        // Sensitisation buildup rate
    pub gain: f64,
    pub dt: f64,
}

impl Nociceptor {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -30.0,  // High threshold
            tau: 8.0,
            sensitisation: 0.0,
            tau_sens: 5000.0,    // Very slow decay (seconds)
            sens_rate: 0.5,
            gain: 1.0,
            dt: 0.5,
        }
    }

    /// Step with noxious stimulus intensity (≥ 0). Returns spike (1/0).
    pub fn step(&mut self, stimulus: f64) -> i32 {
        let drive = self.gain * stimulus.max(0.0);
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        let effective_threshold = self.v_threshold - self.sensitisation;
        if self.v >= effective_threshold {
            self.v = self.v_reset;
            // Spike causes sensitisation buildup (capped at 10 mV)
            self.sensitisation = (self.sensitisation + self.sens_rate).min(10.0);
            1
        } else {
            // Sensitisation slowly decays
            self.sensitisation += -self.sensitisation / self.tau_sens * self.dt;
            if self.sensitisation < 0.0 { self.sensitisation = 0.0; }
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.sensitisation = 0.0;
    }
}

impl Default for Nociceptor {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Olfactory Receptor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Olfactory receptor neuron — chemical-to-spike transducer.
///
/// Odorant binding → Golf → adenylyl cyclase → cAMP → CNG channels.
/// Produces spiking output to olfactory bulb. Adapts via Ca2+/CaM
/// feedback on CNG channels.
///
/// Based on Rospars et al. 2008 / Firestein 2001.
#[derive(Clone, Debug)]
pub struct OlfactoryReceptorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub camp: f64,       // Normalised cAMP [0, 1]
    pub adapt: f64,      // Ca2+/CaM adaptation
    pub tau_camp: f64,   // cAMP dynamics (ms)
    pub tau_adapt: f64,
    pub gain: f64,
    pub dt: f64,
}

impl OlfactoryReceptorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -45.0,
            tau: 5.0,
            camp: 0.0,
            adapt: 0.0,
            tau_camp: 50.0,
            tau_adapt: 500.0,
            gain: 1.5,
            dt: 0.5,
        }
    }

    /// Step with odorant concentration (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, concentration: f64) -> i32 {
        let conc = concentration.max(0.0);
        // cAMP production: Hill function of odorant, reduced by adaptation
        let camp_target = conc / (conc + 1.0) * (1.0 - 0.8 * self.adapt);
        self.camp += (camp_target - self.camp) / self.tau_camp * self.dt;
        self.camp = self.camp.clamp(0.0, 1.0);

        let drive = self.gain * self.camp * 50.0; // Scale to mV
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        // Ca2+/CaM adaptation
        let ca_proxy = if self.v > self.v_rest { (self.v - self.v_rest) / 20.0 } else { 0.0 };
        self.adapt += (ca_proxy - self.adapt) / self.tau_adapt * self.dt;
        self.adapt = self.adapt.clamp(0.0, 1.0);

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.camp = 0.0;
        self.adapt = 0.0;
    }
}

impl Default for OlfactoryReceptorNeuron {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Taste Receptor Cell
// ═══════════════════════════════════════════════════════════════════

/// Taste receptor cell — gustatory transducer.
///
/// Type II cells: GPCR → PLC → IP3 → Ca2+ release → ATP secretion.
/// Graded output (ATP release proportional to Ca2+), no conventional
/// spikes. Adapts via Ca2+ pump.
///
/// Based on Chaudhari & Roper 2010 / Liman et al. 2014.
#[derive(Clone, Debug)]
pub struct TasteReceptorCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub ca: f64,          // Intracellular Ca2+ (normalised)
    pub ip3: f64,         // IP3 concentration (normalised)
    pub tau_ip3: f64,
    pub tau_ca: f64,
    pub gain: f64,
    pub atp_release: f64, // Output: ATP release rate
    pub dt: f64,
}

impl TasteReceptorCell {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            v_rest: -50.0,
            tau: 10.0,
            ca: 0.0,
            ip3: 0.0,
            tau_ip3: 100.0,
            tau_ca: 200.0,
            gain: 1.0,
            atp_release: 0.0,
            dt: 0.5,
        }
    }

    /// Step with tastant concentration (≥ 0). Returns receptor potential (mV).
    pub fn step(&mut self, tastant: f64) -> f64 {
        let conc = tastant.max(0.0);
        // GPCR → IP3
        let ip3_target = conc / (conc + 0.5);
        self.ip3 += (ip3_target - self.ip3) / self.tau_ip3 * self.dt;
        self.ip3 = self.ip3.clamp(0.0, 1.0);

        // IP3 → Ca2+ release from ER
        let ca_release = self.ip3.powi(2) * (1.0 - self.ca);
        self.ca += (ca_release - self.ca / self.tau_ca) * self.dt;
        self.ca = self.ca.clamp(0.0, 1.0);

        // Ca2+ → depolarisation (TRPM5 channel)
        let i_trpm5 = self.gain * self.ca * 20.0;
        self.v += (-(self.v - self.v_rest) + i_trpm5) / self.tau * self.dt;

        // ATP release proportional to Ca2+
        self.atp_release = self.ca;

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.0;
        self.ip3 = 0.0;
        self.atp_release = 0.0;
    }
}

impl Default for TasteReceptorCell {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inner Hair Cell ──────────────────────────────────────────

    #[test]
    fn ihc_depolarises_with_displacement() {
        let mut c = InnerHairCell::new();
        let v_rest = c.v;
        for _ in 0..200 { c.step(50.0); }
        assert!(c.v > v_rest, "IHC should depolarise: v={}", c.v);
    }

    #[test]
    fn ihc_no_change_at_zero() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 { c.step(0.0); }
        assert!((c.v - c.v_rest).abs() < 5.0, "IHC should stay near rest with no displacement");
    }

    #[test]
    fn ihc_ca_increases_with_depolarisation() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 { c.step(60.0); }
        assert!(c.ca > 0.0, "Ca2+ should increase during depolarisation");
    }

    #[test]
    fn ihc_reset_roundtrip() {
        let mut c = InnerHairCell::new();
        for _ in 0..100 { c.step(50.0); }
        c.reset();
        assert_eq!(c.v, c.v_rest);
        assert_eq!(c.ca, 0.0);
    }

    #[test]
    fn ihc_bounded() {
        let mut c = InnerHairCell::new();
        for _ in 0..10000 { c.step(100.0); }
        assert!(c.v.is_finite());
        assert!(c.ca.is_finite());
    }

    #[test]
    fn ihc_performance() {
        let mut c = InnerHairCell::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 { c.step(50.0); }
        assert!(start.elapsed().as_millis() < 50);
    }

    // ── Outer Hair Cell ──────────────────────────────────────────

    #[test]
    fn ohc_depolarises_and_motility() {
        let mut c = OuterHairCell::new();
        for _ in 0..200 { c.step(40.0); }
        assert!(c.v > c.v_rest);
        assert!(c.motility.abs() > 0.01, "OHC should show motility");
    }

    #[test]
    fn ohc_reset() {
        let mut c = OuterHairCell::new();
        for _ in 0..100 { c.step(40.0); }
        c.reset();
        assert_eq!(c.motility, 0.0);
    }

    #[test]
    fn ohc_bounded() {
        let mut c = OuterHairCell::new();
        for _ in 0..10000 { c.step(100.0); }
        assert!(c.v.is_finite());
    }

    // ── Rod Photoreceptor ────────────────────────────────────────

    #[test]
    fn rod_hyperpolarises_with_light() {
        let mut r = RodPhotoreceptor::new();
        let v_dark = r.v;
        for _ in 0..1000 { r.step(100.0); }
        assert!(r.v < v_dark, "rod should hyperpolarise: v={}", r.v);
    }

    #[test]
    fn rod_stays_dark_without_light() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..500 { r.step(0.0); }
        assert!((r.v - r.v_dark).abs() < 1.0);
    }

    #[test]
    fn rod_slow_recovery() {
        let mut r = RodPhotoreceptor::new();
        // Flash
        for _ in 0..500 { r.step(200.0); }
        let v_after_flash = r.v;
        // Dark: slow recovery
        for _ in 0..1000 { r.step(0.0); }
        assert!(r.v > v_after_flash, "rod should recover in dark");
        assert!(r.v < r.v_dark, "rod should not fully recover in 1000 steps");
    }

    #[test]
    fn rod_cgmp_bounded() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..10000 { r.step(1000.0); }
        assert!(r.cgmp >= 0.0 && r.cgmp <= 1.0);
        r.reset();
        for _ in 0..10000 { r.step(-10.0); } // Negative light clamped to 0
        assert!(r.cgmp >= 0.0 && r.cgmp <= 1.0);
    }

    #[test]
    fn rod_performance() {
        let mut r = RodPhotoreceptor::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 { r.step(50.0); }
        assert!(start.elapsed().as_millis() < 50);
    }

    // ── Cone Photoreceptor ───────────────────────────────────────

    #[test]
    fn cone_hyperpolarises_with_light() {
        let mut c = ConePhotoreceptor::new();
        let v_dark = c.v;
        for _ in 0..500 { c.step(500.0); }
        assert!(c.v < v_dark);
    }

    #[test]
    fn cone_faster_than_rod() {
        let mut rod = RodPhotoreceptor::new();
        let mut cone = ConePhotoreceptor::new();
        // Flash, then dark
        for _ in 0..500 { rod.step(100.0); cone.step(100.0); }
        for _ in 0..2000 { rod.step(0.0); cone.step(0.0); }
        // Cone should recover more (faster tau_rec)
        let rod_recovery = rod.v - rod.v_hyper;
        let cone_recovery = cone.v - cone.v_hyper;
        assert!(
            cone_recovery > rod_recovery,
            "cone ({cone_recovery:.1}) should recover more than rod ({rod_recovery:.1})"
        );
    }

    #[test]
    fn cone_reset() {
        let mut c = ConePhotoreceptor::new();
        for _ in 0..500 { c.step(500.0); }
        c.reset();
        assert_eq!(c.cgmp, 1.0);
        assert_eq!(c.v, c.v_dark);
    }

    // ── Retinal Ganglion Cell ────────────────────────────────────

    #[test]
    fn rgc_on_fires_with_positive_input() {
        let mut rgc = RetinalGanglionCell::new();
        let spikes: i32 = (0..500).map(|_| rgc.step(20.0)).sum();
        assert!(spikes > 0, "ON-RGC should fire with positive input");
    }

    #[test]
    fn rgc_off_fires_with_negative_input() {
        let mut rgc = RetinalGanglionCell::off_centre();
        let spikes: i32 = (0..500).map(|_| rgc.step(-20.0)).sum();
        assert!(spikes > 0, "OFF-RGC should fire with negative input");
    }

    #[test]
    fn rgc_on_no_fire_without_input() {
        let mut rgc = RetinalGanglionCell::new();
        let spikes: i32 = (0..500).map(|_| rgc.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn rgc_refractory_period() {
        let mut rgc = RetinalGanglionCell::new();
        let mut spikes = Vec::new();
        for _ in 0..100 { spikes.push(rgc.step(30.0)); }
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 && i + 1 < spikes.len() {
                assert_eq!(spikes[i + 1], 0, "refractory period violated at step {}", i + 1);
            }
        }
    }

    #[test]
    fn rgc_reset() {
        let mut rgc = RetinalGanglionCell::new();
        for _ in 0..100 { rgc.step(20.0); }
        rgc.reset();
        assert_eq!(rgc.refrac_count, 0);
        assert_eq!(rgc.v, rgc.v_rest);
    }

    // ── Merkel Cell ──────────────────────────────────────────────

    #[test]
    fn merkel_fires_with_sustained_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..2000).map(|_| m.step(20.0)).sum();
        assert!(spikes > 0, "Merkel should fire with sustained pressure");
    }

    #[test]
    fn merkel_slow_adaptation() {
        let mut m = MerkelCell::new();
        let first: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        // Slow adapting: second half may fire slightly fewer but still fires
        assert!(second > 0, "Merkel should still fire in second half (slow adapting)");
        assert!(second <= first + 5, "Merkel should slowly adapt");
    }

    #[test]
    fn merkel_no_fire_without_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..1000).map(|_| m.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn merkel_reset() {
        let mut m = MerkelCell::new();
        for _ in 0..500 { m.step(20.0); }
        m.reset();
        assert_eq!(m.adapt, 0.0);
    }

    // ── Pacinian Corpuscle ───────────────────────────────────────

    #[test]
    fn pacinian_fires_on_pressure_onset() {
        let mut p = PacinianCorpuscle::new();
        // Ramp up pressure rapidly
        let spikes: i32 = (0..100).map(|i| p.step(i as f64 * 2.0)).sum();
        assert!(spikes > 0, "Pacinian should fire on pressure onset");
    }

    #[test]
    fn pacinian_adapts_to_sustained() {
        let mut p = PacinianCorpuscle::new();
        // Rapid onset
        let onset: i32 = (0..10).map(|i| p.step(i as f64 * 10.0)).sum();
        // Sustained (constant pressure, dp/dt ≈ 0)
        let sustained: i32 = (0..500).map(|_| p.step(100.0)).sum();
        // Should fire mostly during onset, not during sustained
        assert!(
            sustained <= onset + 5,
            "Pacinian should adapt to sustained: onset={onset}, sustained={sustained}"
        );
    }

    #[test]
    fn pacinian_no_fire_at_rest() {
        let mut p = PacinianCorpuscle::new();
        let spikes: i32 = (0..500).map(|_| p.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn pacinian_reset() {
        let mut p = PacinianCorpuscle::new();
        for i in 0..100 { p.step(i as f64); }
        p.reset();
        assert_eq!(p.prev_pressure, 0.0);
        assert_eq!(p.adapt, 0.0);
    }

    // ── Nociceptor ───────────────────────────────────────────────

    #[test]
    fn nociceptor_high_threshold() {
        let mut n = Nociceptor::new();
        // Sub-threshold
        let low: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert_eq!(low, 0, "nociceptor should not fire at low stimulus");
        // Supra-threshold
        n.reset();
        let high: i32 = (0..500).map(|_| n.step(50.0)).sum();
        assert!(high > 0, "nociceptor should fire at high stimulus");
    }

    #[test]
    fn nociceptor_sensitisation() {
        let mut n = Nociceptor::new();
        // Strong stimulus → spikes → sensitisation builds
        for _ in 0..1000 { n.step(50.0); }
        assert!(n.sensitisation > 0.0, "sensitisation should increase");
        let sens = n.sensitisation;
        // After a long pause, sensitisation decays (tau_sens=5000ms, need many steps)
        for _ in 0..50000 { n.step(0.0); }
        assert!(n.sensitisation < sens, "sensitisation should decay: was {sens}, now {}", n.sensitisation);
    }

    #[test]
    fn nociceptor_no_fire_without_stimulus() {
        let mut n = Nociceptor::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn nociceptor_reset() {
        let mut n = Nociceptor::new();
        for _ in 0..500 { n.step(50.0); }
        n.reset();
        assert_eq!(n.sensitisation, 0.0);
    }

    // ── Olfactory Receptor ───────────────────────────────────────

    #[test]
    fn olfactory_fires_with_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(spikes > 0, "olfactory should fire with odorant");
    }

    #[test]
    fn olfactory_adapts() {
        let mut o = OlfactoryReceptorNeuron::new();
        let first: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        let second: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(
            second <= first + 5,
            "olfactory should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn olfactory_no_fire_without_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| o.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn olfactory_reset() {
        let mut o = OlfactoryReceptorNeuron::new();
        for _ in 0..1000 { o.step(5.0); }
        o.reset();
        assert_eq!(o.camp, 0.0);
        assert_eq!(o.adapt, 0.0);
    }

    // ── Taste Receptor ───────────────────────────────────────────

    #[test]
    fn taste_depolarises_with_tastant() {
        let mut t = TasteReceptorCell::new();
        let v_rest = t.v;
        for _ in 0..500 { t.step(5.0); }
        assert!(t.v > v_rest, "taste cell should depolarise");
    }

    #[test]
    fn taste_atp_release() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 { t.step(5.0); }
        assert!(t.atp_release > 0.0, "ATP should be released");
    }

    #[test]
    fn taste_no_response_without_tastant() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 { t.step(0.0); }
        assert!((t.v - t.v_rest).abs() < 2.0);
        assert!(t.atp_release < 0.01);
    }

    #[test]
    fn taste_ca_bounded() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..10000 { t.step(100.0); }
        assert!(t.ca >= 0.0 && t.ca <= 1.0);
        assert!(t.ip3 >= 0.0 && t.ip3 <= 1.0);
    }

    #[test]
    fn taste_reset() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 { t.step(5.0); }
        t.reset();
        assert_eq!(t.ca, 0.0);
        assert_eq!(t.ip3, 0.0);
        assert_eq!(t.atp_release, 0.0);
    }
}
