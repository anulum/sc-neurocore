// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — RK4 neuron integrator ports

//! Explicit RK4 ports for the priority neuron integrator paths.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const IZH_SPIKE_THRESHOLD: f64 = 30.0;

#[derive(Clone, Debug)]
pub struct IzhikevichRk4 {
    pub v: f64,
    pub u: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub dt: f64,
}

impl IzhikevichRk4 {
    pub fn new(dt: f64) -> Self {
        let c = -65.0;
        let b = 0.2;
        Self {
            v: c,
            u: b * c,
            a: 0.02,
            b,
            c,
            d: 8.0,
            dt,
        }
    }

    fn rhs(&self, v: f64, u: f64, current: f64) -> (f64, f64) {
        let dv = 0.04 * v.powi(2) + 5.0 * v + 140.0 - u + current;
        let du = self.a * (self.b * v - u);
        (dv, du)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let (k1_v, k1_u) = self.rhs(self.v, self.u, current);
        let (k2_v, k2_u) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.u + 0.5 * self.dt * k1_u,
            current,
        );
        let (k3_v, k3_u) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.u + 0.5 * self.dt * k2_u,
            current,
        );
        let (k4_v, k4_u) = self.rhs(self.v + self.dt * k3_v, self.u + self.dt * k3_u, current);

        self.v += (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        self.u += (self.dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u);

        if self.v >= IZH_SPIKE_THRESHOLD {
            self.v = self.c;
            self.u += self.d;
            1
        } else {
            0
        }
    }
}

/// Izhikevich 2007 biophysical parameterisation (NeuroML `izhikevich2007Cell`):
/// `C dv/dt = k (v - vr)(v - vt) - u + I`, `du/dt = a (b (v - vr) - u)`, with a
/// `v >= vpeak -> v = c, u += d` reset. RK4 over the coupled ODE. The right-hand
/// side is exact arithmetic (products, a sum, a division — no transcendental
/// functions), so `simulate` matches the Python reference bit-for-bit.
#[derive(Clone, Debug)]
pub struct Izhikevich2007Rk4 {
    pub v: f64,
    pub u: f64,
    pub cap: f64,
    pub k: f64,
    pub vr: f64,
    pub vt: f64,
    pub vpeak: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub dt: f64,
}

impl Izhikevich2007Rk4 {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            u: 0.0,
            cap: 100.0,
            k: 0.7,
            vr: -60.0,
            vt: -40.0,
            vpeak: 35.0,
            a: 0.03,
            b: -2.0,
            c: -50.0,
            d: 100.0,
            dt: 0.1,
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.v, self.u, self.cap, self.k, self.vr, self.vt, self.vpeak, self.a, self.b, self.c,
            self.d, self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.cap > 0.0
            && self.dt > 0.0
    }

    fn rhs(&self, v: f64, u: f64, current: f64) -> (f64, f64) {
        let dv = (self.k * (v - self.vr) * (v - self.vt) - u + current) / self.cap;
        let du = self.a * (self.b * (v - self.vr) - u);
        (dv, du)
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.is_valid() || !current.is_finite() {
            return Err("invalid Izhikevich 2007 runtime state or current");
        }
        let (k1v, k1u) = self.rhs(self.v, self.u, current);
        let (k2v, k2u) = self.rhs(
            self.v + 0.5 * self.dt * k1v,
            self.u + 0.5 * self.dt * k1u,
            current,
        );
        let (k3v, k3u) = self.rhs(
            self.v + 0.5 * self.dt * k2v,
            self.u + 0.5 * self.dt * k2u,
            current,
        );
        let (k4v, k4u) = self.rhs(self.v + self.dt * k3v, self.u + self.dt * k3u, current);
        let dt6 = self.dt / 6.0;
        let mut v_next = self.v + dt6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        let mut u_next = self.u + dt6 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let event = if v_next >= self.vpeak {
            v_next = self.c;
            u_next += self.d;
            1
        } else {
            0
        };
        if !v_next.is_finite() || !u_next.is_finite() {
            return Err("Izhikevich 2007 candidate state became non-finite");
        }
        self.v = v_next;
        self.u = u_next;
        Ok(event)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// (already reset to `c` on spiking steps) and the spike count. Reuses
    /// `step`, so the trace is bit-identical to the per-step path and to the
    /// Python reference (the right-hand side is exact arithmetic). The final
    /// state is left in `self.v` / `self.u`.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.try_step(current)?;
            trace.push(self.v);
            if spiked == 1 {
                spikes += 1;
            }
        }
        Ok((trace, spikes))
    }

    pub fn try_reset(&mut self) -> Result<(), &'static str> {
        if !self.is_valid() {
            return Err("invalid Izhikevich 2007 parameters for reset");
        }
        let v_next = self.vr;
        let u_next = self.b * (v_next - self.vr);
        if !v_next.is_finite() || !u_next.is_finite() {
            return Err("Izhikevich 2007 reset state became non-finite");
        }
        self.v = v_next;
        self.u = u_next;
        Ok(())
    }

    pub fn reset(&mut self) {
        let _ = self.try_reset();
    }
}

impl Default for Izhikevich2007Rk4 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct AdExRk4 {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl AdExRk4 {
    pub fn new(dt: f64) -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            v_rh: -55.0,
            delta_t: 2.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            c_m: 200.0,
            dt,
        }
    }

    fn rhs(&self, v: f64, w: f64, current: f64) -> (f64, f64) {
        let exp_arg = ((v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(v - self.v_rest) + exp_term) / self.tau + (-w + current) / self.c_m;
        let dw = (self.a * (v - self.v_rest) - w) / self.tau_w;
        (dv, dw)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let (k1_v, k1_w) = self.rhs(self.v, self.w, current);
        let (k2_v, k2_w) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        );
        let (k3_v, k3_w) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        );
        let (k4_v, k4_w) = self.rhs(self.v + self.dt * k3_v, self.w + self.dt * k3_w, current);

        self.v += (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        self.w += (self.dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w);

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
pub struct HodgkinHuxleyRk4 {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub c_m: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HodgkinHuxleyRk4 {
    pub fn new(dt: f64) -> Self {
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            dt,
            v_threshold: 0.0,
        }
    }

    fn alpha_m(v: f64) -> f64 {
        let d = v + 40.0;
        if d.abs() < 1e-7 {
            1.0
        } else {
            0.1 * d / (1.0 - (-d / 10.0).exp())
        }
    }

    fn beta_m(v: f64) -> f64 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    fn alpha_h(v: f64) -> f64 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    fn beta_h(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    fn alpha_n(v: f64) -> f64 {
        let d = v + 55.0;
        if d.abs() < 1e-7 {
            0.1
        } else {
            0.01 * d / (1.0 - (-d / 10.0).exp())
        }
    }

    fn beta_n(v: f64) -> f64 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }

    fn rhs(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        let [v, m, h, n] = state;
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);

        let dm = am * (1.0 - m) - bm * m;
        let dh = ah * (1.0 - h) - bh * h;
        let dn = an * (1.0 - n) - bn * n;
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_l + current) / self.c_m;
        [dv, dm, dh, dn]
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut state = [self.v, self.m, self.h, self.n];
        let substeps = (1.0 / self.dt).round() as usize;
        for _ in 0..substeps {
            let k1 = self.rhs(state, current);
            let k2 = self.rhs(add_scaled(state, k1, 0.5 * self.dt), current);
            let k3 = self.rhs(add_scaled(state, k2, 0.5 * self.dt), current);
            let k4 = self.rhs(add_scaled(state, k3, self.dt), current);
            for idx in 0..4 {
                state[idx] += (self.dt / 6.0) * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
            }
        }
        self.v = state[0];
        self.m = state[1];
        self.h = state[2];
        self.n = state[3];

        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
}

fn add_scaled(state: [f64; 4], deriv: [f64; 4], scale: f64) -> [f64; 4] {
    [
        state[0] + scale * deriv[0],
        state[1] + scale * deriv[1],
        state[2] + scale * deriv[2],
        state[3] + scale * deriv[3],
    ]
}

#[pyfunction]
#[pyo3(signature = (model_name, current_trace, dt=None))]
pub fn py_rk4_neuron_simulate<'py>(
    py: Python<'py>,
    model_name: &str,
    current_trace: PyReadonlyArray1<'py, f64>,
    dt: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let currents = current_trace.as_slice()?;
    match normalise_model_name(model_name).as_str() {
        "izhikevich" | "scizhikevichneuron" | "izhikevichneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(1.0))?;
            simulate_izhikevich(py, currents, dt)
        }
        "hodgkinhuxley" | "hodgkinhuxleyneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(0.01))?;
            simulate_hodgkin_huxley(py, currents, dt)
        }
        "adex" | "adexneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(0.1))?;
            simulate_adex(py, currents, dt)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported RK4 neuron model {model_name:?}"
        ))),
    }
}

fn validate_trace_dt(currents: &[f64], dt: f64) -> PyResult<f64> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be a positive finite scalar"));
    }
    if currents.is_empty() {
        return Err(PyValueError::new_err("current_trace must be non-empty"));
    }
    if currents.iter().any(|current| !current.is_finite()) {
        return Err(PyValueError::new_err(
            "current_trace must contain only finite values",
        ));
    }
    Ok(dt)
}

fn normalise_model_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn simulate_izhikevich<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = IzhikevichRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut u = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        u.push(neuron.u);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("u", u.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

fn simulate_adex<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = AdExRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut w = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        w.push(neuron.w);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("w", w.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

fn simulate_hodgkin_huxley<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = HodgkinHuxleyRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut m = Vec::with_capacity(currents.len());
    let mut h = Vec::with_capacity(currents.len());
    let mut n = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        m.push(neuron.m);
        h.push(neuron.h);
        n.push(neuron.n);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("m", m.into_pyarray(py))?;
    d.set_item("h", h.into_pyarray(py))?;
    d.set_item("n", n.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_rk4_is_deterministic_and_spikes() {
        let mut a = IzhikevichRk4::new(1.0);
        let mut b = IzhikevichRk4::new(1.0);
        let mut spikes = 0;
        for _ in 0..100 {
            spikes += a.step(10.0);
            b.step(10.0);
        }
        assert!(spikes > 0);
        assert_eq!(a.v, b.v);
        assert_eq!(a.u, b.u);
    }

    #[test]
    fn izhikevich2007_checked_trace_and_reset_are_fail_closed() {
        let mut neuron = Izhikevich2007Rk4::new();
        let (trace, events) = neuron.simulate(2_000, 100.0).unwrap();
        assert_eq!(trace.len(), 2_000);
        assert_eq!(events, 3);
        assert_eq!(trace.last().copied(), Some(neuron.v));

        let before = (neuron.v, neuron.u);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.v, neuron.u), before);

        neuron.vr = f64::NAN;
        assert!(neuron.try_reset().is_err());
        assert_eq!((neuron.v, neuron.u), before);
    }

    #[test]
    fn adex_rk4_remains_finite_under_sustained_current() {
        let mut neuron = AdExRk4::new(0.1);
        let mut spikes = 0;
        for _ in 0..3000 {
            spikes += neuron.step(500.0);
        }
        assert!(spikes > 0);
        assert!(neuron.v.is_finite());
        assert!(neuron.w.is_finite());
    }

    #[test]
    fn hodgkin_huxley_rk4_keeps_gates_bounded() {
        let mut neuron = HodgkinHuxleyRk4::new(0.01);
        let mut spikes = 0;
        for _ in 0..1000 {
            spikes += neuron.step(10.0);
        }
        assert!(spikes > 0);
        assert!(neuron.v.is_finite());
        assert!((0.0..=1.0).contains(&neuron.m));
        assert!((0.0..=1.0).contains(&neuron.h));
        assert!((0.0..=1.0).contains(&neuron.n));
    }

    #[test]
    fn model_name_normalisation_accepts_common_aliases() {
        assert_eq!(
            normalise_model_name("Hodgkin-HuxleyNeuron"),
            "hodgkinhuxleyneuron"
        );
        assert_eq!(normalise_model_name("AdEx"), "adex");
    }
}
