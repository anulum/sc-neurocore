#[derive(Debug, Clone, Copy)]
pub struct BoothRinzelState {
    pub vs: f64,
    pub vd: f64,
    pub h: f64,
    pub n: f64,
    pub q: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_ca: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub g_c: f64,
    pub p: f64,
    pub c_m: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub alpha_ca: f64,
    pub k_ca: f64,
    pub f_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl Default for BoothRinzelState {
    fn default() -> Self {
        Self {
            vs: -60.0,
            vd: -60.0,
            h: 0.6,
            n: 0.1,
            q: 0.1,
            ca: 0.1,
            g_na: 120.0,
            g_k: 36.0,
            g_ca: 2.0,
            g_kca: 5.0,
            g_l: 0.3,
            g_c: 1.0,
            p: 0.5,
            c_m: 1.0,
            e_na: 50.0,
            e_k: -77.0,
            e_ca: 120.0,
            e_l: -54.4,
            alpha_ca: 0.002,
            k_ca: 0.01,
            f_ca: 0.01,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }
}

fn gate(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn safe_exp(x: f64) -> f64 {
    clip(x, -100.0, 100.0).exp()
}

fn valid_config(s: &BoothRinzelState) -> bool {
    [
        s.g_na, s.g_k, s.g_ca, s.g_kca, s.g_l, s.g_c, s.c_m, s.alpha_ca, s.k_ca, s.f_ca, s.dt,
    ]
    .iter()
    .all(|x| x.is_finite() && *x > 0.0)
        && s.p.is_finite()
        && s.p > 0.0
        && s.p < 1.0
        && [s.e_na, s.e_k, s.e_ca, s.e_l, s.v_threshold]
            .iter()
            .all(|x| x.is_finite())
}

fn valid_state(vs: f64, vd: f64, h: f64, n: f64, q: f64, ca: f64) -> bool {
    vs.is_finite()
        && vd.is_finite()
        && ca.is_finite()
        && ca >= 0.0
        && gate(h)
        && gate(n)
        && gate(q)
        && (-200.0..=100.0).contains(&vs)
        && (-200.0..=100.0).contains(&vd)
}

fn substep(
    s: &BoothRinzelState,
    mut vs: f64,
    mut vd: f64,
    mut h: f64,
    mut n: f64,
    mut q: f64,
    mut ca: f64,
    current: f64,
    dt: f64,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    if !current.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return None;
    }
    let m_inf = 1.0 / (1.0 + safe_exp(-(vs + 30.0) / 9.5));
    let h_inf = 1.0 / (1.0 + safe_exp((vs + 53.0) / 7.0));
    let n_inf = 1.0 / (1.0 + safe_exp(-(vs + 30.0) / 10.0));
    let q_inf = 1.0 / (1.0 + safe_exp(-(vd + 25.0) / 5.0));

    let tau_h = 1.0 + 7.0 / (safe_exp((vs + 40.0) / 5.0) + safe_exp(-(vs + 40.0) / 5.0));
    let tau_n = 1.0 + 5.0 / (safe_exp((vs + 35.0) / 10.0) + safe_exp(-(vs + 35.0) / 10.0));
    let tau_q = 10.0;

    h = clip(h + dt * (h_inf - h) / tau_h, 0.0, 1.0);
    n = clip(n + dt * (n_inf - n) / tau_n, 0.0, 1.0);
    q = clip(q + dt * (q_inf - q) / tau_q, 0.0, 1.0);

    let i_na = s.g_na * m_inf.powi(3) * h * (vs - s.e_na);
    let i_k = s.g_k * n.powi(4) * (vs - s.e_k);
    let i_l = s.g_l * (vs - s.e_l);
    let i_c = s.g_c * (vs - vd);
    let i_ca = s.g_ca * q.powi(2) * (vd - s.e_ca);
    let i_kca = s.g_kca * (ca / (ca + s.k_ca)) * (vd - s.e_k);

    let d_vs = (current - i_na - i_k - i_l - i_c) / (s.c_m * s.p);
    let d_vd = (-i_ca - i_kca - i_l + i_c) / (s.c_m * (1.0 - s.p));
    let d_ca = -s.alpha_ca * i_ca - s.f_ca * ca;

    vs = clip(vs + dt * d_vs, -200.0, 100.0);
    vd = clip(vd + dt * d_vd, -200.0, 100.0);
    ca = (ca + dt * d_ca).max(0.0);

    valid_state(vs, vd, h, n, q, ca).then_some((vs, vd, h, n, q, ca))
}

pub fn validate_booth_rinzel(s: &BoothRinzelState) -> bool {
    valid_config(s) && valid_state(s.vs, s.vd, s.h, s.n, s.q, s.ca)
}

pub fn booth_rinzel_step(s: &mut BoothRinzelState, current: f64) -> i32 {
    if !validate_booth_rinzel(s) || !current.is_finite() {
        return -1;
    }
    let old_vs = s.vs;
    let (mut vs, mut vd, mut h, mut n, mut q, mut ca) = (s.vs, s.vd, s.h, s.n, s.q, s.ca);
    let dt = s.dt / 4.0;
    for _ in 0..4 {
        match substep(s, vs, vd, h, n, q, ca, current, dt) {
            Some(next) => (vs, vd, h, n, q, ca) = next,
            None => return -1,
        }
    }
    (s.vs, s.vd, s.h, s.n, s.q, s.ca) = (vs, vd, h, n, q, ca);
    if old_vs < s.v_threshold && s.vs >= s.v_threshold {
        1
    } else {
        0
    }
}

pub fn reset_booth_rinzel(s: &mut BoothRinzelState) {
    s.vs = -60.0;
    s.vd = -60.0;
    s.h = 0.6;
    s.n = 0.1;
    s.q = 0.1;
    s.ca = 0.1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn booth_rinzel_rejects_invalid_probability_without_mutation() {
        let mut state = BoothRinzelState {
            p: 1.0,
            ..Default::default()
        };
        let before = state;
        assert_eq!(booth_rinzel_step(&mut state, 10.0), -1);
        assert_eq!(state.vs, before.vs);
        assert_eq!(state.vd, before.vd);
        assert_eq!(state.h, before.h);
        assert_eq!(state.n, before.n);
        assert_eq!(state.q, before.q);
        assert_eq!(state.ca, before.ca);
    }

    #[test]
    fn booth_rinzel_keeps_state_physical_under_drive() {
        let mut state = BoothRinzelState::default();
        for _ in 0..100 {
            assert!(booth_rinzel_step(&mut state, 8.0) >= 0);
            assert!(validate_booth_rinzel(&state));
        }
    }
}
