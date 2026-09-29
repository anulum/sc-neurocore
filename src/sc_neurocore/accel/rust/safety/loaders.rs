// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety acceleration for loaders

#[path = "nmnist.rs"]
pub mod nmnist;

pub fn _synthetic_event_dataset(
    _n_samples: f64,
    _spatial_size: f64,
    _n_classes: f64,
    _t: f64,
    _dt_ms: f64,
    _seed: f64,
) -> f64 {
    // n_samples: int,
    // spatial_size: int,
    // n_classes: int,
    // T: int,
    // dt_ms: float,
    // seed: int,
    // ) -> tuple[list[ndarray], ndarray] {
    // rng = random.default_rng(seed)
    // labels = rng.integers(0, n_classes, size=n_samples)
    // templates = rng.uniform(0.0, 0.3, size=(n_classes, spatial_size, spati
    // samples: list[ndarray] = []
    // for i in range(n_samples) {
    // rates = templates[labels[i]].ravel()
    // spike_train = poisson_encode(rates, T, dt_ms=dt_ms, seed=seed + i + 1)
    // timesteps, neuron_ids = nonzero(spike_train)
    // y_coords, x_coords = divmod(neuron_ids, spatial_size)
    // polarities = rng.integers(0, 2, size=len(timesteps))
    // events = column_stack(
    // [
    // x_coords,
    0.0
}

pub fn _check_root(_root: f64, _dataset_name: f64, _url: f64) -> f64 {
    // p = Path(root)
    // if p.exists() {
    // return p
    // raise FileNotFoundError(f"{dataset_name}: download from {url} into {p}
    0.0
}

pub fn load_nmnist(
    _root: f64,
    _train: f64,
    _dt_ms: f64,
    _t: f64,
    _synthetic: f64,
    _n_samples: f64,
) -> f64 {
    // root: str | Path = "data/nmnist",
    // train: bool = true,
    // dt_ms: float = 1.0,
    // T: int = 300,
    // synthetic: bool = false,
    // n_samples: int = 100,
    // seed: int = 42,
    // ) -> tuple[list[ndarray], ndarray] {
    // if synthetic {
    // return _synthetic_event_dataset(
    // n_samples,
    // _NMNIST_RES,
    // 10,
    // T,
    // dt_ms,
    // seed,
    // )
    // _check_root(root, "N-MNIST", _NMNIST_URL)
    // split_dir = Path(root) / ("Train" if train else "Test")
    // if not split_dir.exists() {
    0.0
}

/// Read real N-MNIST events as row-major float64 millisecond records.
pub fn _parse_nmnist_bin(path: &std::path::Path) -> std::io::Result<Vec<f64>> {
    nmnist::read_nmnist(path)
}

pub fn load_shd(
    _root: f64,
    _train: f64,
    _dt_ms: f64,
    _t: f64,
    _synthetic: f64,
    _n_samples: f64,
) -> f64 {
    // root: str | Path = "data/shd",
    // train: bool = true,
    // dt_ms: float = 1.0,
    // T: int = 1000,
    // synthetic: bool = false,
    // n_samples: int = 100,
    // seed: int = 42,
    // ) -> tuple[list[ndarray], ndarray] {
    // if synthetic {
    // return _synthetic_shd(n_samples, T, dt_ms, seed)
    // _check_root(root, "SHD", _SHD_URL)
    // fname = "shd_train.h5" if train else "shd_test.h5"
    // h5_path = Path(root) / fname
    // if not h5_path.exists() {
    // raise FileNotFoundError(f"{h5_path.resolve()} not found. Download from
    // import h5py
    // samples: list[ndarray] = []
    // with h5py.File(h5_path, "r") as f {
    // spike_times = f["spikes"]["times"]
    // spike_units = f["spikes"]["units"]
    0.0
}

pub fn _synthetic_shd(_n_samples: f64, _t: f64, _dt_ms: f64, _seed: f64) -> f64 {
    // n_samples: int,
    // T: int,
    // dt_ms: float,
    // seed: int,
    // ) -> tuple[list[ndarray], ndarray] {
    // rng = random.default_rng(seed)
    // labels = rng.integers(0, 20, size=n_samples)
    // templates = rng.uniform(0.0, 0.1, size=(20, _SHD_CHANNELS))
    // samples: list[ndarray] = []
    // for i in range(n_samples) {
    // spike_train = poisson_encode(
    // templates[labels[i]],
    // T,
    // dt_ms=dt_ms,
    // seed=seed + i + 1,
    // )
    // samples.append(spike_train)
    // return samples, labels
    0.0
}

pub fn load_dvs_cifar10(
    _root: f64,
    _train: f64,
    _dt_ms: f64,
    _t: f64,
    _synthetic: f64,
    _n_samples: f64,
) -> f64 {
    // root: str | Path = "data/dvs_cifar10",
    // train: bool = true,
    // dt_ms: float = 1.0,
    // T: int = 300,
    // synthetic: bool = false,
    // n_samples: int = 100,
    // seed: int = 42,
    // ) -> tuple[list[ndarray], ndarray] {
    // if synthetic {
    // return _synthetic_event_dataset(
    // n_samples,
    // _DVS_CIFAR10_RES,
    // 10,
    // T,
    // dt_ms,
    // seed,
    // )
    // _check_root(root, "DVS-CIFAR10", _DVS_CIFAR10_URL)
    // split_dir = Path(root) / ("train" if train else "test")
    // if not split_dir.exists() {
    0.0
}
