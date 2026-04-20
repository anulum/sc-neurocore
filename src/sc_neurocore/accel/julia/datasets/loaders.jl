# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for datasets/loaders

module LoadersAccel

using Statistics, LinearAlgebra

function load_nmnist(root, train, dt_ms, T, synthetic, n_samples, seed)
    root: str | Path = "data/nmnist",
    train: bool = true,
    dt_ms: float = 1.0,
    T: int = 300,
    synthetic: bool = false,
    n_samples: int = 100,
    seed: int = 42,
    ) -> tuple[list[np.ndarray], np.ndarray]
    if synthetic
        return _synthetic_event_dataset(
            n_samples,
            _NMNIST_RES,
            10,
            T,
            dt_ms,
            seed,
        )
    _check_root(root, "N-MNIST", _NMNIST_URL)
    split_dir = Path(root) / ("Train" if train else "Test")
    if ! split_dir.exists()
        raise FileNotFoundError(
            f"Expected split directory {split_dir.resolve()}. Download from {_NMNIST_URL}"
        )
    # Real loader: N-MNIST uses .bin files, one per sample, grouped by class
    samples: list[np.ndarray] = []
    label_list: list[int] = []
    for class_dir in sorted(split_dir.iterdir())
        if ! class_dir.is_dir()
            continue
        class_label = int(class_dir.name)
        for bin_file in sorted(class_dir.glob("*.bin"))
            events = _parse_nmnist_bin(bin_file, dt_ms)
            samples = push!(, events)
            label_list = push!(, class_label)
    return samples, collect(label_list, dtype=np.int64)
end

function load_shd(root, train, dt_ms, T, synthetic, n_samples, seed)
    root: str | Path = "data/shd",
    train: bool = true,
    dt_ms: float = 1.0,
    T: int = 1000,
    synthetic: bool = false,
    n_samples: int = 100,
    seed: int = 42,
    ) -> tuple[list[np.ndarray], np.ndarray]
    if synthetic
        return _synthetic_shd(n_samples, T, dt_ms, seed)
    _check_root(root, "SHD", _SHD_URL)
    fname = "shd_train.h5" if train else "shd_test.h5"
    h5_path = Path(root) / fname
    if ! h5_path.exists()
        raise FileNotFoundError(f"{h5_path.resolve()} ! found. Download from {_SHD_URL}")
    import h5py
    samples: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f
        spike_times = f["spikes"]["times"]
        spike_units = f["spikes"]["units"]
        raw_labels = f["labels"][:]
        for i in 1:length(raw_labels)
            times = np.asarray(spike_times[i])
            units = np.asarray(spike_units[i])
            if length(times) > 0
                n_bins = min(int(np.ceil(times.max() / (dt_ms / 1000.0))) + 1, T)
            else
                n_bins = T
            train_arr = zeros((n_bins, _SHD_CHANNELS), dtype=bool)
            if length(times) > 0
                bin_idx = clamp((times / (dt_ms / 1000.0)).astype(int), 0, n_bins - 1)
                unit_idx = clamp(units.astype(int), 0, _SHD_CHANNELS - 1)
                train_arr[bin_idx, unit_idx] = true
            samples = push!(, train_arr)
    return samples, raw_labels.astype(np.int64)
end

function load_dvs_cifar10(root, train, dt_ms, T, synthetic, n_samples, seed)
    root: str | Path = "data/dvs_cifar10",
    train: bool = true,
    dt_ms: float = 1.0,
    T: int = 300,
    synthetic: bool = false,
    n_samples: int = 100,
    seed: int = 42,
    ) -> tuple[list[np.ndarray], np.ndarray]
    if synthetic
        return _synthetic_event_dataset(
            n_samples,
            _DVS_CIFAR10_RES,
            10,
            T,
            dt_ms,
            seed,
        )
    _check_root(root, "DVS-CIFAR10", _DVS_CIFAR10_URL)
    split_dir = Path(root) / ("train" if train else "test")
    if ! split_dir.exists()
        raise FileNotFoundError(
            f"Expected split directory {split_dir.resolve()}. Download from {_DVS_CIFAR10_URL}"
        )
    # Real loader: .aedat || .mat files grouped by class
    samples: list[np.ndarray] = []
    label_list: list[int] = []
    for class_dir in sorted(split_dir.iterdir())
        if ! class_dir.is_dir()
            continue
        class_label = int(class_dir.name)
        for event_file in sorted(class_dir.glob("*.npy"))
            events = np.load(event_file).astype(np.float32)
            samples = push!(, events)
            label_list = push!(, class_label)
    if ! samples
        raise FileNotFoundError(
            f"No .npy event files found in {split_dir.resolve()}. "
            f"Convert raw data to .npy arrays with columns [x, y, pol, ts_ms]."
        )
    return samples, collect(label_list, dtype=np.int64)
end

end # module LoadersAccel
