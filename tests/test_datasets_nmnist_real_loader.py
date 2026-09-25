# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNMNISTRealLoader from former test_datasets.py

"""Focused suite: TestNMNISTRealLoader from former test_datasets.py."""

from __future__ import annotations

from tests.datasets_support import *  # noqa: F403


class TestNMNISTRealLoader:
    """Test N-MNIST real-data path with synthetic .bin files."""

    def test_parse_nmnist_bin(self, tmp_path):
        from sc_neurocore.datasets.loaders import _parse_nmnist_bin

        rng = np.random.default_rng(0)
        n_events = 50
        raw = rng.integers(0, 256, size=(n_events, 5), dtype=np.uint8)
        bin_file = tmp_path / "sample.bin"
        raw.tofile(bin_file)
        events = _parse_nmnist_bin(bin_file)
        assert events.shape == (n_events, 4)
        assert events.dtype == np.float32

    def test_parse_nmnist_bin_decodes_the_published_40_bit_event(self, tmp_path):
        """Byte 0 is x, byte 1 y, the top bit of byte 2 polarity, 23 bits of microseconds."""
        from sc_neurocore.datasets.loaders import _parse_nmnist_bin

        def encode(x: int, y: int, polarity: int, ts_us: int) -> bytes:
            return bytes([x, y, (polarity << 7) | (ts_us >> 16), (ts_us >> 8) & 0xFF, ts_us & 0xFF])

        bin_file = tmp_path / "known.bin"
        bin_file.write_bytes(
            encode(33, 0, 1, 1_234_567) + encode(0, 33, 0, 8_388_607) + encode(17, 5, 1, 0)
        )
        events = _parse_nmnist_bin(bin_file)
        np.testing.assert_allclose(
            events,
            [[33, 0, 1, 1234.567], [0, 33, 0, 8388.607], [17, 5, 1, 0.0]],
            rtol=0,
            atol=1e-3,
        )

    def test_load_nmnist_real_path(self, tmp_path):
        rng = np.random.default_rng(0)
        split = tmp_path / "Train"
        for cls in range(3):
            d = split / str(cls)
            d.mkdir(parents=True)
            for s in range(2):
                raw = rng.integers(0, 256, size=(20, 5), dtype=np.uint8)
                (d / f"s{s}.bin").write_bytes(raw.tobytes())
        (split / "README.txt").touch()
        samples, labels = load_nmnist(root=tmp_path, train=True, synthetic=False)
        assert len(samples) == 6
        assert set(labels.tolist()) == {0, 1, 2}

    def test_load_nmnist_missing_split_raises(self, tmp_path):
        (tmp_path / "sentinel").touch()
        with pytest.raises(FileNotFoundError, match="Expected split directory"):
            load_nmnist(root=tmp_path, train=True, synthetic=False)
