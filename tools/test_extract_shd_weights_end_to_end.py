# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEndToEnd from former test_extract_shd_weights.py

"""Focused suite: TestEndToEnd from former test_extract_shd_weights.py."""

from __future__ import annotations

from extract_shd_weights_support import *  # noqa: F403


@pytest.mark.skipif(not os.path.exists(CKPT), reason="dcls_max checkpoint not present")
class TestEndToEnd:
    def test_extraction_produces_all_files(self, tmp_path: Path) -> None:
        out = str(tmp_path / "artifacts")
        stats = extract(CKPT, out, checkpoint_sha256=_sha256(CKPT))
        files = set(os.listdir(out))
        for layer in SHD_LAYERS:
            assert f"weights_{layer.name}_int8.hex" in files
            assert f"weights_{layer.name}.csr.json" in files
            if layer.delay_key is not None:
                assert f"delays_{layer.name}.hex" in files
        assert "scales.json" in files
        assert "network_params.vh" in files
        assert "metadata.json" in files

    def test_dequant_error_bounded_for_real_weights(self, tmp_path: Path) -> None:
        out = str(tmp_path / "artifacts")
        stats = extract(CKPT, out, checkpoint_sha256=_sha256(CKPT))
        for layer in stats["layers"]:
            # Per-tensor symmetric int8 quantisation gives bounded error
            assert layer["max_quant_err"] <= layer["scale"] / 2 + 1e-6
            assert layer["rel_quant_err"] < 0.01

    def test_extracted_delays_are_integers(self, tmp_path: Path) -> None:
        out = str(tmp_path / "artifacts")
        extract(CKPT, out, checkpoint_sha256=_sha256(CKPT))
        # delays_*.hex should only contain values that decode to [-15, 15]
        for fname in ["delays_layer1_input_to_h1.hex", "delays_layer2_h1_to_h2.hex"]:
            with open(os.path.join(out, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    byte = int(line, 16)
                    if byte >= 128:
                        byte -= 256
                    assert -15 <= byte <= 15
