#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for tools/extract_shd_weights.py
import os
import sys

import pytest

# Skip the entire module if torch is not installed (CI without venv)
torch = pytest.importorskip("torch")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extract_shd_weights import (  # noqa: E402
    SHD_LAYERS,
    extract,
    quantise_per_tensor_symmetric,
    to_csr,
    write_int8_hex,
    write_delays_hex,
)


# ----- Quantisation -----


class TestQuantisation:
    def test_zero_tensor(self):
        w = torch.zeros(10, 5)
        w_q, scale = quantise_per_tensor_symmetric(w)
        assert (w_q == 0).all()
        assert scale == 0.0

    def test_symmetric_range(self):
        w = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # max abs = 1.0 → scale = 1/127
        assert scale == pytest.approx(1.0 / 127, rel=1e-9)
        # ±1.0 should map to ±127
        assert int(w_q[0, 0]) == -127
        assert int(w_q[0, 4]) == 127
        assert int(w_q[0, 2]) == 0

    def test_int8_dtype(self):
        w = torch.randn(10, 10) * 0.5
        w_q, _ = quantise_per_tensor_symmetric(w)
        assert w_q.dtype == torch.int8
        assert int(w_q.min()) >= -128
        assert int(w_q.max()) <= 127

    def test_dequant_max_error_bounded(self):
        w = torch.randn(50, 50) * 2.5
        w_q, scale = quantise_per_tensor_symmetric(w)
        w_dequant = w_q.float() * scale
        max_err = float((w - w_dequant).abs().max().item())
        # Per-tensor symmetric int8 has max error ≈ scale/2 = abs_max/254
        assert max_err <= scale / 2 + 1e-6

    def test_clipping_outliers(self):
        # Single huge outlier shouldn't make all other weights zero
        w = torch.tensor([[100.0, 0.5, -0.5, 0.0, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # 100 / (100/127) = 127
        assert int(w_q[0, 0]) == 127
        # 0.5 / (100/127) = 0.635 → rounds to 1
        assert int(w_q[0, 1]) == 1


# ----- CSR conversion -----


class TestCsr:
    def test_dense_matrix(self):
        # 90% sparse: 1 nz out of 10
        w = torch.zeros(2, 5, dtype=torch.int8)
        w[0, 2] = 5
        w[1, 4] = -3
        csr = to_csr(w)
        assert csr["shape"] == [2, 5]
        assert csr["nnz"] == 2
        assert csr["row_ptr"] == [0, 1, 2]
        assert csr["col_idx"] == [2, 4]
        assert csr["values"] == [5, -3]
        assert csr["sparsity_pct"] == pytest.approx(80.0)

    def test_all_zero(self):
        w = torch.zeros(3, 4, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 0
        assert csr["row_ptr"] == [0, 0, 0, 0]
        assert csr["col_idx"] == []
        assert csr["values"] == []

    def test_dense_no_zeros(self):
        w = torch.full((2, 3), 5, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 6
        assert csr["sparsity_pct"] == 0.0
        assert csr["row_ptr"] == [0, 3, 6]


# ----- Hex IO -----


class TestHexIO:
    def test_write_signed_int8_hex_round_trip(self, tmp_path):
        w = torch.tensor([[-128, -1, 0, 1, 127]], dtype=torch.int8)
        path = str(tmp_path / "w.hex")
        write_int8_hex(w, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -128 → 0x80, -1 → 0xff, 0 → 0x00, 1 → 0x01, 127 → 0x7f
        assert lines == ["80", "ff", "00", "01", "7f"]

    def test_write_delays_negative_range(self, tmp_path):
        delays = [-15, -1, 0, 1, 15]
        path = str(tmp_path / "d.hex")
        write_delays_hex(delays, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -15 → -15+256 = 241 = 0xf1, -1 → 0xff, 0 → 0x00, 1 → 0x01, 15 → 0x0f
        assert lines == ["f1", "ff", "00", "01", "0f"]


# ----- Layer spec sanity -----


class TestLayerSpec:
    def test_three_layers_defined(self):
        assert len(SHD_LAYERS) == 3

    def test_layer_dimensions_match_architecture(self):
        # 140 → 128 → 128 → 20
        assert SHD_LAYERS[0].in_features == 140
        assert SHD_LAYERS[0].out_features == 128
        assert SHD_LAYERS[1].in_features == 128
        assert SHD_LAYERS[1].out_features == 128
        assert SHD_LAYERS[2].in_features == 128
        assert SHD_LAYERS[2].out_features == 20

    def test_only_first_two_layers_have_delays(self):
        assert SHD_LAYERS[0].delay_key is not None
        assert SHD_LAYERS[1].delay_key is not None
        assert SHD_LAYERS[2].delay_key is None  # output layer has no axonal delay


# ----- End-to-end on real checkpoint (skipped if not available) -----

REPO = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
CKPT = f"{REPO}/data/masquelier_shd/cloud_results/dcls_max/dcls_max/last.pth"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="dcls_max checkpoint not present")
class TestEndToEnd:
    def test_extraction_produces_all_files(self, tmp_path):
        out = str(tmp_path / "artifacts")
        stats = extract(CKPT, out)
        files = set(os.listdir(out))
        for layer in SHD_LAYERS:
            assert f"weights_{layer.name}_int8.hex" in files
            assert f"weights_{layer.name}.csr.json" in files
            if layer.delay_key is not None:
                assert f"delays_{layer.name}.hex" in files
        assert "scales.json" in files
        assert "network_params.vh" in files
        assert "metadata.json" in files

    def test_dequant_error_bounded_for_real_weights(self, tmp_path):
        out = str(tmp_path / "artifacts")
        stats = extract(CKPT, out)
        for layer in stats["layers"]:
            # Per-tensor symmetric int8 quantisation gives bounded error
            assert layer["max_quant_err"] <= layer["scale"] / 2 + 1e-6
            assert layer["rel_quant_err"] < 0.01

    def test_extracted_delays_are_integers(self, tmp_path):
        out = str(tmp_path / "artifacts")
        extract(CKPT, out)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
