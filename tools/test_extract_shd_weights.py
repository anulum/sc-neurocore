#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for tools/extract_shd_weights.py
import os
import sys
import hashlib
from pathlib import Path
from typing import cast

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
from sc_neurocore.security.checkpoint_loading import CheckpointTrustError  # noqa: E402


# ----- Quantisation -----


class TestQuantisation:
    def test_zero_tensor(self) -> None:
        w = torch.zeros(10, 5)
        w_q, scale = quantise_per_tensor_symmetric(w)
        assert (w_q == 0).all()
        assert scale == 0.0

    def test_symmetric_range(self) -> None:
        w = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # max abs = 1.0 → scale = 1/127
        assert scale == pytest.approx(1.0 / 127, rel=1e-9)
        # ±1.0 should map to ±127
        assert int(w_q[0, 0]) == -127
        assert int(w_q[0, 4]) == 127
        assert int(w_q[0, 2]) == 0

    def test_int8_dtype(self) -> None:
        w = torch.randn(10, 10) * 0.5
        w_q, _ = quantise_per_tensor_symmetric(w)
        assert w_q.dtype == torch.int8
        assert int(w_q.min()) >= -128
        assert int(w_q.max()) <= 127

    def test_dequant_max_error_bounded(self) -> None:
        w = torch.randn(50, 50) * 2.5
        w_q, scale = quantise_per_tensor_symmetric(w)
        w_dequant = w_q.float() * scale
        max_err = float((w - w_dequant).abs().max().item())
        # Per-tensor symmetric int8 has max error ≈ scale/2 = abs_max/254
        assert max_err <= scale / 2 + 1e-6

    def test_clipping_outliers(self) -> None:
        # Single huge outlier shouldn't make all other weights zero
        w = torch.tensor([[100.0, 0.5, -0.5, 0.0, 1.0]])
        w_q, scale = quantise_per_tensor_symmetric(w)
        # 100 / (100/127) = 127
        assert int(w_q[0, 0]) == 127
        # 0.5 / (100/127) = 0.635 → rounds to 1
        assert int(w_q[0, 1]) == 1


# ----- CSR conversion -----


class TestCsr:
    def test_dense_matrix(self) -> None:
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

    def test_all_zero(self) -> None:
        w = torch.zeros(3, 4, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 0
        assert csr["row_ptr"] == [0, 0, 0, 0]
        assert csr["col_idx"] == []
        assert csr["values"] == []

    def test_dense_no_zeros(self) -> None:
        w = torch.full((2, 3), 5, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 6
        assert csr["sparsity_pct"] == 0.0
        assert csr["row_ptr"] == [0, 3, 6]


# ----- Hex IO -----


class TestHexIO:
    def test_write_signed_int8_hex_round_trip(self, tmp_path: Path) -> None:
        w = torch.tensor([[-128, -1, 0, 1, 127]], dtype=torch.int8)
        path = str(tmp_path / "w.hex")
        write_int8_hex(w, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -128 → 0x80, -1 → 0xff, 0 → 0x00, 1 → 0x01, 127 → 0x7f
        assert lines == ["80", "ff", "00", "01", "7f"]

    def test_write_delays_negative_range(self, tmp_path: Path) -> None:
        delays = [-15, -1, 0, 1, 15]
        path = str(tmp_path / "d.hex")
        write_delays_hex(delays, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -15 → -15+256 = 241 = 0xf1, -1 → 0xff, 0 → 0x00, 1 → 0x01, 15 → 0x0f
        assert lines == ["f1", "ff", "00", "01", "0f"]


# ----- Layer spec sanity -----


class TestLayerSpec:
    def test_three_layers_defined(self) -> None:
        assert len(SHD_LAYERS) == 3

    def test_layer_dimensions_match_architecture(self) -> None:
        # 140 → 128 → 128 → 20
        assert SHD_LAYERS[0].in_features == 140
        assert SHD_LAYERS[0].out_features == 128
        assert SHD_LAYERS[1].in_features == 128
        assert SHD_LAYERS[1].out_features == 128
        assert SHD_LAYERS[2].in_features == 128
        assert SHD_LAYERS[2].out_features == 20

    def test_only_first_two_layers_have_delays(self) -> None:
        assert SHD_LAYERS[0].delay_key is not None
        assert SHD_LAYERS[1].delay_key is not None
        assert SHD_LAYERS[2].delay_key is None  # output layer has no axonal delay


class TestCheckpointTrustBoundary:
    def test_extract_requires_matching_checkpoint_sha256(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256="0" * 64)

    def test_extract_rejects_missing_checkpoint_sha256(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="checkpoint_sha256 is required"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256=cast(str, None))

    @pytest.mark.parametrize("bad_digest", ["abc123", "g" * 64])
    def test_extract_rejects_invalid_checkpoint_sha256_format(self, tmp_path: Path, bad_digest: str) -> None:
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="64 hexadecimal characters"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256=bad_digest)

    def test_extract_rejects_missing_net_state_dict(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "missing_net.pth"
        torch.save({"acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="must contain a dictionary 'net'"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_non_tensor_net_values(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "bad_net_values.pth"
        torch.save({"net": {"layers.1.weight": [1, 2, 3]}, "acc": 0.0, "sigma": 0.23}, checkpoint)

        with pytest.raises(ValueError, match="values must be tensors"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_non_finite_metadata(self, tmp_path: Path) -> None:
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.zeros(140),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_meta.pth"
        torch.save({"net": net, "acc": float("nan"), "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="metadata 'acc'"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_delay_length_mismatch(self, tmp_path: Path) -> None:
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.zeros(139),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_delay_len.pth"
        torch.save({"net": net, "acc": 0.0, "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="unexpected delay length"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_out_of_range_delays(self, tmp_path: Path) -> None:
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.full((140,), 16.0),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_delay_range.pth"
        torch.save({"net": net, "acc": 0.0, "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="must stay within \\[-15, 15\\]"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )


# ----- End-to-end on real checkpoint (skipped if not available) -----

REPO = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
CKPT = f"{REPO}/data/masquelier_shd/cloud_results/dcls_max/dcls_max/last.pth"


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
