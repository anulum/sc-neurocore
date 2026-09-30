# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo owned dense IF C ownership acceptance

"""Exercise the shared real C ownership contract against compiled Mojo, with FP contraction off."""

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.conversion.if_parameters import OutputMode

from tests.test_accel_go_if_abi import (
    test_go_categorical_refusal_preserves_owner_slot as categorical_contract,
)
from tests.test_accel_go_if_abi import (
    test_pinned_complete_result_buffers_and_caller_custody as complete_buffer_contract,
)
from sc_neurocore.accel.mojo.isa_baseline import pin_isa


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile actual Mojo ownership symbols used by the shared complete C ABI contract."""
    root = Path(__file__).resolve().parents[1]
    kernels = root / "src/sc_neurocore/accel/mojo/kernels"
    target = tmp_path_factory.mktemp("mojo-if-abi") / "if-replay.so"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--fp-mode",
                "contract=off",
                "--diagnose-missing-doc-strings",
                "--Werror",
                "-I",
                str(kernels),
                "--emit",
                "shared-lib",
                "-o",
                str(target),
                str(kernels / "ann_to_snn_native.mojo"),
            ]
        ),
        capture_output=True,
        check=True,
        timeout=120,
    )
    return target


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
def test_mojo_complete_c_buffer_contract(
    library: Path, mode: OutputMode, binary: bool, shape: tuple[int, int]
) -> None:
    """Run every shared complete C-buffer case against the actual Mojo ownership library."""
    complete_buffer_contract(library, mode, binary, shape)


def test_mojo_categorical_refusal_preserves_owner_slot(library: Path) -> None:
    """Run shared domain/resource/overflow and untouched-slot refusals against actual Mojo."""
    categorical_contract(library)
