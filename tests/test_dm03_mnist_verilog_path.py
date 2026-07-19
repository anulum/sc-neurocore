# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "examples/dm03_mnist_verilog_path.md"


def test_dm03_points_only_to_existing_runnable_paths() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert (REPO / "examples/mnist_fpga/demo.py").is_file()
    assert (REPO / "notebooks/13_quantisation_pipeline.ipynb").is_file()
    assert (REPO / "notebooks/08_equation_to_verilog.ipynb").is_file()
    assert (REPO / "notebooks/27_python_to_proven_silicon.ipynb").is_file()
    assert "python examples/mnist_fpga/demo.py" in text
    assert "documentation only" in text.lower()
    assert "Does not prove" in text
