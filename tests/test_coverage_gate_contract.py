# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python coverage threshold cross-surface contract

"""Keep local, preflight, and hosted Python coverage gates at exact closure."""

from __future__ import annotations

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10 fallback


_ROOT = Path(__file__).resolve().parents[1]


def _cov_thresholds(path: Path) -> tuple[int, ...]:
    """Return explicit pytest-cov thresholds from one owned command surface."""

    text = path.read_text(encoding="utf-8")
    return tuple(int(value) for value in re.findall(r"--cov-fail-under(?:=|\s+)(\d+)", text))


def test_python_coverage_threshold_is_exact_closure() -> None:
    """The canonical config drives preflight and must reject any missing statement."""

    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["coverage"]["report"]["fail_under"] == 100


def test_explicit_local_and_hosted_thresholds_match_canonical_gate() -> None:
    """Makefile and primary hosted CI must not weaken the canonical threshold."""

    assert _cov_thresholds(_ROOT / "Makefile") == (100,)
    assert _cov_thresholds(_ROOT / ".github/workflows/ci.yml") == (100,)


def test_nagumo_sato_and_sc_map_exact_coverage_and_parity_is_hosted() -> None:
    """Both preserved map identities need branch and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index(
        "- name: Nagumo-Sato and SC adaptive-map exact coverage and backend parity"
    )
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/nagumo_sato_map_neuron.py" in step
    assert "src/sc_neurocore/neurons/models/sc_adaptive_threshold_map_neuron.py" in step
    assert "tests/test_model_nagumo_sato_map_neuron.py" in step
    assert "tests/test_nagumo_sato_map_backends.py" in step
    assert "tests/test_model_sc_adaptive_threshold_map_neuron.py" in step
    assert "tests/test_sc_adaptive_threshold_map_backends.py" in step
    assert "tests/test_model_kilinc_bhatt_map_neuron.py" in step
    assert "tests/test_kilinc_bhatt_map_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_ih_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted channel model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: Ih exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/ih_neuron.py'" in step
    assert "tests/test_model_ih_neuron.py" in step
    assert "tests/test_ih_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_persistent_na_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted INaP model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: PersistentNa exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/persistent_na_neuron.py'" in step
    assert "tests/test_model_persistent_na_neuron.py" in step
    assert "tests/test_persistent_na_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_nmda_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted NMDA channel model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: NMDA exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/nmda_neuron.py'" in step
    assert "tests/test_model_nmda_neuron.py" in step
    assert "tests/test_nmda_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_sk_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted SK channel model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: SK exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/sk_neuron.py'" in step
    assert "tests/test_model_sk_neuron.py" in step
    assert "tests/test_sk_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_ttype_ca_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted T-type model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: TTypeCa exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/ttype_ca_neuron.py'" in step
    assert "tests/test_model_ttype_ca_neuron.py" in step
    assert "tests/test_ttype_ca_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_glm_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted GLM model needs an explicit branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: GLM exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/glm_neuron.py'" in step
    assert "tests/test_model_glm_neuron_glm_atomicity.py" in step
    assert "tests/test_glm_neuron_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_mainen_sejnowski_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The omitted two-compartment model needs a branch gate and backend custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: MainenSejnowski exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/neurons/models/mainen_sejnowski.py'" in step
    assert "tests/test_model_mainen_sejnowski_ms_atomicity.py" in step
    assert "tests/test_mainen_sejnowski_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_tc_lif_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The TC-LIF family (canonical + both SC identities) needs branch custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: TwoCompartmentLIF exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/tc_lif.py" in step
    assert "src/sc_neurocore/neurons/models/sc_leaky_tc_lif.py" in step
    assert "src/sc_neurocore/neurons/models/sc_exponential_tc_lif.py" in step
    assert "tests/test_model_tc_lif_atomicity.py" in step
    assert "tests/test_tc_lif_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_psn_exact_coverage_and_backend_parity_is_hosted() -> None:
    """The sliding PSN family (canonical + SC identity) needs branch custody."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: ParallelSpiking exact coverage and backend parity")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/neurons/models/psn.py" in step
    assert "src/sc_neurocore/neurons/models/sc_resetting_psn.py" in step
    assert "tests/test_model_psn_atomicity.py" in step
    assert "tests/test_psn_backends.py" in step
    assert "--fail-under=100 --show-missing" in step


def test_hdc_exact_coverage_is_hosted() -> None:
    """The rebuilt HDC surface needs an explicit branch gate."""

    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("- name: HDC exact coverage")
    end = workflow.index("\n      - name:", start + 1)
    step = workflow[start:end]

    assert "if: matrix.primary" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "src/sc_neurocore/hdc/base.py" in step
    assert "src/sc_neurocore/hdc/classifier.py" in step
    assert "tests/test_hdc/" in step
    assert "--fail-under=100 --show-missing" in step


def test_exactly_one_primary_matrix_leg_carries_the_exact_coverage_lanes() -> None:
    """Every exact-coverage lane rides `if: matrix.primary`; losing the single
    `primary: true` matrix entry would silently disable all of them on every
    leg, so the matrix must always declare exactly one primary interpreter."""

    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - PyYAML ships with the dev env.
        import pytest

        pytest.skip("PyYAML is not installed")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    include = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    primary_legs = [leg for leg in include if leg.get("primary") is True]
    assert len(primary_legs) == 1, (
        "the test matrix must declare exactly one primary leg; "
        f"found {len(primary_legs)} of {len(include)}"
    )
