# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue receipt input reuse

"""Measure actual receipt reads across public catalogue and model-detail calls."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sc_neurocore.neurons.facet_receipts import RECEIPT_DIR, FacetReceiptError, load_receipt
from sc_neurocore.studio.catalogue_readiness import ReceiptSnapshot


def test_snapshot_keeps_one_operation_and_retries_a_refused_inventory(tmp_path: Path) -> None:
    """Keep parsed inputs local and recover normally after a malformed store is repaired.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Mutable receipt store, separate from all canonical evidence files.
    """
    directory = tmp_path / "receipts"
    snapshot = ReceiptSnapshot(directory)
    assert not directory.exists()
    empty = snapshot.read()
    assert not empty and not directory.exists()
    source = next(iter(sorted(RECEIPT_DIR.glob("*.json"))))
    original = source.read_bytes()
    expected = load_receipt(source)
    directory.mkdir()
    destination = directory / source.name
    destination.write_bytes(original)
    assert snapshot.read() is empty
    fresh = ReceiptSnapshot(directory)
    destination.write_text("{invalid receipt")
    with pytest.raises(FacetReceiptError):
        fresh.read()
    destination.write_bytes(original)
    key = (expected.class_name, expected.facet, expected.profile)
    assert fresh.read() == {key: (destination, expected)}
    assert snapshot.read() is empty


def test_catalogue_reads_each_receipt_once_and_detail_reads_fresh_inputs(tmp_path: Path) -> None:
    """Reuse parsed receipt inputs in one catalogue build, then read fresh detail evidence.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Receipt for actual file-open counts from an isolated source interpreter.
    """
    program = textwrap.dedent(
        """
        import collections
        import json
        import sys
        from pathlib import Path
        from sc_neurocore.neurons.facet_receipts import RECEIPT_DIR
        from sc_neurocore.neurons.models import __all__ as published_models
        from sc_neurocore.studio.model_catalogue import get_model_detail, list_models

        paths = {str(path.resolve()) for path in RECEIPT_DIR.glob('*.json')}
        assert paths, 'This source lane requires the actual receipt corpus.'
        reads = collections.Counter()
        def observe(event, arguments):
            if event == 'open' and arguments and isinstance(arguments[0], str):
                name = str(Path(arguments[0]).absolute())
                if name in paths:
                    reads[name] += 1
        sys.addaudithook(observe)
        models = list_models()
        assert {row['name'] for row in models} == set(published_models)
        built = dict(reads)
        assert list_models() is models
        assert dict(reads) == built
        detail = get_model_detail('SCStochasticRateAdaptationNeuron')
        assert detail is not None and detail['readiness']['verified']['source'] == 'receipts'
        after_detail = dict(reads)
        Path(sys.argv[1]).write_text(json.dumps({
            'models': models, 'detail': detail,
            'corpus_files': len(paths),
            'build_opens': sum(built.values()),
            'detail_opens': sum(after_detail.values()) - sum(built.values()),
            'build_counts': built, 'after_detail_counts': after_detail,
        }, sort_keys=True))
        assert set(built) == paths
        assert set(built.values()) == {1}, (len(paths), sum(built.values()))
        assert set(after_detail) == paths
        assert set(after_detail.values()) == {2}, (
            len(paths), sum(after_detail.values()) - sum(built.values())
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(tmp_path / "receipt-inputs.json")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads((tmp_path / "receipt-inputs.json").read_text())
    assert observed["build_opens"] == observed["detail_opens"] == observed["corpus_files"]
