# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConversionPackageFacade from former test_conversion_ann_snn.py

"""Focused suite: TestConversionPackageFacade from former test_conversion_ann_snn.py."""

from __future__ import annotations

from collections.abc import Callable
import builtins
import pytest
from tests.conversion_ann_snn_support import (
    ConvertedSNN,
    QCFSActivation,
    replace_relu_with_qcfs,
)


class TestConversionPackageFacade:
    """The lazy public facade of the conversion package."""

    def test_lazy_facade_resolves_public_exports(self) -> None:
        """The facade resolves each public export lazily."""
        import sc_neurocore.conversion as conversion
        from sc_neurocore.conversion.ann_to_snn import convert

        assert conversion.__getattr__("convert") is convert
        assert conversion.__getattr__("ConvertedSNN") is ConvertedSNN
        assert conversion.__getattr__("QCFSActivation") is QCFSActivation
        assert conversion.__getattr__("replace_relu_with_qcfs") is replace_relu_with_qcfs

    def test_lazy_facade_rejects_unknown_export(self) -> None:
        """The facade rejects an unknown export."""
        import sc_neurocore.conversion as conversion

        with pytest.raises(AttributeError, match="not_exported"):
            conversion.__getattr__("not_exported")

    def test_lazy_facade_reports_qcfs_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed QCFS import is reported with its requirement."""
        import sc_neurocore.conversion as conversion

        original_import: Callable[..., object] = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "qcfs":
                raise ImportError("forced missing torch surface")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(ImportError, match="QCFSActivation requires PyTorch"):
            conversion.__getattr__("QCFSActivation")
