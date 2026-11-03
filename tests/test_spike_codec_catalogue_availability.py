# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual codec availability and explicit unsupported combinations

"""Probe the installed public implementations without substituting native or optional modules."""

from __future__ import annotations

from importlib import import_module
import subprocess
import sys

import pytest

from sc_neurocore.spike_codec import list_codec_descriptors, probe_codec


@pytest.mark.parametrize("policy", ["permission", "runtime", "extension"])
def test_embedding_import_policy_reports_actual_loader_failure(policy: str) -> None:
    """A real audit policy denies native loading while public Python coding remains usable."""
    program = '''
import importlib.util
import sys

import numpy as np

policy = sys.argv[1]
installed = importlib.util.find_spec("sc_neurocore_engine") is not None
attempts: list[str] = []

def enforce(event: str, arguments: tuple[object, ...]) -> None:
    """Apply the host's import policy to the real native extension loader."""
    if event != "import" or str(arguments[0]) != "sc_neurocore_engine.sc_neurocore_engine":
        return
    attempts.append(str(arguments[0]))
    message = "native loading denied by embedding audit policy"
    if policy == "permission":
        raise PermissionError(message)
    if policy == "runtime":
        raise RuntimeError(message)
    raise ImportError(message)

sys.addaudithook(enforce)
from sc_neurocore.spike_codec import WaveformCodec, list_codec_descriptors, probe_codec

assert len(list_codec_descriptors()) == 8
assert not attempts
assert not any(name.startswith("sc_neurocore_engine") for name in sys.modules)
waveform = probe_codec("waveform", "lossless")
predictive = probe_codec("predictive")
assert waveform.python_available
assert not waveform.rust_components_available
assert not predictive.rust_components_available
assert waveform.diagnostics and predictive.diagnostics
if installed:
    assert attempts
    if policy == "extension":
        assert waveform.backend_choices == ("auto", "python")
        assert predictive.python_available
        assert any("ImportError" in line for line in waveform.diagnostics)
    else:
        assert waveform.backend_choices == ("python",)
        assert not predictive.python_available
        error_name = "PermissionError" if policy == "permission" else "RuntimeError"
        assert any(error_name in line for line in waveform.diagnostics)
        assert any(error_name in line for line in predictive.diagnostics)
else:
    assert not attempts
    assert waveform.backend_choices == ("auto", "python")
    assert predictive.python_available

samples = (np.arange(32, dtype=np.int16).reshape(-1, 1) * 7) - 90
encoder = WaveformCodec(mode="lossless", backend="python")
packet, result = encoder.compress(samples)
decoded = WaveformCodec.decompress(packet, backend="python")
assert result.lossless_waveform and decoded.lossless_waveform
np.testing.assert_array_equal(decoded.waveform, samples)
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, policy],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_public_export_is_unavailable_until_the_real_class_is_restored() -> None:
    """Detect a damaged public namespace and recover the actual lossless codec afterwards."""
    import numpy as np

    from sc_neurocore.spike_codec import describe_codec

    descriptor = describe_codec("waveform")
    module_name, _, symbol = descriptor.encoder_entrypoint.rpartition(".")
    module = import_module(module_name)
    original = getattr(module, symbol)
    samples = np.arange(64, dtype=np.int16).reshape(-1, 1)
    packet, _ = original(mode="lossless", backend="python").compress(samples)
    delattr(module, symbol)
    try:
        unavailable = probe_codec("waveform", "lossless")
        assert not unavailable.python_available
        assert unavailable.backend_choices == ()
        assert any("missing callable(s): " + symbol in line for line in unavailable.diagnostics)
        assert len(list_codec_descriptors()) == 8
        assert describe_codec("waveform") is descriptor
    finally:
        setattr(module, symbol, original)
    available = probe_codec("waveform", "lossless")
    assert available.python_available and "python" in available.backend_choices
    restored = getattr(module, symbol)
    assert restored is original
    np.testing.assert_array_equal(restored.decompress(packet, backend="python").waveform, samples)


def test_all_variants_report_actual_importable_components() -> None:
    """Probe results agree with the actual Python entry points and installed native symbols."""
    for descriptor in list_codec_descriptors():
        for variant in descriptor.variants:
            result = probe_codec(descriptor.name, variant.name)
            assert (result.name, result.variant) == (descriptor.name, variant.name)
            assert result.native_scope == variant.native_scope
            python_available = True
            for module in variant.optional_modules:
                try:
                    import_module(module)
                except ImportError:
                    python_available = False
            assert result.python_available == python_available
            native_available = False
            if variant.native_module:
                try:
                    native = import_module(variant.native_module)
                except ImportError:
                    assert result.diagnostics
                else:
                    native_available = all(
                        callable(getattr(native, symbol, None)) for symbol in variant.native_symbols
                    )
            assert result.rust_components_available == native_available
            if descriptor.backend_parameter:
                assert "python" in result.backend_choices
                assert ("rust" in result.backend_choices) == native_available
            else:
                assert result.backend_choices == ()
            if not python_available:
                assert variant.name == "world_model" and result.diagnostics


def test_no_gpu_or_placeholder_backend_is_advertised() -> None:
    """A Vulkan Rust encoder and generated language files do not create Python backend selectors."""
    for descriptor in list_codec_descriptors():
        result = probe_codec(descriptor.name)
        assert set(result.backend_choices) <= {"python", "rust", "auto"}
        assert result.variant == descriptor.variant().name
    for name in ("go", "julia", "mojo", "rust", "gpu", "waveform_stream"):
        with pytest.raises(ValueError, match="Unknown codec"):
            probe_codec(name)
    with pytest.raises(ValueError, match="Unknown variant"):
        probe_codec("waveform", "gpu")


def test_unavailable_explicit_rust_is_refused_by_real_public_constructors() -> None:
    """A base-only installed environment must reject explicit Rust rather than return dummy output."""
    for name in ("waveform", "lossless_stream", "spike_events"):
        result = probe_codec(name)
        if result.rust_components_available:
            assert "rust" in result.backend_choices
            continue
        from sc_neurocore.spike_codec import describe_codec

        descriptor = describe_codec(name)
        module, _, symbol = descriptor.encoder_entrypoint.rpartition(".")
        cls = getattr(import_module(module), symbol)
        kwargs: dict[str, str | int | float] = {"backend": "rust"}
        if name != "waveform":
            kwargs["channels"] = 2
        if name == "spike_events":
            kwargs["sample_rate_hz"] = 19531.0
        with pytest.raises(ValueError, match="[Rr]ust"):
            cls(**kwargs)


def test_blocked_optional_module_is_unavailable_and_its_import_state_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a genuine import-system refusal and restore the optional module afterwards."""
    from sc_neurocore.spike_codec import describe_codec

    selected = describe_codec("predictive").variant("world_model")
    module_name = selected.optional_modules[0]
    baseline = probe_codec("predictive", "world_model")
    original = sys.modules.get(module_name)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, module_name, None)
        with pytest.raises(ModuleNotFoundError, match="None in sys.modules"):
            import_module(module_name)
        unavailable = probe_codec("predictive", "world_model")
        assert not unavailable.python_available
        assert unavailable.backend_choices == ()
        assert unavailable.diagnostics
        diagnostic = unavailable.diagnostics[0]
        assert diagnostic.startswith(module_name + ": ModuleNotFoundError:")
        assert "None in sys.modules" in diagnostic
    assert sys.modules.get(module_name) is original
    assert probe_codec("predictive", "world_model").python_available == baseline.python_available
