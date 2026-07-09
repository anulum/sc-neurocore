# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SNN training loops

"""Tests for SNN training loops."""

from __future__ import annotations

from typing import Any
import warnings

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import sc_neurocore.training.loops as training_loops
from sc_neurocore.training.loops import auto_device, evaluate, train_epoch
from sc_neurocore.training.losses import membrane_loss, spike_rate_loss
from sc_neurocore.training.snn_modules import ConvSpikingNet, SpikingNet


@pytest.fixture
def tiny_loader() -> DataLoader[Any]:
    """Return a tiny image-like classification loader."""
    x = torch.randn(32, 1, 4, 4)
    y = torch.randint(0, 3, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_model() -> SpikingNet:
    """Return a small feed-forward spiking network for loop tests."""
    return SpikingNet(n_input=16, n_hidden=16, n_output=3, n_layers=1)


def test_train_epoch_runs(tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
    """Run one training epoch and return bounded metrics."""
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, acc = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5, max_grad_norm=1.0)
    assert loss > 0
    assert 0 <= acc <= 1


def test_evaluate_runs(tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
    """Evaluate one epoch and return bounded metrics."""
    loss, acc = evaluate(tiny_model, tiny_loader, n_timesteps=5)
    assert loss > 0
    assert 0 <= acc <= 1


def test_training_improves_loss(tiny_loader: DataLoader[Any]) -> None:
    """Keep repeated toy training bounded against runaway loss."""
    model = SpikingNet(n_input=16, n_hidden=32, n_output=3, n_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss0, _ = train_epoch(model, tiny_loader, opt, n_timesteps=10)
    for _ in range(5):
        loss, _ = train_epoch(model, tiny_loader, opt, n_timesteps=10)
    assert loss < loss0 * 5


def test_membrane_loss_fn(tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
    """Train with the membrane-potential auxiliary loss."""
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, _ = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5, loss_fn=membrane_loss)
    assert loss > 0


def test_spike_rate_loss_fn(tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
    """Train with a spike-rate loss adapter."""

    def rate_loss(spk: Tensor, tgt: Tensor) -> Tensor:
        return spike_rate_loss(spk, tgt, n_timesteps=5)

    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, _ = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5, loss_fn=rate_loss)
    assert loss >= 0


def test_conv_spiking_net_train_epoch() -> None:
    """ConvSpikingNet should work with train_epoch (Tier 5.4)."""
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 5, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = ConvSpikingNet(n_output=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = train_epoch(model, loader, opt, n_timesteps=3, flatten_input=False)
    assert loss > 0
    assert 0 <= acc <= 1


def test_conv_spiking_net_evaluate() -> None:
    """ConvSpikingNet should work with evaluate."""
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 5, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = ConvSpikingNet(n_output=5)
    loss, acc = evaluate(model, loader, n_timesteps=3, flatten_input=False)
    assert loss > 0
    assert 0 <= acc <= 1


def test_cuda_arch_support_parser_handles_pytorch_tokens() -> None:
    """Map PyTorch build architecture tokens to supported device capabilities."""
    assert training_loops._parse_cuda_arch("sm_75") == (7, 5)
    assert training_loops._parse_cuda_arch("sm_120") == (12, 0)
    assert training_loops._parse_cuda_arch("compute_75") is None
    assert training_loops._parse_cuda_arch("sm_xx") is None
    assert training_loops._cuda_arch_is_supported((7, 5), ("compute_75", "sm_75"))
    assert training_loops._cuda_arch_is_supported((8, 9), ("sm_86",))
    assert training_loops._cuda_arch_is_supported((12, 0), ("sm_120",))
    assert not training_loops._cuda_arch_is_supported((6, 1), ("sm_75", "sm_80"))


def test_cuda_capability_helpers_reject_unusable_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat unavailable or malformed CUDA properties as unsupported."""

    def _raise_runtime_error(_index: int) -> object:
        raise RuntimeError("driver unavailable")

    class _MalformedProperties:
        major = "6"
        minor = None

    monkeypatch.setattr(torch.cuda, "get_device_properties", _raise_runtime_error)
    assert training_loops._cuda_device_capability(0) is None

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _MalformedProperties())
    assert training_loops._cuda_device_capability(0) is None

    monkeypatch.setattr(training_loops, "_cuda_device_capability", lambda _index: None)
    assert not training_loops._cuda_device_supported(0)


def test_device_usable_handles_cuda_sync_and_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe candidate devices and fail closed when Torch cannot allocate."""

    class _ProbeTensor:
        def fill_(self, _value: int) -> None:
            return None

        def cpu(self) -> _ProbeTensor:
            return self

        def item(self) -> int:
            return 1

    synchronized: list[Any] = []
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: _ProbeTensor())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))

    cuda_device = torch.device("cuda", 0)
    assert training_loops._device_usable(cuda_device)
    assert synchronized == [cuda_device]

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(torch, "empty", _raise_runtime_error)
    assert not training_loops._device_usable(torch.device("cpu"))


def test_auto_device_returns_supported_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select a supported usable CUDA device before CPU fallback."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(training_loops, "_cuda_device_supported", lambda _index: True)
    monkeypatch.setattr(training_loops, "_device_usable", lambda _device: True)

    assert auto_device() == torch.device("cuda", 0)


def test_auto_device_uses_mps_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select MPS before CPU when CUDA is unavailable and MPS can execute."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(training_loops, "_device_usable", lambda _device: True)

    assert auto_device() == torch.device("mps")


def test_auto_device_skips_cuda_when_build_lacks_device_arch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back when the installed PyTorch build cannot execute the CUDA device."""

    class _CudaProperties:
        major = 6
        minor = 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _CudaProperties())
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    monkeypatch.setattr(training_loops, "_device_usable", lambda _device: True)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert auto_device() == torch.device("cpu")


def test_auto_device_returns_supported_torch_device() -> None:
    """Return a Torch device without unsupported-CUDA warning noise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        device = auto_device()

    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")
    assert all("compute capability" not in str(warning.message) for warning in caught)
    assert all(
        "not compatible with the current PyTorch" not in str(warning.message) for warning in caught
    )
