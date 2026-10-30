# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real snnTorch admission and SC-contract refusals

"""Keep unsupported dynamics, source features and SC contracts out of inference."""

from typing import Any, Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
pytest.importorskip("nirtorch")
from torch import nn

from sc_neurocore.nir_bridge import from_snntorch, to_snntorch
from tests.test_nir_bridge.snntorch_models import make_model as _model


@pytest.mark.parametrize(
    "change,message",
    [
        ({"reset_delay": True}, "Delayed reset"),
        ({"inhibition": True}, "inhibition"),
        ({"init_hidden": False}, "init_hidden"),
        ({"output": True}, "output=False"),
        ({"reset_mechanism": "none"}, "zero or subtract"),
        ({"graded_spikes_factor": 2.0}, "Graded spikes"),
        ({"beta": -0.1}, "beta"),
        ({"beta": 1.1}, "beta"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"threshold": -1.0}, "threshold"),
        ({"beta": float("nan")}, "finite"),
    ],
)
def test_real_cell_features_refused(change: dict[str, Any], message: str) -> None:
    """Refuse source cell features whose target dynamics would change."""
    options = dict(alpha=0.8, beta=0.7, init_hidden=True, reset_delay=False)
    options.update({key: value for key, value in change.items() if key != "graded_spikes_factor"})
    cell = snn.Synaptic(**options)
    if "graded_spikes_factor" in change:
        cell.graded_spikes_factor = torch.tensor(change["graded_spikes_factor"])
    model = nn.Sequential(nn.Linear(4, 3, dtype=torch.float64), cell).double()
    with pytest.raises(ValueError, match=message):
        from_snntorch(model, torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize(
    "sample",
    [
        torch.ones(4),
        torch.ones(1, 4, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        torch.full((4,), float("inf"), dtype=torch.float64),
    ],
)
def test_precision_and_input_boundary_refused(sample: Any) -> None:
    """Reject unsupported tensor precision, batches and nonfinite input samples."""
    with pytest.raises(ValueError):
        from_snntorch(_model(), sample)


def test_incompatible_topologies_and_sources_refused() -> None:
    """Reject real layer, topology, sharing and hook configurations outside the profile."""
    x = torch.ones(4, dtype=torch.float64)
    for model in [nn.Sequential(), nn.Linear(4, 3), object()]:
        with pytest.raises(TypeError):
            from_snntorch(model, x)
    with pytest.raises(TypeError, match="Torch tensor"):
        from_snntorch(_model(), np.ones(4))
    for model in [
        nn.Sequential(nn.ReLU()),
        nn.Sequential(nn.Sequential(nn.Linear(4, 3))),
        nn.Sequential(nn.Linear(5, 3, dtype=torch.float64)),
        nn.Sequential(nn.Linear(4, 3)),
        nn.Sequential(
            snn.Leaky(beta=torch.tensor([0.2, 0.5]), init_hidden=True, reset_delay=False)
        ).double(),
        nn.Sequential(
            snn.Leaky(beta=0.8, init_hidden=True, reset_delay=False, reset_mechanism="zero"),
            snn.Leaky(beta=0.8, init_hidden=True, reset_delay=False),
        ).double(),
    ]:
        with pytest.raises(ValueError):
            from_snntorch(model, x)
    shared = snn.Leaky(beta=0.8, init_hidden=True, reset_delay=False).double()
    with pytest.raises(ValueError, match="Shared module"):
        from_snntorch(nn.Sequential(shared, shared), x)
    hooked = _model()
    hook = hooked[0].register_forward_hook(lambda module, args, output: output * 2)
    try:
        with pytest.raises(ValueError, match="hooks"):
            from_snntorch(hooked, x)
    finally:
        hook.remove()


def test_stochastic_contracts_refused_in_both_directions() -> None:
    # Real SC-NIR semantics, rather than pretending the float path simulates streams.
    """Keep actual SC-NIR stream contracts out of floating framework inference."""
    from sc_neurocore.ir.scnir_convert import SCNIRConversionConfig, build_scnir_from_neuron_graph
    from sc_neurocore.nir_bridge import from_scnetwork

    model = _model(reset="zero")
    network = from_snntorch(model, torch.ones(4, dtype=torch.float64))
    document = build_scnir_from_neuron_graph(
        from_scnetwork(network),
        config=SCNIRConversionConfig(bitstream_length=256, data_width=16, fraction=8),
    )
    conversions: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
        (from_snntorch, (model, torch.ones(4, dtype=torch.float64))),
        (to_snntorch, (network,)),
    ]
    for convert, args in conversions:
        with pytest.raises(ValueError, match="bitstream length, encoding, Q-format"):
            convert(*args, scnir=document)
    with pytest.raises(TypeError, match="Q-format"):
        to_snntorch(document)
    for embedded in (
        document,
        document.streams[0],
        document.streams[0].precision,
        document.streams[0].source,
    ):
        network.nir_metadata["stream_contract"] = embedded
        with pytest.raises(ValueError, match="bitstream length, encoding, Q-format"):
            to_snntorch(network)
    model.scnir = document
    with pytest.raises(ValueError, match="bitstream length, encoding, Q-format"):
        from_snntorch(model, torch.ones(4, dtype=torch.float64))


def test_metadata_annotations_and_cyclic_annotation_ownership_do_not_change_inference() -> None:
    """Accept ordinary annotations while refusing SC stream declarations separately."""
    model = _model(reset="zero")
    annotation: dict[str, Any] = {"encoding": "utf8", "label": "{experiment annotation}"}
    annotation["self"] = annotation
    model.nir_metadata = annotation
    network = from_snntorch(model, torch.ones(4, dtype=torch.float64))
    network.nir_metadata["annotation"] = annotation
    exported = to_snntorch(network)
    with torch.no_grad():
        np.testing.assert_array_equal(
            network.step({network.input_nodes[0]: np.ones(4)})[network.output_nodes[0]],
            exported(torch.ones(4, dtype=torch.float64)).numpy(),
        )


def test_forward_callbacks_invalid_buffers_and_incompatible_precision_are_refused() -> None:
    """Refuse actual source features that would change spike decisions or execution."""
    model = _model()
    hook = model.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    try:
        with pytest.raises(ValueError, match="hooks"):
            from_snntorch(model, torch.ones(4, dtype=torch.float64))
    finally:
        hook.remove()
    model[1].beta = None
    with pytest.raises(ValueError, match="must be a Torch tensor"):
        from_snntorch(model, torch.ones(4, dtype=torch.float64))
    custom = nn.Sequential(
        snn.Leaky(
            beta=0.8, init_hidden=True, reset_delay=False, spike_grad=lambda x: (x >= 0).double()
        )
    ).double()
    with pytest.raises(ValueError, match="spike decision is qualified"):
        from_snntorch(custom, torch.ones(4, dtype=torch.float64))
    incompatible = nn.Sequential(
        nn.Linear(4, 3, dtype=torch.float64),
        snn.Leaky(beta=0.8, init_hidden=True, reset_delay=False),
        nn.Linear(3, 2, dtype=torch.float64),
    ).double()
    with pytest.raises(ValueError, match="source sequence cannot execute"):
        from_snntorch(incompatible, torch.ones(4, dtype=torch.float64))
