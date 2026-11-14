# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The Q-formats offered for a model hold the neuron the Studio compiles

"""A model is offered the Q-formats that hold the neuron the Studio sends, not only its schema's own.

The compile configuration stated which formats can hold a model by building
the schema's neuron at the schema's defaults. The Studio sends the catalogue's
defaults, and for one model they differ in eight parameters, so its default
format, Q8.8, was offered and then refused ("parameter tau_2=200.0 becomes
-56.0"). The formats are now judged on the request the Studio sends.
"""

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.studio.model_catalogue import get_model_detail, list_models
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
from sc_neurocore.studio.model_compile_parameters import CatalogueParametersRefused


def _studio_default_request(model_name: str, q_format: str) -> dict[str, object]:
    """Build the request the Studio sends for a model at its defaults, at one format.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    q_format : str
        The format asked for.

    Returns
    -------
    dict[str, object]
        The catalogue defaults, the model's own step and the default integrator.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    return {
        "model_name": model_name,
        "dt": detail["dt"],
        "integrator": detail["compile_configuration"]["default_integrator"],
        "q_format": q_format,
        "params": {parameter["name"]: parameter["default"] for parameter in detail["params"]},
    }


def _compiled_models() -> list[str]:
    """Return every catalogue model the Studio offers 'Generate RTL' for.

    Returns
    -------
    list of str
        Their names, in the catalogue's order.
    """
    names = []
    for summary in list_models():
        detail = get_model_detail(summary["name"])
        configuration = detail.get("compile_configuration") if detail else None
        if configuration and configuration["q_formats"]:
            names.append(summary["name"])
    return names


def test_the_scaled_reset_neuron_is_offered_only_a_format_that_holds_its_catalogue_defaults() -> (
    None
):
    """Q8.8 cannot hold tau_2 = 200, the catalogue's default, so it is not offered; Q16.16 is the default."""
    detail = get_model_detail("SCScaledResetAdaptiveIFNeuron")
    assert detail is not None
    configuration = detail["compile_configuration"]
    assert configuration["q_formats"] == ["Q16.16"]
    assert configuration["default_q_format"] == "Q16.16"
    refused = configuration["numeric_contracts"]["Q8.8"]
    assert refused["representable"] is False
    assert "tau_2=200.0 becomes -56.0" in refused["refusal"]
    resolved = resolve_model_compile_configuration(
        _studio_default_request("SCScaledResetAdaptiveIFNeuron", "Q16.16"),
        detail_getter=get_model_detail,
    )
    assert resolved.params["tau_2"] == 200.0


@pytest.mark.parametrize("model_name", _compiled_models())
def test_every_offered_format_holds_the_request_the_studio_sends(model_name: str) -> None:
    """Each format the configuration offers compiles the Studio's default request for that model.

    Where the request itself cannot be compiled for another reason that is
    stated in words (the Amari field's 64 sites against its schema's four), the
    formats are judged on the schema's own neuron, which a reader reaches by
    following that reason.

    Parameters
    ----------
    model_name : str
        A model that offers 'Generate RTL'.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    offered = detail["compile_configuration"]["q_formats"]
    assert detail["compile_configuration"]["default_q_format"] == offered[0]
    for q_format in offered:
        request = _studio_default_request(model_name, q_format)
        try:
            resolve_model_compile_configuration(request, detail_getter=get_model_detail)
        except CatalogueParametersRefused:
            resolve_model_compile_configuration(
                {"model_name": model_name, "q_format": q_format}, detail_getter=get_model_detail
            )


def test_a_default_request_the_schema_cannot_name_leaves_the_schema_s_neuron() -> None:
    """A catalogue name the schema does not know cannot judge the formats; the schema's neuron does.

    The compile refuses such a request by name, so the formats offered are
    those of the neuron the reader compiles once the name is dropped. The
    scaled-reset neuron is used because its catalogue defaults differ from
    its schema's, so the two neurons, and their RTL, can be told apart.
    """
    from dataclasses import replace

    from sc_neurocore.neurons.model_catalogue import load_descriptor
    from sc_neurocore.neurons.universal_dsl import load_schema
    from sc_neurocore.studio.model_catalogue import _studio_default_neuron

    def rtl(neuron: UniversalNeuron) -> str:
        """Return the neuron's RTL at Q16.16.

        Parameters
        ----------
        neuron : UniversalNeuron
            The neuron.

        Returns
        -------
        str
            Its Verilog.
        """
        return neuron.to_verilog(module_name="m", data_width=32, fraction=16)

    descriptor = load_descriptor("SCScaledResetAdaptiveIFNeuron")
    assert descriptor is not None
    schema_name = "sc_scaled_reset_adaptive_if"
    schema = load_schema(schema_name)
    schema_rtl = rtl(UniversalNeuron.from_schema(schema_name))
    assert rtl(_studio_default_neuron(descriptor, schema_name, schema, "rk4")) != schema_rtl
    extra = replace(descriptor.parameters[0], name="not_in_the_schema", default=1.0)
    widened = replace(descriptor, parameters=(*descriptor.parameters, extra))
    assert rtl(_studio_default_neuron(widened, schema_name, schema, "rk4")) == schema_rtl


def test_the_formats_are_judged_at_the_step_the_studio_sends() -> None:
    """The neuron that judges the formats runs at the catalogue's step, which the Studio sends.

    Today every catalogue step equals its schema's, so this holds a model to
    a finer step than its schema's: at 1e-6 ms, Q16.16 rounds the step to zero
    and can no longer hold the neuron, though it holds the schema's own.
    """
    from dataclasses import replace

    from sc_neurocore.neurons.model_catalogue import load_descriptor
    from sc_neurocore.neurons.universal_dsl import load_schema
    from sc_neurocore.studio.model_catalogue import _studio_default_neuron
    from sc_neurocore.studio.model_numeric_contracts import studio_numeric_contracts

    descriptor = load_descriptor("SCScaledResetAdaptiveIFNeuron")
    assert descriptor is not None
    schema_name = "sc_scaled_reset_adaptive_if"
    schema = load_schema(schema_name)
    at_its_step = studio_numeric_contracts(
        _studio_default_neuron(descriptor, schema_name, schema, "rk4")
    )
    assert at_its_step["Q16.16"].representable
    finer = studio_numeric_contracts(
        _studio_default_neuron(replace(descriptor, dt=1e-6), schema_name, schema, "rk4")
    )
    assert not finer["Q16.16"].representable
    assert "time step dt=1e-06" in finer["Q16.16"].refusal()
