# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Inference graph capture with source model custody

"""Capture source execution without changing the caller's model or CPU RNG."""

from copy import deepcopy

from torch import fx, nn

from .if_resources import admit_parameter_storage
from .qcfs import QCFSActivation
from .random_custody import preserved_random_state


class ConversionTracer(fx.Tracer):
    """Retain QCFS as an atomic source activation alongside Torch leaf modules."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """Keep actual QCFS invocations instead of tracing threshold validation."""
        return type(m) is QCFSActivation or super().is_leaf_module(m, module_qualified_name)


def capture_inference_graph(model: nn.Module, max_working_bytes: int) -> fx.GraphModule:
    """Trace an independent inference copy and restore every global random generator.

    Parameters
    ----------
    model : nn.Module
        Source network; its tensors, module modes and hook registries are retained.
    max_working_bytes : int
        Positive numeric budget checked before copying source tensor storage.

    Returns
    -------
    GraphModule
        Runnable inference graph retaining every actual forward invocation.

    Raises
    ------
    MemoryError
        If independent source tensor storage exceeds the declared byte budget.
    ValueError
        If the source forward requires untraceable data-dependent control flow.

    Notes
    -----
    User forward code executes during symbolic tracing. Model state is isolated
    by deepcopy; Python, NumPy, Torch CPU and accelerator generator states are
    restored on both paths, serialised against other conversions in the process.
    This does not undo external effects performed by custom user forward code.
    """
    admit_parameter_storage((), (), max_working_bytes)
    tensor_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
    tensor_bytes += sum(t.numel() * t.element_size() for t in model.buffers())
    if 2 * tensor_bytes > max_working_bytes:
        raise MemoryError("source model copy exceeds working byte budget")
    with preserved_random_state():
        copied = nn.Sequential(deepcopy(model)).eval()
        tracer = ConversionTracer()
        try:
            graph = tracer.trace(copied)
        except TypeError as error:
            raise ValueError("source forward is incompatible with single-input tracing") from error
        except fx.proxy.TraceError as error:
            raise ValueError("source model has untraceable data-dependent control flow") from error
        return fx.GraphModule(copied, graph).eval()
