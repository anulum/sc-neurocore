# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete adjacent primitive-mapping stages

"""Compose complete primitive mappings through one unchanged intermediate module."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.function_synthesis_dsp_proof import prove_primitive_residual, read_primitive_residual
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_toolchain import SynthesisToolchain


def prove_mapping_stages(
    post: Path,
    intermediate: Path,
    physical: Path,
    input_widths: Mapping[str, int],
    output_width: int,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
) -> None:
    """Prove both complete mappings against the same retained intermediate graph.

    Parameters
    ----------
    post, intermediate, physical : pathlib.Path
        Actual post-DSP, generic pre-LUT and unchanged physical checkpoints.
    input_widths : mapping of str to int
        Every independent original argument and its complete width.
    output_width : int
        Complete original result width.
    tools : SynthesisToolchain
        Existing inspected native profile with the caller's inherited deadline.
    directory : pathlib.Path
        Existing composition directory; both component directories must be new.
    timeout : float
        Native command allowance, clamped by each component to ninety seconds.

    Raises
    ------
    OSError, ValueError
        For altered primitives, observed unmatched cells, incomplete ports or
        any unproved component. Actual acquisitions remain for diagnosis.
    SynthesisToolError
        If a native export or complete equivalence query fails or times out.

    Notes
    -----
    Each component proves its entire original function by primitive induction.
    Both workers use the same immutable intermediate checkpoint and inherited
    deadline. Their native files are written to separate directories.

    """
    with ThreadPoolExecutor(max_workers=2) as workers:
        proofs = [
            workers.submit(
                prove_primitive_residual,
                source,
                target,
                input_widths,
                output_width,
                tools,
                directory / name,
                timeout=timeout,
            )
            for source, target, name in (
                (post, intermediate, "word-mapping-proof"),
                (intermediate, physical, "primitive-proof"),
            )
        ]
        for proof in proofs:
            proof.result()


def read_mapping_stages(
    post: Netlist,
    intermediate: Netlist,
    physical: Netlist,
    input_widths: Mapping[str, int],
    output_width: int,
    tools: SynthesisToolchain,
    directory: Path,
) -> dict[str, object]:
    """Reconstruct both source cuts and validate every retained native component.

    Parameters
    ----------
    post, intermediate, physical : Netlist
        Actual complete checkpoints; the identical intermediate object binds
        the result of the first proof to the source of the second.
    input_widths : mapping of str to int
        Every independent original argument and its complete width.
    output_width : int
        Complete original result width.
    tools : SynthesisToolchain
        Inspected existing executables and qualified technology-library hashes.
    directory : pathlib.Path
        Completed composition artifacts; this reader never writes or reruns tools.

    Returns
    -------
    dict of str to object
        Both complete primitive domains and their qualified native engines.

    Raises
    ------
    OSError, ValueError
        For altered intermediate functions, missing source edges, changed cuts,
        source flows or native evidence without a complete positive verdict.

    """
    return {
        key: read_primitive_residual(
            source, target, input_widths, output_width, tools, directory / name
        )
        for source, target, name, key in (
            (post, intermediate, "word-mapping-proof", "pre_mapping_primitives"),
            (intermediate, physical, "primitive-proof", "primitives"),
        )
    }
