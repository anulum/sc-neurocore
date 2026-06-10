# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Digital twin shadow generator

"""Generate software digital twins that mirror deployed hardware state."""

from __future__ import annotations


def generate_digital_twin(
    module_name: str,
    equations: dict[str, str],
    profile_name: str,
) -> str:
    """Generate a Python digital twin that mirrors deployed hardware."""
    vars_list = list(equations.keys())
    lines = [
        f'"""Digital twin for {module_name} targeting {profile_name}."""',
        "",
        f"class {module_name.title().replace('_', '')}Twin:",
        '    """Software shadow of deployed hardware state."""',
        "",
        "    def __init__(self):",
    ]
    for v in vars_list:
        lines.append(f"        self.{v} = 0.0")
    lines.extend(
        [
            "        self.cycle = 0",
            "",
            "    def step(self, inputs: dict[str, float]) -> dict[str, float]:",
            '        """Execute one timestep, mirroring hardware state."""',
        ]
    )
    for v, expr in equations.items():
        lines.append(f"        # {v} = {expr}")
        lines.append(f"        self.{v} = inputs.get('{v}', self.{v})")
    lines.extend(
        [
            "        self.cycle += 1",
            f"        return {{{', '.join(repr(v) + ': self.' + v for v in vars_list)}}}",
            "",
            "    def compare(self, hw_state: dict[str, float]) -> dict[str, float]:",
            '        """Compare twin state against hardware telemetry."""',
            "        return {k: abs(getattr(self, k, 0) - hw_state.get(k, 0))",
            f"                for k in {vars_list!r}}}",
        ]
    )

    return "\n".join(lines)
