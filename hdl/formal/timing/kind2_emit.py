# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations

from .sby_orchestrator import TimingProperty


def emit_kind2_module(prop: TimingProperty) -> str:
    """Emit a Kind 2 Lustre node for a bounded timing property."""

    bound = prop.bound_cycles
    return f"""-- SC-NeuroCore timing property: {prop.name}
-- Kind: {prop.kind}
node {prop.name}({prop.reset_n}: bool; {prop.trigger}: bool; {prop.response}: bool) returns (ok: bool);
var
  active: bool;
  age: int;
  pre_active: bool;
  pre_age: int;
  violation: bool;
let
  pre_active = false -> pre(active);
  pre_age = 0 -> pre(age);
  active = if not {prop.reset_n} then false
           else if pre_active and {prop.response} then false
           else if (not pre_active) and {prop.trigger} and (not {prop.response}) then true
           else pre_active;
  age = if not {prop.reset_n} then 0
        else if pre_active and {prop.response} then 0
        else if (not pre_active) and {prop.trigger} then 0
        else if pre_active and pre_age < {bound} then pre_age + 1
        else pre_age;
  violation = pre_active and (not {prop.response}) and pre_age >= {bound};
  ok = not violation;
  --%PROPERTY ok;
tel
"""
