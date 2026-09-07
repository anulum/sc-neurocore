# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A declared variable carries the role its profile states

"""A run records the role its own profile assigns, including under two names.

Two defects met in one variable. The committed descriptor calls the Brunel-Wang
and Compte-WM refractory register `ref_remaining` while the canonical schema
calls it `refractory_time`, and the role join matched by name alone, so the
variable came back `unassigned` on every run — carrying none of the role its
profile states. Separately, neither schema authored a `[profile]` section at
all, so every one of their eleven and thirty-one state variables resolved as
`biological`, including the ones each schema's own `state_scope_note` calls
deterministic lowering registers. A role split where everything is biological
carries no information.

Both schemas now author the split their note already stated, and the join
carries an explicit per-identity record of the descriptor's word for the
refractory register. The schema side is never repeated: the profile names it
itself, so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.universal_dsl import load_schema
from sc_neurocore.studio.state_layout import _REFRACTORY_ALIASES, declared_state

#: The two identities whose descriptor and schema name one variable differently.
DIVERGENT = {"BrunelWangNeuron": "brunel_wang", "CompteWMNeuron": "compte_wm"}

#: Models still carrying an unassigned declared variable, because their schema
#: authors no profile section. Pinned rather than sampled: the gap shrinks only
#: by a deliberate test change, never by accident. It was 106 before the two
#: schemas below were authored.
UNASSIGNED_MODEL_COUNT = 104

#: Name fragments the schemas' own notes call deterministic lowering registers.
LOWERING_FRAGMENTS = ("phase", "latched", "k1", "k2", "mid", "pre_")


def _roles(class_name: str) -> dict[str, str]:
    """Return the declared role of every variable a run of *class_name* records."""
    return {variable.name: variable.role for variable in declared_state(class_name)[2]}


class TestTheRefractoryRegisterCarriesItsRole:
    @pytest.mark.parametrize("class_name", sorted(DIVERGENT))
    def test_it_is_no_longer_unassigned(self, class_name: str) -> None:
        """The defect: the join matched by name and the names differ."""
        assert _roles(class_name)["ref_remaining"] == "auxiliary"

    @pytest.mark.parametrize("class_name", sorted(DIVERGENT))
    def test_nothing_the_model_declares_is_unassigned(self, class_name: str) -> None:
        """A recorded variable with no role is a run that cannot say what it holds."""
        assert "unassigned" not in set(_roles(class_name).values())


class TestTheAliasIsPerIdentityAndCannotGoStale:
    def test_every_alias_names_a_real_divergence(self) -> None:
        """A table entry for a name that matches would be dead weight."""
        for class_name, descriptor_name in _REFRACTORY_ALIASES.items():
            stem = DIVERGENT[class_name]
            schema_name = resolve_profile(
                load_schema(stem), stem=stem
            ).numerical.event.refractory_register
            assert schema_name
            assert schema_name != descriptor_name

    def test_the_table_holds_only_the_measured_divergences(self) -> None:
        """A new divergence must be added deliberately, not matched by leftovers."""
        assert set(_REFRACTORY_ALIASES) == set(DIVERGENT)

    def test_the_schema_side_is_not_duplicated_in_the_table(self) -> None:
        """The profile names it; recording it twice is how two records drift."""
        assert set(_REFRACTORY_ALIASES.values()) == {"ref_remaining"}


class TestTheRoleSplitCarriesInformation:
    @pytest.mark.parametrize("stem", sorted(DIVERGENT.values()))
    def test_the_profile_resolves_without_a_single_problem(self, stem: str) -> None:
        """An authored section that contradicts its schema is worse than none."""
        assert resolve_profile(load_schema(stem), stem=stem).problems == ()

    @pytest.mark.parametrize("stem", sorted(DIVERGENT.values()))
    def test_both_roles_occur(self, stem: str) -> None:
        """The defect: every variable resolved biological, which says nothing."""
        profile = resolve_profile(load_schema(stem), stem=stem)
        assert profile.scientific.biological_state
        assert profile.numerical.auxiliary_registers

    @pytest.mark.parametrize("stem", sorted(DIVERGENT.values()))
    def test_the_lowering_registers_are_auxiliary(self, stem: str) -> None:
        """Held to the schema's own note, so the transcription cannot drift."""
        profile = resolve_profile(load_schema(stem), stem=stem)
        biological = {variable.name for variable in profile.scientific.biological_state}
        misfiled = sorted(
            name for name in biological if any(fragment in name for fragment in LOWERING_FRAGMENTS)
        )
        assert misfiled == []

    @pytest.mark.parametrize("stem", sorted(DIVERGENT.values()))
    def test_the_refractory_register_is_auxiliary_as_the_contract_requires(self, stem: str) -> None:
        """The contract refuses a biological refractory register; hold to that."""
        profile = resolve_profile(load_schema(stem), stem=stem)
        auxiliary = {variable.name for variable in profile.numerical.auxiliary_registers}
        assert profile.numerical.event.refractory_register in auxiliary


class TestTheRestOfTheCatalogueIsUnchanged:
    def test_the_unassigned_census_is_the_pinned_one(self) -> None:
        """Most schemas author no profile; that gap is tracked, not hidden."""
        unassigned = sorted(
            name for name in _CLASS_TO_MODULE if "unassigned" in set(_roles(name).values())
        )
        assert len(unassigned) == UNASSIGNED_MODEL_COUNT
        assert not set(DIVERGENT) & set(unassigned)
