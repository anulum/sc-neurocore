# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM benchmark generation orchestration

"""Compose parsed RTL, configuration, emitters, and benchmark artifacts."""

from __future__ import annotations

from typing import List, Optional

from sc_neurocore.uvm_gen._benchmark import UVMBenchmark
from sc_neurocore.uvm_gen._component_emitters import (
    _emit_agent,
    _emit_coverage,
    _emit_driver,
    _emit_env,
    _emit_monitor,
    _emit_scoreboard,
    _emit_sequence,
    _emit_transaction,
)
from sc_neurocore.uvm_gen._config import (
    CoverageSpec,
    FormalLink,
    ScoreboardConfig,
    StimulusConfig,
)
from sc_neurocore.uvm_gen._harness_emitters import (
    _emit_bind,
    _emit_makefile,
    _emit_regression_list,
    _emit_sby,
    _emit_top,
    _filelist,
    _generate_formal_links,
)
from sc_neurocore.uvm_gen._rtl import RTLModule


class UVMGenerator:
    """Generates complete UVM verification IP from an RTL module spec."""

    def __init__(
        self,
        stimulus: Optional[StimulusConfig] = None,
        coverage: Optional[CoverageSpec] = None,
        scoreboard: Optional[ScoreboardConfig] = None,
    ):
        self.stimulus = stimulus or StimulusConfig()
        self.coverage = coverage or CoverageSpec()
        self.scoreboard = scoreboard or ScoreboardConfig()

    def generate(self, rtl: RTLModule) -> UVMBenchmark:
        """Generate the complete UVM testbench for an RTL module."""
        return UVMBenchmark(
            module_name=rtl.name,
            transaction_sv=self._emit_transaction(rtl),
            sequence_sv=self._emit_sequence(rtl),
            driver_sv=self._emit_driver(rtl),
            monitor_sv=self._emit_monitor(rtl),
            scoreboard_sv=self._emit_scoreboard(rtl),
            coverage_sv=self._emit_coverage(rtl),
            agent_sv=self._emit_agent(rtl),
            env_sv=self._emit_env(rtl),
            top_sv=self._emit_top(rtl),
            sby_config=self._emit_sby(rtl),
            bind_sv=self._emit_bind(rtl),
            makefile=self._emit_makefile(rtl),
            regression_list=self._emit_regression_list(rtl),
            filelist=self._filelist(rtl),
        )

    def generate_multi(self, modules: List[RTLModule]) -> List[UVMBenchmark]:
        """Generate UVM testbenches for multiple modules."""
        return [self.generate(rtl) for rtl in modules]

    def _emit_transaction(self, rtl: RTLModule) -> str:
        return _emit_transaction(rtl, self.stimulus)

    def _emit_sequence(self, rtl: RTLModule) -> str:
        return _emit_sequence(rtl, self.stimulus)

    def _emit_driver(self, rtl: RTLModule) -> str:
        return _emit_driver(rtl)

    def _emit_monitor(self, rtl: RTLModule) -> str:
        return _emit_monitor(rtl)

    def _emit_scoreboard(self, rtl: RTLModule) -> str:
        return _emit_scoreboard(rtl, self.scoreboard)

    def _emit_coverage(self, rtl: RTLModule) -> str:
        return _emit_coverage(rtl, self.coverage)

    def _emit_agent(self, rtl: RTLModule) -> str:
        return _emit_agent(rtl)

    def _emit_env(self, rtl: RTLModule) -> str:
        return _emit_env(rtl)

    def _emit_top(self, rtl: RTLModule) -> str:
        return _emit_top(rtl)

    def _emit_sby(self, rtl: RTLModule) -> str:
        return _emit_sby(rtl)

    def _filelist(self, rtl: RTLModule) -> List[str]:
        return _filelist(rtl)

    def _emit_bind(self, rtl: RTLModule) -> str:
        return _emit_bind(rtl)

    def _emit_makefile(self, rtl: RTLModule, sim: str = "vcs") -> str:
        return _emit_makefile(rtl, sim)

    def _emit_regression_list(self, rtl: RTLModule) -> str:
        return _emit_regression_list(rtl)

    def generate_formal_links(self, rtl: RTLModule) -> List[FormalLink]:
        """Generate formal-to-dynamic links for existing SymbiYosys modules."""
        return _generate_formal_links(rtl)
