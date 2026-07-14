# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated UVM benchmark artifact contract

"""Package the deterministic artifacts emitted for one RTL module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UVMBenchmark:
    """Complete generated UVM testbench."""

    module_name: str
    transaction_sv: str
    sequence_sv: str
    driver_sv: str
    monitor_sv: str
    scoreboard_sv: str
    coverage_sv: str
    agent_sv: str
    env_sv: str
    top_sv: str
    sby_config: str
    bind_sv: str = ""
    makefile: str = ""
    regression_list: str = ""
    filelist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        """Return generated artefacts keyed by their output filenames."""
        artifacts = {
            f"{self.module_name}_transaction.sv": self.transaction_sv,
            f"{self.module_name}_sequence.sv": self.sequence_sv,
            f"{self.module_name}_driver.sv": self.driver_sv,
            f"{self.module_name}_monitor.sv": self.monitor_sv,
            f"{self.module_name}_scoreboard.sv": self.scoreboard_sv,
            f"{self.module_name}_coverage.sv": self.coverage_sv,
            f"{self.module_name}_agent.sv": self.agent_sv,
            f"{self.module_name}_env.sv": self.env_sv,
            f"tb_{self.module_name}_top.sv": self.top_sv,
            f"{self.module_name}_verify.sby": self.sby_config,
        }
        if self.bind_sv:
            artifacts[f"{self.module_name}_bind.sv"] = self.bind_sv
        if self.makefile:
            artifacts["Makefile"] = self.makefile
        if self.regression_list:
            artifacts["regression.list"] = self.regression_list
        return artifacts
