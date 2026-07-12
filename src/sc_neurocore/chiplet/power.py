# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet voltage-island and power-gating RTL

"""Power-domain ownership and sequenced isolation-controller emission."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from sc_neurocore.chiplet._sv import SPDX_HEADER


@dataclass
class PowerDomain:
    """Describe a voltage island spanning one or more package dies."""

    domain_id: int
    die_ids: list[int] = field(default_factory=list)
    voltage_mv: int = 800
    is_active: bool = True

    def __post_init__(self) -> None:
        """Validate domain identity, die ownership, and voltage boundaries."""
        if self.domain_id < 0:
            raise ValueError("domain_id must be >= 0")
        if not self.die_ids:
            raise ValueError("die_ids must contain at least one die")
        if any(die_id < 0 or die_id >= 64 for die_id in self.die_ids):
            raise ValueError("die_ids must be in the range [0, 63]")
        if len(set(self.die_ids)) != len(self.die_ids):
            raise ValueError("die_ids must not contain duplicates")
        if self.voltage_mv <= 0:
            raise ValueError("voltage_mv must be > 0")

    @property
    def is_gated(self) -> bool:
        """Return whether the domain is currently marked inactive."""
        return not self.is_active

    @property
    def die_mask(self) -> int:
        """Return the 64-bit ownership mask used by generated RTL."""
        mask = 0
        for die_id in self.die_ids:
            mask |= 1 << die_id
        return mask


@dataclass
class PowerDomainMap:
    """Maintain non-overlapping voltage-domain ownership for package dies."""

    domains: list[PowerDomain] = field(default_factory=list)

    def add_domain(self, domain: PowerDomain) -> None:
        """Add a domain after rejecting duplicate die ownership."""
        assigned = {die_id for existing in self.domains for die_id in existing.die_ids}
        overlap = assigned.intersection(domain.die_ids)
        if overlap:
            die_list = ", ".join(str(die_id) for die_id in sorted(overlap))
            raise ValueError(f"die_ids already assigned to a power domain: {die_list}")
        self.domains.append(domain)

    def domain_for_die(self, die_id: int) -> PowerDomain | None:
        """Return the domain owning ``die_id``, or ``None`` when unassigned."""
        return next((domain for domain in self.domains if die_id in domain.die_ids), None)

    def active_dies(self) -> list[int]:
        """Return sorted dies belonging to active domains."""
        return sorted(
            die_id for domain in self.domains if domain.is_active for die_id in domain.die_ids
        )

    def gated_dies(self) -> list[int]:
        """Return sorted dies belonging to inactive domains."""
        return sorted(
            die_id for domain in self.domains if not domain.is_active for die_id in domain.die_ids
        )


def emit_power_gating_sv(domain: PowerDomain) -> str:
    """Emit a sequenced isolation and switch controller for ``domain``."""
    die_list = ", ".join(str(die_id) for die_id in domain.die_ids)
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Power domain {domain.domain_id} controller
// Dies: [{die_list}]
// Voltage: {domain.voltage_mv} mV

module sc_chiplet_pwr_domain_{domain.domain_id} #(
    parameter DOMAIN_ID = {domain.domain_id},
    parameter DIE_COUNT = {len(domain.die_ids)},
    parameter [63:0] DIE_MASK = 64'h{domain.die_mask:016X},
    parameter VOLTAGE_MV = {domain.voltage_mv}
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    output reg  domain_active,
    output reg  isolation_en,
    output reg  power_switch_en
);

    localparam PWR_OFF     = 2'd0;
    localparam PWR_ON      = 2'd1;
    localparam PWR_ISOLATE = 2'd2;
    localparam PWR_RESTORE = 2'd3;
    localparam ISO_CYCLES  = 4;
    localparam RESTORE_CYCLES = 4;

    reg [1:0] state;
    reg [2:0] iso_count;
    reg [2:0] restore_count;

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= PWR_OFF;
            iso_count     <= 0;
            restore_count <= 0;
            domain_active <= 1'b0;
            isolation_en  <= 1'b1;
            power_switch_en <= 1'b0;
        end else begin
            case (state)
                PWR_OFF: begin
                    domain_active <= 1'b0;
                    isolation_en <= 1'b1;
                    power_switch_en <= 1'b0;
                    iso_count <= 0;
                    restore_count <= 0;
                    if (enable)
                        state <= PWR_RESTORE;
                end
                PWR_RESTORE: begin
                    power_switch_en <= 1'b1;
                    isolation_en <= 1'b1;
                    domain_active <= 1'b0;
                    iso_count <= 0;
                    if (restore_count == RESTORE_CYCLES[2:0] - 1'b1) begin
                        restore_count <= 0;
                        state <= PWR_ON;
                    end else begin
                        restore_count <= restore_count + 1'b1;
                    end
                end
                PWR_ON: begin
                    domain_active <= 1'b1;
                    isolation_en <= 1'b0;
                    power_switch_en <= 1'b1;
                    iso_count <= 0;
                    restore_count <= 0;
                    if (!enable)
                        state <= PWR_ISOLATE;
                end
                PWR_ISOLATE: begin
                    domain_active <= 1'b1;
                    isolation_en <= 1'b1;
                    power_switch_en <= 1'b1;
                    restore_count <= 0;
                    if (iso_count == ISO_CYCLES[2:0]) begin
                        domain_active <= 1'b0;
                        power_switch_en <= 1'b0;
                        state <= PWR_OFF;
                    end else begin
                        iso_count <= iso_count + 1'b1;
                    end
                end
                default: state <= PWR_OFF;
            endcase
        end
    end

endmodule
""")


__all__ = ["PowerDomain", "PowerDomainMap", "emit_power_gating_sv"]
