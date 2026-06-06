# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - UltraScale+ timing report extraction

if {![info exists TOP]} { error "TOP must be set before sourcing timing.tcl" }
if {![info exists OUT_DIR]} { set OUT_DIR out/ultrascale_plus }
file mkdir $OUT_DIR
report_timing_summary -file $OUT_DIR/${TOP}_timing.rpt
report_utilization -file $OUT_DIR/${TOP}_utilisation.rpt
