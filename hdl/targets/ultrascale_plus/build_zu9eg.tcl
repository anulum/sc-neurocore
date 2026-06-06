# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - ZU9EG Vivado batch entry

if {$argc != 1} { error "usage: vivado -mode batch -source build_zu9eg.tcl -tclargs generated_project.tcl" }
set PROJECT_TCL [lindex $argv 0]
source $PROJECT_TCL
