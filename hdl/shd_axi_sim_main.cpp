// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Timed SHD AXI simulation with RTL line coverage
#include "Vtb_sc_shd_axi_wrapper.h"
#include "verilated.h"
#include "verilated_cov.h"

int main(int argc, char** argv) {
    VerilatedContext context;
    context.commandArgs(argc, argv);
    Vtb_sc_shd_axi_wrapper model{&context};
    while (!context.gotFinish()) {
        model.eval();
        if (context.gotFinish()) break;
        if (!model.eventsPending()) return 1;
        context.time(model.nextTimeSlot());
    }
    model.final();
    context.coveragep()->write("coverage.dat");
    return 0;
}
