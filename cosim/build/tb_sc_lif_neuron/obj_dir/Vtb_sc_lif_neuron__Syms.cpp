// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtb_sc_lif_neuron__pch.h"
#include "Vtb_sc_lif_neuron.h"
#include "Vtb_sc_lif_neuron___024root.h"

// FUNCTIONS
Vtb_sc_lif_neuron__Syms::~Vtb_sc_lif_neuron__Syms()
{
}

Vtb_sc_lif_neuron__Syms::Vtb_sc_lif_neuron__Syms(VerilatedContext* contextp, const char* namep, Vtb_sc_lif_neuron* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
        // Check resources
        Verilated::stackCheck(62);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
}
