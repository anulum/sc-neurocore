// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_sc_lif_neuron.h for the primary calling header

#include "Vtb_sc_lif_neuron__pch.h"
#include "Vtb_sc_lif_neuron__Syms.h"
#include "Vtb_sc_lif_neuron___024root.h"

void Vtb_sc_lif_neuron___024root___ctor_var_reset(Vtb_sc_lif_neuron___024root* vlSelf);

Vtb_sc_lif_neuron___024root::Vtb_sc_lif_neuron___024root(Vtb_sc_lif_neuron__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtb_sc_lif_neuron___024root___ctor_var_reset(this);
}

void Vtb_sc_lif_neuron___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtb_sc_lif_neuron___024root::~Vtb_sc_lif_neuron___024root() {
}
