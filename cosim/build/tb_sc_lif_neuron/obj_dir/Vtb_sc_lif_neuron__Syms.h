// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTB_SC_LIF_NEURON__SYMS_H_
#define VERILATED_VTB_SC_LIF_NEURON__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vtb_sc_lif_neuron.h"

// INCLUDE MODULE CLASSES
#include "Vtb_sc_lif_neuron___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vtb_sc_lif_neuron__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtb_sc_lif_neuron* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtb_sc_lif_neuron___024root    TOP;

    // CONSTRUCTORS
    Vtb_sc_lif_neuron__Syms(VerilatedContext* contextp, const char* namep, Vtb_sc_lif_neuron* modelp);
    ~Vtb_sc_lif_neuron__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
