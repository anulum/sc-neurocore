// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_sc_lif_neuron.h for the primary calling header

#ifndef VERILATED_VTB_SC_LIF_NEURON___024ROOT_H_
#define VERILATED_VTB_SC_LIF_NEURON___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_sc_lif_neuron__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_sc_lif_neuron___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_sc_lif_neuron__DOT__clk;
    CData/*0:0*/ tb_sc_lif_neuron__DOT__rst_n;
    CData/*0:0*/ tb_sc_lif_neuron__DOT__spike_out;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__rst_n__0;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__leak_k;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__gain_k;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__I_t;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__noise_in;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__v_out;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__dut__DOT__v_reg;
    SData/*15:0*/ tb_sc_lif_neuron__DOT__dut__DOT__v_next;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__scan_ret;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__stim_leak;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__stim_gain;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__stim_it;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__stim_noise;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter;
    IData/*31:0*/ __VactIterCount;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_h0bce3b11__0;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<3> __VactTriggered;
    VlTriggerVec<3> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtb_sc_lif_neuron__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtb_sc_lif_neuron___024root(Vtb_sc_lif_neuron__Syms* symsp, const char* v__name);
    ~Vtb_sc_lif_neuron___024root();
    VL_UNCOPYABLE(Vtb_sc_lif_neuron___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
