// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_sc_lif_neuron.h for the primary calling header

#include "Vtb_sc_lif_neuron__pch.h"
#include "Vtb_sc_lif_neuron__Syms.h"
#include "Vtb_sc_lif_neuron___024root.h"

VL_INLINE_OPT VlCoroutine Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__1(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x1388ULL, 
                                             nullptr, 
                                             "C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 
                                             55);
        vlSelfRef.tb_sc_lif_neuron__DOT__clk = (1U 
                                                & (~ (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__clk)));
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__act(Vtb_sc_lif_neuron___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_sc_lif_neuron___024root___eval_triggers__act(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_triggers__act\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.tb_sc_lif_neuron__DOT__clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((~ (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__rst_n)) 
                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__rst_n__0)));
    vlSelfRef.__VactTriggered.setBit(2U, vlSelfRef.__VdlySched.awaitingCurrentTime());
    vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__clk__0 
        = vlSelfRef.tb_sc_lif_neuron__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__rst_n__0 
        = vlSelfRef.tb_sc_lif_neuron__DOT__rst_n;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_sc_lif_neuron___024root___dump_triggers__act(vlSelf);
    }
#endif
}
