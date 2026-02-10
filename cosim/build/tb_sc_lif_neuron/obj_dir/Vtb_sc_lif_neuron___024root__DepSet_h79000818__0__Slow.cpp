// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_sc_lif_neuron.h for the primary calling header

#include "Vtb_sc_lif_neuron__pch.h"
#include "Vtb_sc_lif_neuron___024root.h"

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_static(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_static\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__clk__0 
        = vlSelfRef.tb_sc_lif_neuron__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__rst_n__0 
        = vlSelfRef.tb_sc_lif_neuron__DOT__rst_n;
}

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_initial__TOP(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_initial__TOP\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_sc_lif_neuron__DOT__clk = 0U;
}

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_final(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_final\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__stl(Vtb_sc_lif_neuron___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_sc_lif_neuron___024root___eval_phase__stl(Vtb_sc_lif_neuron___024root* vlSelf);

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_settle(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_settle\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY(((0x64U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtb_sc_lif_neuron___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 16, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtb_sc_lif_neuron___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelfRef.__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__stl(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___dump_triggers__stl\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VstlTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vtb_sc_lif_neuron___024root___act_comb__TOP__0(Vtb_sc_lif_neuron___024root* vlSelf);

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_stl(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_stl\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        Vtb_sc_lif_neuron___024root___act_comb__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_triggers__stl(Vtb_sc_lif_neuron___024root* vlSelf);

VL_ATTR_COLD bool Vtb_sc_lif_neuron___024root___eval_phase__stl(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_phase__stl\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtb_sc_lif_neuron___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelfRef.__VstlTriggered.any();
    if (__VstlExecute) {
        Vtb_sc_lif_neuron___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__act(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___dump_triggers__act\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge tb_sc_lif_neuron.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge tb_sc_lif_neuron.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__nba(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___dump_triggers__nba\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge tb_sc_lif_neuron.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge tb_sc_lif_neuron.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___ctor_var_reset(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___ctor_var_reset\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->tb_sc_lif_neuron__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18210711391708579224ull);
    vlSelf->tb_sc_lif_neuron__DOT__rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12931296623397030223ull);
    vlSelf->tb_sc_lif_neuron__DOT__leak_k = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 18391054631912439451ull);
    vlSelf->tb_sc_lif_neuron__DOT__gain_k = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14581707503114749578ull);
    vlSelf->tb_sc_lif_neuron__DOT__I_t = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 678464454322609604ull);
    vlSelf->tb_sc_lif_neuron__DOT__noise_in = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3092156382626811389ull);
    vlSelf->tb_sc_lif_neuron__DOT__spike_out = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10713622244114453478ull);
    vlSelf->tb_sc_lif_neuron__DOT__v_out = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 274328075813363471ull);
    vlSelf->tb_sc_lif_neuron__DOT__scan_ret = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 37402584499366047ull);
    vlSelf->tb_sc_lif_neuron__DOT__stim_leak = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1219985514719032156ull);
    vlSelf->tb_sc_lif_neuron__DOT__stim_gain = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8696888370092718059ull);
    vlSelf->tb_sc_lif_neuron__DOT__stim_it = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1568895685693623855ull);
    vlSelf->tb_sc_lif_neuron__DOT__stim_noise = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12641601537940674976ull);
    vlSelf->tb_sc_lif_neuron__DOT__dut__DOT__v_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4201278690266330133ull);
    vlSelf->tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4925541058710249477ull);
    vlSelf->tb_sc_lif_neuron__DOT__dut__DOT__v_next = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15559160957467419140ull);
    vlSelf->__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5985466876465060082ull);
    vlSelf->__Vtrigprevexpr___TOP__tb_sc_lif_neuron__DOT__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17376595070074821846ull);
}
