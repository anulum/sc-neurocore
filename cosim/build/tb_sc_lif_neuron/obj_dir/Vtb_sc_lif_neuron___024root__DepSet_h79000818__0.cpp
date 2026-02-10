// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_sc_lif_neuron.h for the primary calling header

#include "Vtb_sc_lif_neuron__pch.h"
#include "Vtb_sc_lif_neuron___024root.h"

VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___eval_initial__TOP(Vtb_sc_lif_neuron___024root* vlSelf);
VlCoroutine Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__0(Vtb_sc_lif_neuron___024root* vlSelf);
VlCoroutine Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__1(Vtb_sc_lif_neuron___024root* vlSelf);

void Vtb_sc_lif_neuron___024root___eval_initial(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_initial\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtb_sc_lif_neuron___024root___eval_initial__TOP(vlSelf);
    Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

VL_INLINE_OPT VlCoroutine Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__0(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ tb_sc_lif_neuron__DOT__stim_file;
    tb_sc_lif_neuron__DOT__stim_file = 0;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__result_file;
    tb_sc_lif_neuron__DOT__result_file = 0;
    IData/*31:0*/ tb_sc_lif_neuron__DOT__step;
    tb_sc_lif_neuron__DOT__step = 0;
    VlWide<3>/*95:0*/ __Vtemp_1;
    VlWide<5>/*159:0*/ __Vtemp_2;
    // Body
    __Vtemp_1[0U] = 0x2e747874U;
    __Vtemp_1[1U] = 0x6d756c69U;
    __Vtemp_1[2U] = 0x737469U;
    tb_sc_lif_neuron__DOT__stim_file = VL_FOPEN_NN(
                                                   VL_CVT_PACK_STR_NW(3, __Vtemp_1)
                                                   , 
                                                   std::string{"r"});
    ;
    if (VL_UNLIKELY(((0U == tb_sc_lif_neuron__DOT__stim_file)))) {
        VL_WRITEF_NX("ERROR: Cannot open stimuli.txt\n",0);
        VL_FINISH_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 71, "");
    }
    __Vtemp_2[0U] = 0x2e747874U;
    __Vtemp_2[1U] = 0x696c6f67U;
    __Vtemp_2[2U] = 0x5f766572U;
    __Vtemp_2[3U] = 0x756c7473U;
    __Vtemp_2[4U] = 0x726573U;
    tb_sc_lif_neuron__DOT__result_file = VL_FOPEN_NN(
                                                     VL_CVT_PACK_STR_NW(5, __Vtemp_2)
                                                     , 
                                                     std::string{"w"});
    ;
    if (VL_UNLIKELY(((0U == tb_sc_lif_neuron__DOT__result_file)))) {
        VL_WRITEF_NX("ERROR: Cannot open results_verilog.txt for writing\n",0);
        VL_FINISH_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 77, "");
    }
    vlSelfRef.tb_sc_lif_neuron__DOT__rst_n = 0U;
    vlSelfRef.tb_sc_lif_neuron__DOT__leak_k = 0U;
    vlSelfRef.tb_sc_lif_neuron__DOT__gain_k = 0U;
    vlSelfRef.tb_sc_lif_neuron__DOT__I_t = 0U;
    vlSelfRef.tb_sc_lif_neuron__DOT__noise_in = 0U;
    co_await vlSelfRef.__VtrigSched_h0bce3b11__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_sc_lif_neuron.clk)", 
                                                         "C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 
                                                         88);
    co_await vlSelfRef.__VtrigSched_h0bce3b11__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_sc_lif_neuron.clk)", 
                                                         "C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 
                                                         89);
    vlSelfRef.tb_sc_lif_neuron__DOT__rst_n = 1U;
    tb_sc_lif_neuron__DOT__step = 0U;
    while (VL_GTS_III(32, 0x3e8U, tb_sc_lif_neuron__DOT__step)) {
        vlSelfRef.tb_sc_lif_neuron__DOT__scan_ret = VL_FSCANF_INX(tb_sc_lif_neuron__DOT__stim_file,"%# %# %# %#\n",0,
                                                                  32,
                                                                  &(vlSelfRef.tb_sc_lif_neuron__DOT__stim_leak),
                                                                  32,
                                                                  &(vlSelfRef.tb_sc_lif_neuron__DOT__stim_gain),
                                                                  32,
                                                                  &(vlSelfRef.tb_sc_lif_neuron__DOT__stim_it),
                                                                  32,
                                                                  &(vlSelfRef.tb_sc_lif_neuron__DOT__stim_noise)) ;
        if ((4U != vlSelfRef.tb_sc_lif_neuron__DOT__scan_ret)) {
            VL_WRITEF_NX("INFO: End of stimuli at step %0d\n",0,
                         32,tb_sc_lif_neuron__DOT__step);
            tb_sc_lif_neuron__DOT__step = 0x3e8U;
        } else {
            vlSelfRef.tb_sc_lif_neuron__DOT__leak_k 
                = (0xffffU & vlSelfRef.tb_sc_lif_neuron__DOT__stim_leak);
            vlSelfRef.tb_sc_lif_neuron__DOT__gain_k 
                = (0xffffU & vlSelfRef.tb_sc_lif_neuron__DOT__stim_gain);
            vlSelfRef.tb_sc_lif_neuron__DOT__I_t = 
                (0xffffU & vlSelfRef.tb_sc_lif_neuron__DOT__stim_it);
            vlSelfRef.tb_sc_lif_neuron__DOT__noise_in 
                = (0xffffU & vlSelfRef.tb_sc_lif_neuron__DOT__stim_noise);
            co_await vlSelfRef.__VtrigSched_h0bce3b11__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge tb_sc_lif_neuron.clk)", 
                                                                 "C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 
                                                                 105);
            co_await vlSelfRef.__VdlySched.delay(0x3e8ULL, 
                                                 nullptr, 
                                                 "C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 
                                                 107);
            VL_FWRITEF_NX(tb_sc_lif_neuron__DOT__result_file,"%0# %0d\n",0,
                          1,vlSelfRef.tb_sc_lif_neuron__DOT__spike_out,
                          16,(IData)(vlSelfRef.tb_sc_lif_neuron__DOT__v_out));
        }
        tb_sc_lif_neuron__DOT__step = ((IData)(1U) 
                                       + tb_sc_lif_neuron__DOT__step);
    }
    VL_FCLOSE_I(tb_sc_lif_neuron__DOT__stim_file); VL_FCLOSE_I(tb_sc_lif_neuron__DOT__result_file); VL_WRITEF_NX("Co-simulation complete. Results written to results_verilog.txt\n",0);
    VL_FINISH_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 115, "");
}

void Vtb_sc_lif_neuron___024root___act_comb__TOP__0(Vtb_sc_lif_neuron___024root* vlSelf);

void Vtb_sc_lif_neuron___024root___eval_act(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_act\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((5ULL & vlSelfRef.__VactTriggered.word(0U))) {
        Vtb_sc_lif_neuron___024root___act_comb__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vtb_sc_lif_neuron___024root___act_comb__TOP__0(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___act_comb__TOP__0\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_next 
        = (0xffffU & ((IData)(vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg) 
                      + (VL_SHIFTRS_III(16,32,32, VL_MULS_III(32, 
                                                              (- 
                                                               VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg))), 
                                                              VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__leak_k))), 8U) 
                         + (VL_SHIFTRS_III(16,32,32, 
                                           VL_MULS_III(32, 
                                                       VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__I_t)), 
                                                       VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__gain_k))), 8U) 
                            + (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__noise_in)))));
}

void Vtb_sc_lif_neuron___024root___nba_sequent__TOP__0(Vtb_sc_lif_neuron___024root* vlSelf);

void Vtb_sc_lif_neuron___024root___eval_nba(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_nba\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtb_sc_lif_neuron___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((7ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtb_sc_lif_neuron___024root___act_comb__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vtb_sc_lif_neuron___024root___nba_sequent__TOP__0(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___nba_sequent__TOP__0\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (vlSelfRef.tb_sc_lif_neuron__DOT__rst_n) {
        if ((0U < vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter)) {
            vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter 
                = (vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter 
                   - (IData)(1U));
            vlSelfRef.tb_sc_lif_neuron__DOT__spike_out = 0U;
            vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg = 0U;
            vlSelfRef.tb_sc_lif_neuron__DOT__v_out = 0U;
        } else if (VL_LTES_III(16, 0x100U, (IData)(vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_next))) {
            vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter = 2U;
            vlSelfRef.tb_sc_lif_neuron__DOT__spike_out = 1U;
            vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg = 0U;
            vlSelfRef.tb_sc_lif_neuron__DOT__v_out = 0U;
        } else {
            vlSelfRef.tb_sc_lif_neuron__DOT__spike_out = 0U;
            vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg 
                = vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_next;
            vlSelfRef.tb_sc_lif_neuron__DOT__v_out 
                = vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_next;
        }
    } else {
        vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__refractory_counter = 0U;
        vlSelfRef.tb_sc_lif_neuron__DOT__dut__DOT__v_reg = 0U;
        vlSelfRef.tb_sc_lif_neuron__DOT__v_out = 0U;
        vlSelfRef.tb_sc_lif_neuron__DOT__spike_out = 0U;
    }
}

void Vtb_sc_lif_neuron___024root___timing_resume(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___timing_resume\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_h0bce3b11__0.resume(
                                                   "@(posedge tb_sc_lif_neuron.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtb_sc_lif_neuron___024root___timing_commit(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___timing_commit\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((! (1ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_h0bce3b11__0.commit(
                                                   "@(posedge tb_sc_lif_neuron.clk)");
    }
}

void Vtb_sc_lif_neuron___024root___eval_triggers__act(Vtb_sc_lif_neuron___024root* vlSelf);

bool Vtb_sc_lif_neuron___024root___eval_phase__act(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_phase__act\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<3> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb_sc_lif_neuron___024root___eval_triggers__act(vlSelf);
    Vtb_sc_lif_neuron___024root___timing_commit(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vtb_sc_lif_neuron___024root___timing_resume(vlSelf);
        Vtb_sc_lif_neuron___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_sc_lif_neuron___024root___eval_phase__nba(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_phase__nba\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtb_sc_lif_neuron___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__nba(Vtb_sc_lif_neuron___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_sc_lif_neuron___024root___dump_triggers__act(Vtb_sc_lif_neuron___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_sc_lif_neuron___024root___eval(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY(((0x64U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_sc_lif_neuron___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 16, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtb_sc_lif_neuron___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("C:\\Users\\forti\\OneDrive\\Documents\\aaa_God_of_the_Math_Collection\\03_CODE\\sc-neurocore\\hdl\\tb_sc_lif_neuron.v", 16, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vtb_sc_lif_neuron___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vtb_sc_lif_neuron___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtb_sc_lif_neuron___024root___eval_debug_assertions(Vtb_sc_lif_neuron___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_sc_lif_neuron___024root___eval_debug_assertions\n"); );
    Vtb_sc_lif_neuron__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
