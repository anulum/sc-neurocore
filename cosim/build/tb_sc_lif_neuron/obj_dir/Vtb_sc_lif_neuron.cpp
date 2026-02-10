// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_sc_lif_neuron__pch.h"

//============================================================
// Constructors

Vtb_sc_lif_neuron::Vtb_sc_lif_neuron(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_sc_lif_neuron__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_sc_lif_neuron::Vtb_sc_lif_neuron(const char* _vcname__)
    : Vtb_sc_lif_neuron(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_sc_lif_neuron::~Vtb_sc_lif_neuron() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_sc_lif_neuron___024root___eval_debug_assertions(Vtb_sc_lif_neuron___024root* vlSelf);
#endif  // VL_DEBUG
void Vtb_sc_lif_neuron___024root___eval_static(Vtb_sc_lif_neuron___024root* vlSelf);
void Vtb_sc_lif_neuron___024root___eval_initial(Vtb_sc_lif_neuron___024root* vlSelf);
void Vtb_sc_lif_neuron___024root___eval_settle(Vtb_sc_lif_neuron___024root* vlSelf);
void Vtb_sc_lif_neuron___024root___eval(Vtb_sc_lif_neuron___024root* vlSelf);

void Vtb_sc_lif_neuron::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_sc_lif_neuron::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_sc_lif_neuron___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtb_sc_lif_neuron___024root___eval_static(&(vlSymsp->TOP));
        Vtb_sc_lif_neuron___024root___eval_initial(&(vlSymsp->TOP));
        Vtb_sc_lif_neuron___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtb_sc_lif_neuron___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtb_sc_lif_neuron::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty(); }

uint64_t Vtb_sc_lif_neuron::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_sc_lif_neuron::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtb_sc_lif_neuron___024root___eval_final(Vtb_sc_lif_neuron___024root* vlSelf);

VL_ATTR_COLD void Vtb_sc_lif_neuron::final() {
    Vtb_sc_lif_neuron___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_sc_lif_neuron::hierName() const { return vlSymsp->name(); }
const char* Vtb_sc_lif_neuron::modelName() const { return "Vtb_sc_lif_neuron"; }
unsigned Vtb_sc_lif_neuron::threads() const { return 1; }
void Vtb_sc_lif_neuron::prepareClone() const { contextp()->prepareClone(); }
void Vtb_sc_lif_neuron::atClone() const {
    contextp()->threadPoolpOnClone();
}
