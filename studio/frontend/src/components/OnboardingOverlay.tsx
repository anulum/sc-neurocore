// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useState, useEffect } from "react";

const STORAGE_KEY = "sc-studio-onboarding-dismissed";

const steps = [
  {
    title: "Welcome to SC-NeuroCore Studio",
    text: "Design, train, compile, and deploy spiking neural networks from your browser. This quick tour shows you the key features.",
  },
  {
    title: "Model Browser",
    text: "Browse 118 neuron models by category. Click any model to load it — the trace updates live. Adjust parameters with sliders in the left panel.",
  },
  {
    title: "ODE Mode",
    text: "Switch to ODE mode (top toggle) to write custom neuron equations. The Monaco editor provides syntax highlighting for derivatives, parameters, and functions.",
  },
  {
    title: "Analysis Views",
    text: "The tab bar gives you 20+ analysis views: trace, phase portrait, f-I curve, bifurcation, heatmap, sensitivity, characterisation, and more. Click any tab to switch.",
  },
  {
    title: "FPGA Pipeline",
    text: "In ODE mode, click IR → SV → FPGA to compile your equation to SystemVerilog and synthesise for ice40, ECP5, Gowin, or Xilinx. No other SNN framework does this.",
  },
  {
    title: "Network Canvas",
    text: "Click the Canvas tab to design networks visually. Add excitatory (blue) and inhibitory (red) populations, connect them by dragging, then simulate or run the full pipeline.",
  },
  {
    title: "Training Monitor",
    text: "Click Train to start SNN training with surrogate gradients. Watch loss curves, accuracy, and per-layer spike rates update live.",
  },
  {
    title: "Keyboard Shortcuts",
    text: "Press ? for the shortcut overlay. Space = run simulation, 1-5 = switch tabs. Press ? again to dismiss.",
  },
];

export default function OnboardingOverlay() {
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const dismissed = localStorage.getItem(STORAGE_KEY);
    if (!dismissed) setVisible(true);
  }, []);

  if (!visible) return null;

  function dismiss() {
    localStorage.setItem(STORAGE_KEY, "true");
    setVisible(false);
  }

  function next() {
    if (step < steps.length - 1) setStep(step + 1);
    else dismiss();
  }

  function prev() {
    if (step > 0) setStep(step - 1);
  }

  const current = steps[step];

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9999,
      background: "rgba(0, 0, 0, 0.75)",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <div style={{
        background: "var(--bg-secondary)", border: "1px solid var(--border)",
        borderRadius: 12, padding: 24, maxWidth: 480, width: "90%",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {step + 1} / {steps.length}
          </span>
          <button onClick={dismiss} style={{
            background: "none", border: "none", color: "var(--text-muted)",
            cursor: "pointer", fontSize: 14,
          }}>Skip tour</button>
        </div>

        <h2 style={{ fontSize: 16, fontWeight: 600, color: "var(--accent)", marginBottom: 8 }}>
          {current.title}
        </h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 20 }}>
          {current.text}
        </p>

        {/* Progress dots */}
        <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 16 }}>
          {steps.map((_, i) => (
            <div key={i} style={{
              width: 8, height: 8, borderRadius: "50%",
              background: i === step ? "var(--accent)" : "var(--bg-tertiary)",
              cursor: "pointer",
            }} onClick={() => setStep(i)} />
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <button onClick={prev} disabled={step === 0} style={{
            background: "var(--bg-tertiary)", color: "var(--text-secondary)",
            border: "1px solid var(--border)", borderRadius: 6,
            padding: "6px 16px", fontSize: 12, cursor: "pointer",
            opacity: step === 0 ? 0.3 : 1,
          }}>Back</button>
          <button onClick={next} style={{
            background: "var(--accent)", color: "var(--bg-primary)",
            border: "none", borderRadius: 6,
            padding: "6px 20px", fontSize: 12, fontWeight: 600, cursor: "pointer",
          }}>
            {step < steps.length - 1 ? "Next" : "Get Started"}
          </button>
        </div>
      </div>
    </div>
  );
}
