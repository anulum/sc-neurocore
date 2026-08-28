# First-Time Onboarding

On first visit, the Studio displays a guided tour introducing key
features. The overlay walks through 8 steps covering the full workflow
from model selection to FPGA deployment. It appears once and doesn't
return (state stored in `localStorage`).

## Tour Steps

| Step | Title | What It Explains |
|------|-------|-----------------|
| 1 | Welcome | What the Studio does — design, train, compile, deploy SNNs |
| 2 | Model Browser | 179 runtime catalogue entries with parameter sliders |
| 3 | ODE Mode | Custom equations with syntax highlighting |
| 4 | Analysis Views | 18+ views: trace, phase, ISI, f-I, bifurcation, heatmap |
| 5 | FPGA Pipeline | Compiler Inspector → Synthesis Dashboard |
| 6 | Network Canvas | Drag-and-drop population/projection graph editor |
| 7 | Training | Surrogate gradient training with live metrics |
| 8 | Keyboard Shortcuts | Press `?` for the shortcut overlay |

## Behaviour

- **First visit:** overlay appears automatically after the app loads
- **Navigation:** Next / Back buttons, step counter (e.g. "3 / 8")
- **Dismissal:** "Skip Tour" link or reaching the final step
- **Persistence:** `localStorage` key `sc-studio-onboarding-dismissed`
- **Reset:** clear the key in browser DevTools to see the tour again:
  ```js
  localStorage.removeItem("sc-studio-onboarding-dismissed");
  ```

## Visual Design

The overlay uses a semi-transparent backdrop (`rgba(0,0,0,0.7)`) with
a centred card:

- Dark background (`#161b22`) matching the Studio theme
- Accent-coloured title bar
- Step counter in muted text
- Navigation buttons styled consistently with Studio controls
- Skip link in subtle `#8b949e` text

The card is responsive — on screens narrower than 480px it fills the
available width with appropriate padding.

## Why an Onboarding Tour

SC-NeuroCore Studio combines simulation, analysis, compilation, synthesis,
training, and network design in one interface. Without guidance, a new
user sees 20+ tabs and buttons with no obvious starting point. The tour
provides a 60-second orientation that reduces time-to-first-simulation.

This matters for the target audience: the ominux-type hardware engineer
who forked the repo to evaluate the FPGA pipeline. They need to know
where the compiler and synthesis tools are without reading all the docs
first.

## Implementation

- `studio/frontend/src/components/OnboardingOverlay.tsx` — component
- `studio/frontend/src/App.tsx` — mounted at bottom of app tree
- Tour content is defined as a static array — add steps by appending
  to the `steps` array in `OnboardingOverlay.tsx`
