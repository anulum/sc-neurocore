# ODE Equation Editor

The equation editor uses [Monaco](https://microsoft.github.io/monaco-editor/)
(the VS Code engine) with a custom language definition for neuron ODE
equations. Syntax highlighting makes equations readable at a glance and
catches typos before simulation.

## Language: `sc-ode`

The custom language tokeniser recognises six token categories:

| Category | Colour | Examples |
|----------|--------|----------|
| Derivatives | Blue bold | `dv/dt`, `dw/dt`, `dm/dt`, `dn/dt`, `dh/dt` |
| Parameters | Teal | `E_L`, `tau_m`, `g_Na`, `V_threshold`, `C_m`, `E_K` |
| State variables | Light blue | `v`, `w`, `m`, `h`, `n`, `s`, `u`, `a`, `b` |
| Functions | Yellow | `exp`, `sqrt`, `tanh`, `abs`, `log`, `sin`, `cos`, `pow`, `min`, `max`, `sign`, `heaviside`, `sigmoid` |
| Directives | Purple | `threshold:`, `reset:` |
| Comments | Green | `# ...` |

Numbers, operators (`+`, `-`, `*`, `/`, `**`, `>=`, `<=`), and
parentheses are styled with standard editor colours.

## Theme: `sc-ode-dark`

A dark theme (`#0d1117` background) matching the Studio's colour palette.
The theme is registered automatically when the editor mounts — no user
configuration needed.

## Writing Equations

The editor expects one ODE per line in the form `dX/dt = ...`, followed
by `threshold:` and `reset:` directives:

```
dv/dt = -(v - E_L) / tau_m + I / C_m
threshold: v >= V_threshold
reset: v = E_L
```

Multi-variable models work the same way:

```
# Adaptive Exponential (AdEx)
dv/dt = -(v - E_L) / tau_m + delta_T * exp((v - V_T) / delta_T) / tau_m - w / C_m + I / C_m
dw/dt = a * (v - E_L) / tau_w - w / tau_w
threshold: v >= V_threshold
reset: v = V_reset; w = w + b
```

## Parameter Recognition

The tokeniser matches common neuroscience parameter naming conventions:

- `E_*` — reversal potentials (E_L, E_Na, E_K, E_syn)
- `g_*` — conductances (g_Na, g_K, g_L, g_syn)
- `tau_*` — time constants (tau_m, tau_w, tau_syn)
- `V_*` — voltage thresholds/references (V_threshold, V_reset, V_T)
- `C_m` — membrane capacitance
- `I_*` — current sources (I_ext, I_syn)
- `alpha_*`, `beta_*` — rate constants for Hodgkin-Huxley gating

Any parameter not in the built-in list still gets default styling —
the tokeniser doesn't reject unknown names.

## Editor Configuration

The Monaco instance is configured for equation editing:

| Setting | Value | Why |
|---------|-------|-----|
| Height | 180px | Fits 6–8 equations without scroll |
| Line numbers | Off | Equations, not code |
| Minimap | Off | Too small to be useful |
| Word wrap | On | Long equations stay visible |
| Scrollbar | Hidden | Clean appearance |
| Folding | Off | Flat structure |
| Suggestions | Off | No autocomplete (parameters are model-specific) |

## Switching Between Modes

The Studio has two input modes:

1. **Model browser** — select from 118 built-in neuron models,
   adjust parameters with sliders
2. **ODE mode** — write custom equations in the Monaco editor

Toggle with the **ODE** button in the header. Both modes feed into
the same simulation engine and all analysis views work with either.

## Implementation

- `studio/frontend/src/ode-language.ts` — language ID, tokeniser, theme
- `studio/frontend/src/components/EquationEditor.tsx` — Monaco wrapper
- Language registration happens once via `beforeMount` callback
