// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the custom-ODE editor and the text it writes back to the store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ODE_LANGUAGE_ID, registerODELanguage } from "../ode-language";
import EquationEditor from "./EquationEditor";

/** What the editor wrapper was handed on its last render. */
interface EditorProps {
  height: string;
  defaultLanguage: string;
  value: string;
  theme: string;
  options: Record<string, unknown>;
  onChange: (value: string | undefined) => void;
  beforeMount: (monaco: unknown) => void;
}

const editor = vi.hoisted(() => ({ props: null as EditorProps | null }));

const store = vi.hoisted(() => ({
  equations: ["dv/dt = -v / tau"],
  threshold: "v > 1",
  reset: "v = 0",
  setEquations: vi.fn(),
  setThreshold: vi.fn(),
  setReset: vi.fn(),
}));

vi.mock("@monaco-editor/react", () => ({
  default: (props: EditorProps) => {
    editor.props = props;
    return <div className="editor-stand-in" />;
  },
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => store,
}));

vi.mock("../ode-language", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ode-language")>();
  return { ...actual, registerODELanguage: vi.fn() };
});

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  editor.props = null;
  store.equations = ["dv/dt = -v / tau"];
  store.threshold = "v > 1";
  store.reset = "v = 0";
  store.setEquations.mockClear();
  store.setThreshold.mockClear();
  store.setReset.mockClear();
  vi.mocked(registerODELanguage).mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * Mount the editor and return what its wrapper was handed.
 *
 * @returns The wrapper's props.
 */
async function mount(): Promise<EditorProps> {
  await act(async () => { root?.render(<EquationEditor />); await Promise.resolve(); });
  if (editor.props === null) throw new Error("the editor wrapper was not rendered");
  return editor.props;
}

describe("EquationEditor", () => {
  it("writes the equations, then the threshold and reset as comment lines", async () => {
    store.equations = ["dv/dt = -v / tau", "dw/dt = -w"];
    const props = await mount();
    expect(props.value).toBe("dv/dt = -v / tau\ndw/dt = -w\n\n# threshold: v > 1\n# reset: v = 0");
    expect(container.querySelector(".monaco-wrapper .editor-stand-in")).not.toBeNull();
  });

  it("marks a missing threshold and reset as none", async () => {
    store.threshold = "";
    store.reset = "";
    const props = await mount();
    expect(props.value).toBe("dv/dt = -v / tau\n\n# threshold: (none)\n# reset: (none)");
  });

  it("opens in the Studio's own language and theme, with the chrome turned down", async () => {
    const props = await mount();
    expect(props.defaultLanguage).toBe(ODE_LANGUAGE_ID);
    expect(props.theme).toBe("sc-ode-dark");
    expect(props.height).toBe("180px");
    expect(props.options).toMatchObject({ minimap: { enabled: false }, lineNumbers: "off", wordWrap: "on" });
  });

  it("registers the language before the editor mounts", async () => {
    const props = await mount();
    const monaco = { languages: {} };
    props.beforeMount(monaco);
    expect(registerODELanguage).toHaveBeenCalledWith(monaco);
  });

  it("carries edited equations, threshold and reset back to the store", async () => {
    const props = await mount();
    props.onChange("  dv/dt = (I - v) / tau  \n# a note\nv = v + 1\ndelta = 2\n# threshold: v > 2\n# reset: v = -1");
    expect(store.setEquations).toHaveBeenCalledWith(["dv/dt = (I - v) / tau"]);
    expect(store.setThreshold).toHaveBeenCalledWith("v > 2");
    expect(store.setReset).toHaveBeenCalledWith("v = -1");
  });

  it("clears a threshold or reset marked as none", async () => {
    const props = await mount();
    props.onChange("dv/dt = -v\n# threshold: (none)\n# reset: (none)");
    expect(store.setThreshold).toHaveBeenCalledWith("");
    expect(store.setReset).toHaveBeenCalledWith("");
  });

  it("keeps the equations, threshold and reset that an edit leaves out", async () => {
    const props = await mount();
    props.onChange("# only a note");
    expect(store.setEquations).not.toHaveBeenCalled();
    expect(store.setThreshold).toHaveBeenCalledWith("v > 1");
    expect(store.setReset).toHaveBeenCalledWith("v = 0");
  });

  it("ignores the empty value the editor reports while it loads", async () => {
    const props = await mount();
    props.onChange(undefined);
    props.onChange("");
    expect(store.setEquations).not.toHaveBeenCalled();
    expect(store.setThreshold).not.toHaveBeenCalled();
    expect(store.setReset).not.toHaveBeenCalled();
  });
});

/**
 * Hand the editor's text to its change handler, as a key press does, and
 * re-render with the store the handler wrote to.
 *
 * @param text - The whole text the editor holds after the key.
 * @returns What the wrapper was handed next.
 */
async function type(text: string): Promise<EditorProps> {
  const before = editor.props;
  if (before === null) throw new Error("the editor wrapper was not rendered");
  await act(async () => {
    before.onChange(text);
    root?.render(<EquationEditor />);
    await Promise.resolve();
  });
  if (editor.props === null) throw new Error("the editor wrapper was not rendered");
  return editor.props;
}

describe("EquationEditor while a person types", () => {
  beforeEach(() => {
    // The stand-in store takes what the editor writes, as the Studio's does.
    store.setEquations.mockImplementation((equations: string[]) => { store.equations = equations; });
    store.setThreshold.mockImplementation((threshold: string) => { store.threshold = threshold; });
    store.setReset.mockImplementation((reset: string) => { store.reset = reset; });
  });

  afterEach(() => {
    store.setEquations.mockReset();
    store.setThreshold.mockReset();
    store.setReset.mockReset();
  });

  it("hands the typed text back unchanged, also once it reads as an equation", async () => {
    await mount();
    // Select all, then type: each prefix is the editor's whole text after one more key.
    const typed = "dv/dt = (-v + 1) / 10";
    for (let end = 1; end <= typed.length; end += 1) {
      expect((await type(typed.slice(0, end))).value).toBe(typed.slice(0, end));
    }
    expect(store.equations).toEqual([typed]);
    expect(store.threshold).toBe("v > 1");
    expect(store.reset).toBe("v = 0");
  });

  it("keeps a typed line's own spacing and a threshold typed below it", async () => {
    await mount();
    const typed = "  dv/dt = -v / 2  \n# threshold: v > 2";
    expect((await type(typed)).value).toBe(typed);
    expect(store.equations).toEqual(["dv/dt = -v / 2"]);
    expect(store.threshold).toBe("v > 2");
  });

  it("still shows a change made elsewhere, such as a chosen template, over what was typed", async () => {
    await mount();
    await type("dv/dt = (-v + 1) / 10");
    store.equations = ["dx/dt = 1"];
    store.threshold = "x > 2";
    await act(async () => { root?.render(<EquationEditor />); await Promise.resolve(); });
    expect(editor.props?.value).toBe("dx/dt = 1\n\n# threshold: x > 2\n# reset: v = 0");
  });

  it("shows the rendering, not old typed text, when the store comes back to what was typed", async () => {
    await mount();
    // Type an equation with no threshold line: the store keeps its threshold and reset.
    await type("dv/dt = -v / 2");
    // A template is chosen elsewhere, and shown.
    store.equations = ["dx/dt = 1"];
    store.threshold = "x > 2";
    await act(async () => { root?.render(<EquationEditor />); await Promise.resolve(); });
    expect(editor.props?.value).toBe("dx/dt = 1\n\n# threshold: x > 2\n# reset: v = 0");
    // A saved project brings back exactly the state the typed text led to.
    store.equations = ["dv/dt = -v / 2"];
    store.threshold = "v > 1";
    store.reset = "v = 0";
    await act(async () => { root?.render(<EquationEditor />); await Promise.resolve(); });
    expect(editor.props?.value).toBe("dv/dt = -v / 2\n\n# threshold: v > 1\n# reset: v = 0");
  });
});
