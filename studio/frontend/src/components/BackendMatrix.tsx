import type { ModelBackendSupport } from "../api/client";

const PARITY_COLOR: Record<string, string> = {
  exact: "var(--success)",
  "ulp-bounded": "var(--accent)",
  approximate: "var(--warning)",
  "n/a": "var(--text-muted)",
};

/** Order backends with the reference (python) first, then the accelerated tiers. */
const ORDER = ["python", "rust", "julia", "go", "mojo"];

export function orderedBackends(backends: ModelBackendSupport[]): ModelBackendSupport[] {
  return [...backends]
    .filter((b) => b.status === "implemented")
    .sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
}

export default function BackendMatrix({ backends }: { backends: ModelBackendSupport[] }) {
  const impl = orderedBackends(backends);
  if (impl.length < 2) return null;
  return (
    <div className="panel-section">
      <div className="panel-header">Compute backends ({impl.length})</div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", padding: "2px 0" }}>
        {impl.map((b) => (
          <span
            key={b.name}
            title={`${b.name}: ${b.parity} parity vs the Python reference`}
            style={{
              fontSize: 9, padding: "1px 5px", borderRadius: 3,
              fontFamily: "var(--font-mono)",
              background: "var(--bg-tertiary)",
              border: `1px solid ${PARITY_COLOR[b.parity] ?? "var(--border)"}`,
              color: "var(--text-secondary)",
            }}
          >
            {b.name}
            <span style={{ color: PARITY_COLOR[b.parity] ?? "var(--text-muted)", marginLeft: 4 }}>
              {b.name === "python" ? "ref" : b.parity}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
