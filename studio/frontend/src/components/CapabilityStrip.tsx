import { useStudioStore } from "../stores/studio";
import { summarizeCapabilities } from "../capabilityShell";

export default function CapabilityStrip() {
  const { capabilities, capabilitiesError, capabilitiesLoading } = useStudioStore();

  if (capabilitiesLoading) {
    return (
      <div className="capability-strip" title="Loading Studio capability registry">
        capabilities...
      </div>
    );
  }

  if (capabilitiesError) {
    return (
      <div className="capability-strip capability-strip-warning" title={capabilitiesError}>
        capability check failed
      </div>
    );
  }

  if (capabilities.length === 0) return null;

  const summary = summarizeCapabilities(capabilities);
  const degraded = capabilities.find((capability) => !capability.healthy);

  return (
    <details
      className={degraded ? "capability-strip capability-strip-warning" : "capability-strip"}
      title={degraded ? `${degraded.title}: ${degraded.message}` : "All registered capabilities are available"}
    >
      <summary>
        <span>{summary.headline}</span>
        {degraded && <span className="capability-strip-detail">{degraded.title}</span>}
      </summary>
      <div className="capability-menu">
        {capabilities.map((capability) => {
          const missing = capability.requirements.filter((requirement) => !requirement.available);
          return (
            <section key={capability.capability_id} className="capability-menu-row">
              <div className="capability-menu-title">
                <span>{capability.title}</span>
                <span className={`capability-badge capability-badge-${capability.status}`}>
                  {capability.status}
                </span>
              </div>
              <p>{capability.summary}</p>
              <div className="capability-menu-meta">
                <span>{capability.ui_placement}</span>
                <span>{capability.evidence.join(", ") || "no evidence"}</span>
                {capability.docs_path && (
                  <a href={`/${capability.docs_path}`} target="_blank" rel="noreferrer">docs</a>
                )}
              </div>
              {missing.length > 0 && (
                <ul className="capability-menu-requirements">
                  {missing.map((requirement) => (
                    <li key={requirement.name}>{requirement.name}: {requirement.detail}</li>
                  ))}
                </ul>
              )}
            </section>
          );
        })}
      </div>
    </details>
  );
}
