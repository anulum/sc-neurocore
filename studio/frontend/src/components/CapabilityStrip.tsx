import { useStudioStore } from "../stores/studio";

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

  const healthyCount = capabilities.filter((capability) => capability.healthy).length;
  const degraded = capabilities.find((capability) => !capability.healthy);
  const summary = `${healthyCount}/${capabilities.length} capabilities`;

  return (
    <div
      className={degraded ? "capability-strip capability-strip-warning" : "capability-strip"}
      title={degraded ? `${degraded.title}: ${degraded.message}` : "All registered capabilities are available"}
    >
      <span>{summary}</span>
      {degraded && <span className="capability-strip-detail">{degraded.title}</span>}
    </div>
  );
}
