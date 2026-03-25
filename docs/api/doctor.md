# Architecture Doctor

Automated SNN diagnostics: coding efficiency, hardware fit, spike health, actionable recommendations.

```python
from sc_neurocore.doctor import ArchitectureDoctor

doc = ArchitectureDoctor()
report = doc.diagnose(model)
for issue in report.issues:
    print(f"[{issue.severity}] {issue.message}")
```

See [Tutorial 56: Architecture Doctor](../tutorials/56_architecture_doctor.md).

::: sc_neurocore.doctor
    options:
      show_root_heading: true
