from __future__ import annotations

from .sby_orchestrator import TimingProperty


def emit_nuxmv_module(prop: TimingProperty) -> str:
    """Emit a nuXmv transition model for a bounded timing property."""

    bound = prop.bound_cycles
    return f"""-- SC-NeuroCore timing property: {prop.name}
-- Kind: {prop.kind}
MODULE main
VAR
  {prop.reset_n} : boolean;
  {prop.trigger} : boolean;
  {prop.response} : boolean;
  active : boolean;
  age : 0..{bound};
  violation : boolean;
ASSIGN
  init(active) := FALSE;
  init(age) := 0;
  init(violation) := FALSE;
  next(active) := case
    !next({prop.reset_n}) : FALSE;
    violation : active;
    active & next({prop.response}) : FALSE;
    !active & next({prop.trigger}) & !next({prop.response}) : TRUE;
    TRUE : active;
  esac;
  next(age) := case
    !next({prop.reset_n}) : 0;
    violation : age;
    active & next({prop.response}) : 0;
    !active & next({prop.trigger}) : 0;
    active & age < {bound} : age + 1;
    TRUE : age;
  esac;
  next(violation) := case
    !next({prop.reset_n}) : FALSE;
    violation : TRUE;
    active & !next({prop.response}) & age >= {bound} : TRUE;
    TRUE : FALSE;
  esac;
INVARSPEC !violation
"""
