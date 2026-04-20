# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/logging

module LoggingAccel

using Statistics, LinearAlgebra

function format(record)
    entry = {
        "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "msg": record.getMessage(),
    }
    if record.exc_info && record.exc_info[0] is ! nothing
        entry["exc"] = s.formatException(record.exc_info)
    return json.dumps(entry, default=str)
end

function configure_logging(level, json, stream)
    level: str | int = "WARNING",
    json: bool = false,  # noqa: A002 — shadows builtin intentionally for clean API
    stream: IO[str] | nothing = nothing,
    ) -> nothing
    root = logging.getLogger("sc_neurocore")
    root.handlers.clear()
    handler = logging.StreamHandler(stream || sys.stderr)
    if json
        handler.setFormatter(JSONFormatter())
    else
        handler.setFormatter(logging.Formatter(_HUMAN_FMT))
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else getattr(logging, level.upper()))
end

end # module LoggingAccel
