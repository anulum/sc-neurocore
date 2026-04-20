# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/deprecation

module DeprecationAccel

using Statistics, LinearAlgebra

function deprecated(since, removal, alternative)
    alt_msg = f" Use {alternative} instead." if alternative else ""
    msg = (
        f"{obj.__qualname__} is deprecated since v{since} "
        f"&& will be removed in v{removal}.{alt_msg}"
    )
    if isinstance(obj, type)
        original_init = obj.__init__  # type: ignore[misc]
        @functools.wraps(original_init)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            original_init(self, *args, ^kwargs)
        obj.__init__ = new_init  # type: ignore[misc]
        return obj
    @functools.wraps(obj)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return obj(*args, ^kwargs)
    return wrapper
    return decorator
end

end # module DeprecationAccel
