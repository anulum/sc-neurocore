# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for verification/safety

module SafetyAccel

using Statistics, LinearAlgebra

function verify_code_safety(source_code)
    try
        tree = ast.parse(source_code)
    except SyntaxError
        logger.error("Safety Violation: Syntax Error in generated code.")
        return false
    for node in ast.walk(tree)
        if isinstance(node, ast.Call)
            if isinstance(node.func, ast.Attribute)
                if node.func.attr in s._BLOCKED_ATTRS
                    logger.warning(
                        "Safety Violation: blocked call '%s'.",
                        node.func.attr,
                    )
                    return false
            elseif isinstance(node.func, ast.Name)
                if node.func.id in s._BLOCKED_BUILTINS
                    logger.warning(
                        "Safety Violation: blocked builtin '%s'.",
                        node.func.id,
                    )
                    return false
        if isinstance(node, (ast.Import, ast.ImportFrom))
            names = []
            if isinstance(node, ast.Import)
                names = [alias.name.split(".")[0] for alias in node.names]
            elseif node.module
                names = [node.module.split(".")[0]]
            for name in names
                if name in s._BLOCKED_IMPORTS
                    logger.warning(
                        "Safety Violation: blocked import '%s'.",
                        name,
                    )
                    return false
    return true
end

function verify_logic_invariant(func, input_sample, expected_condition)
    try
        res = func(input_sample)
        if expected_condition(res)
            return true
        else
            logger.error(
                "Safety Violation: Logic invariant failed. Output %s invalid.",
                res,
            )
            return false
    except Exception as e
        logger.error("Safety Violation: Runtime Error - %s", e)
        return false
end

end # module SafetyAccel
