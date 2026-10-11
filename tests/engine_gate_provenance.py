# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine gate source binding contracts

"""Resolve static gate bindings and retain reviewed standalone consumer prefixes."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_REVIEWED_PREFIXES = {
    "test_default_neuron_engine_binding.py": "4442012005ef12da616880f4284b1f483d79c172db3737bf4e2abeef4af24af1",
    "test_engine_global_pickle.py": "38a4501367d631d5b181110fc68dd0d014043592cf48ca84f54037bbacbb7480",
}


def _expression_name(node: ast.expr) -> str:
    """Resolve a name or qualified attribute without executing source code."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else ""
    return ""


def _ast_signature(value: object) -> object:
    """Normalize source structure across Python versions without losing gate code."""
    if isinstance(value, ast.AST):
        return [
            type(value).__name__,
            [
                [name, _ast_signature(item)]
                for name, item in ast.iter_fields(value)
                if name != "type_params" or item
            ],
        ]
    if isinstance(value, list):
        return [_ast_signature(item) for item in value]
    return [type(value).__name__, repr(value)]


class _BindingWrites(ast.NodeVisitor):
    """Collect import-time binding changes, excluding nested local scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.attribute_names: set[str] = set()
        self.module_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Record module assignment and deletion targets."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Invalidate a namespace whose attributes are replaced or deleted."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            name = _expression_name(node).split(".")[0]
            self.names.add(name)
            self.attribute_names.add(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        """Identify class-body stores that bind the enclosing module namespace."""
        self.global_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        """Record the actual namespace names bound by an import."""
        self.names.update(a.asname or a.name.split(".")[0] for a in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record named and star-import binding replacements."""
        self.names.update(a.asname or a.name for a in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record the definition while leaving its local body unexecuted."""
        self.names.add(node.name)
        self._visit_callable_inputs(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an async definition without traversing its local body."""
        self.names.add(node.name)
        self._visit_callable_inputs(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Scan executed defaults without treating lambda-local stores as global."""
        self._visit_callable_inputs(node)

    def _visit_callable_inputs(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> None:
        """Include decorator, default and annotation expressions executed at import."""
        for decorator in getattr(node, "decorator_list", []):
            self.visit(decorator)
        for value in [*node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                self.visit(value)
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        returns = getattr(node, "returns", None)
        if returns is not None:
            self.visit(returns)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Keep class-local names separate from global and namespace mutations."""
        self.names.add(node.name)
        for expression in [*node.decorator_list, *node.bases]:
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        writes = _BindingWrites()
        for statement in node.body:
            writes.visit(statement)
        module_names = (
            (writes.names & writes.global_names) | writes.attribute_names | writes.module_names
        )
        self.names.update(module_names)
        self.module_names.update(module_names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Invalidate exception-bound names and visit their executed handlers."""
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)


def gate_has_trusted_binding(tree: ast.Module, index: int, path: Path) -> bool:
    """Require a maintained import binding or an unchanged reviewed fallback.

    Parameters
    ----------
    tree
        Actual test module parsed by the repository import-dominance sweep.
    index
        Top-level statement containing the otherwise unconditional gate call.
    path
        Actual module path; standalone exceptions are limited to these two files.

    Returns
    -------
    bool
        Whether the source binds the call to the maintained gate or pytest,
        or preserves an independently reviewed consumer prefix. Assignments,
        replacement imports and unreviewed local definitions invalidate trust.

    Notes
    -----
    This is static binding validation, not a proof against arbitrary dynamic
    Python execution. Reviewed prefix digests cover the complete local gate
    wiring and body, including the actual call, rather than trusting its name.
    """
    statement = tree.body[index]
    if not isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
        return False
    call = statement.value
    if not isinstance(call, ast.Call):
        return False
    prefix = ast.Module(body=tree.body[: index + 1], type_ignores=[])
    digest = hashlib.sha256(repr(_ast_signature(prefix)).encode()).hexdigest()
    if path.parent.name == "tests" and _REVIEWED_PREFIXES.get(path.name) == digest:
        return True

    bindings: dict[str, str] = {}
    for preceding in tree.body[:index]:
        writes = _BindingWrites()
        writes.visit(preceding)
        bindings = {
            name: origin
            for name, origin in bindings.items()
            if "*" not in writes.names and name.split(".")[0] not in writes.names
        }
        if isinstance(preceding, ast.Import):
            for alias in preceding.names:
                bindings[alias.asname or alias.name] = alias.name
        elif isinstance(preceding, ast.ImportFrom) and preceding.level == 0:
            for alias in preceding.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{preceding.module}.{alias.name}"

    function = _expression_name(call.func)
    maintained = bindings.get(function) == "tests.engine_requirement.require_engine"
    if isinstance(call.func, ast.Attribute):
        namespace = bindings.get(_expression_name(call.func.value))
        maintained = maintained or (
            namespace == "tests.engine_requirement" and call.func.attr == "require_engine"
        )
        if namespace == "pytest" and call.func.attr == "importorskip":
            return True
    if not maintained:
        return False
    if any(keyword.arg != "module" for keyword in call.keywords):
        return False
    arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
    if not arguments:
        return True
    if len(arguments) != 1:
        return False
    argument = arguments[0]
    return (
        isinstance(argument, ast.Constant)
        and argument.value == "sc_neurocore_engine.sc_neurocore_engine"
    ) or (
        isinstance(argument, ast.Name)
        and bindings.get(argument.id) == "tests.engine_requirement.ENGINE_EXTENSION_MODULE"
    )
