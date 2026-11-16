# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source code equivalence

"""Say whether two versions of a source file differ only in documentation and comments.

The benchmark census accepts a recorded reason for not measuring a record again only when the bound source's code
did not change (:mod:`tools.source_change_reviews`). The code form of a file is what is left once documentation and
comments are removed:

* **Python:** the syntax tree without docstrings (a module's, class's or function's first statement when it is a
  string). Comments are not part of the tree.
* **Rust:** the token text without line, block (nested) and doc comments, with each run of whitespace outside a
  literal reduced to one space. String, raw string (``r``, ``br`` and ``cr``), byte string and character literals
  are kept as they are, so a ``//`` inside one is never taken for a comment.

Every other language has no code form here, and neither has a file that is not UTF-8 or does not parse; two such
versions are never called the same, so the comparison fails closed.
"""

from __future__ import annotations

import ast
import re

#: A raw string's opening: ``r``, its hashes and the quote.
_RAW_OPENING = re.compile(r'r(#*)"')

#: Characters that continue an identifier, so an ``r`` after one is not a raw string's prefix.
_IDENTIFIER = re.compile(r"[A-Za-z0-9_]")


def _python_form(text: str) -> str | None:
    """Return the dump of the Python syntax tree without docstrings, or ``None`` if it does not parse.

    Parameters
    ----------
    text : str
        The module's source.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _quoted_end(text: str, start: int, quote: str) -> int:
    """Return the index after the quote that closes a literal opened at ``start``, or -1.

    Parameters
    ----------
    text : str
        The source.
    start : int
        The index of the opening quote.
    quote : str
        The quote character; a backslash escapes the next character.
    """
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index + 1
        index += 1
    return -1


def _block_comment_end(text: str, start: int) -> int:
    """Return the index after the ``*/`` that closes the block comment at ``start``, or -1; they nest.

    Parameters
    ----------
    text : str
        The source.
    start : int
        The index of the opening ``/*``.
    """
    depth, index = 1, start + 2
    while index < len(text) and depth:
        if text.startswith("/*", index):
            depth, index = depth + 1, index + 2
        elif text.startswith("*/", index):
            depth, index = depth - 1, index + 2
        else:
            index += 1
    return index if depth == 0 else -1


def _raw_string_end(text: str, index: int) -> int | None:
    """Return the index after a raw string that starts at ``index``, -1 if it never ends, ``None`` if none starts.

    Parameters
    ----------
    text : str
        The source.
    index : int
        The index of a possible ``r``.
    """
    opening = _RAW_OPENING.match(text, index)
    if opening is None:
        return None
    before = index - 1
    if before >= 0 and _IDENTIFIER.match(text[before]):
        # ``br"..."`` is a raw byte string and ``cr"..."`` a raw C string; any other letter makes an identifier.
        prefix_is_literal = text[before] in "bc" and (
            before == 0 or not _IDENTIFIER.match(text[before - 1])
        )
        if not prefix_is_literal:
            return None
    closing = '"' + opening.group(1)
    end = text.find(closing, opening.end())
    return -1 if end < 0 else end + len(closing)


def _rust_form(text: str) -> str | None:
    """Return Rust's token text without comments, or ``None`` if a comment or a literal never ends.

    Parameters
    ----------
    text : str
        The file's source.
    """
    pieces: list[str] = []
    space = False
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            space = True
            continue
        if text.startswith("/*", index):
            index = _block_comment_end(text, index)
            if index < 0:
                return None
            space = True
            continue
        character = text[index]
        if character.isspace():
            space = True
            index += 1
            continue
        end = _raw_string_end(text, index) if character == "r" else None
        if end is None and character == '"':
            end = _quoted_end(text, index, '"')
        if end is None and character == "'":
            if index + 1 < len(text) and text[index + 1] == "\\":
                end = _quoted_end(text, index, "'")
            elif index + 2 < len(text) and text[index + 2] == "'":
                end = index + 3
            else:
                end = index + 1
        if end is not None and end < 0:
            return None
        stop = end if end is not None else index + 1
        if space and pieces:
            pieces.append(" ")
        space = False
        pieces.append(text[index:stop])
        index = stop
    return "".join(pieces)


def code_form(path: str, data: bytes) -> str | None:
    """Return a source file's code form, or ``None`` when it has none here.

    Parameters
    ----------
    path : str
        The file's path; its suffix picks the language.
    data : bytes
        The file's bytes.

    Returns
    -------
    str or None
        The text two versions are compared by, or ``None`` for a language without a code form here, bytes that are
        not UTF-8, or a source that does not parse.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if path.endswith(".py"):
        return _python_form(text)
    if path.endswith(".rs"):
        return _rust_form(text)
    return None


def same_code(path: str, old: bytes, new: bytes) -> bool:
    """Say whether two versions of a source have the same code form.

    Parameters
    ----------
    path : str
        The file's path; its suffix picks the language.
    old, new : bytes
        The two versions.

    Returns
    -------
    bool
        True only when both have a code form and the forms are equal.
    """
    old_form = code_form(path, old)
    return old_form is not None and old_form == code_form(path, new)


__all__ = ["code_form", "same_code"]
