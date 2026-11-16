# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source code equivalence tests

"""Two versions of a source differ in code only when more than documentation and comments changed.

The benchmark census accepts a recorded reason for not measuring a record again only when the bound source's code
did not change, so this comparison must never call a code change a comment change. Each case below that changes
code inside a string, a raw string or after a character literal would pass a comparison that only cut ``//`` to the
end of the line.
"""

from __future__ import annotations

import pytest

from tools.source_code_equivalence import code_form, same_code

_PYTHON = b'"""Module."""\n\n\ndef step(v):\n    """Advance."""\n    # leak\n    return v * 0.5\n'


@pytest.mark.parametrize(
    "new",
    [
        b'"""Module, reworded."""\n\n\ndef step(v):\n    """Advance one step."""\n    # leak\n    return v * 0.5\n',
        b'"""Module."""\n\n\ndef step(v):\n    """Advance."""\n    # the leak, explained\n    return v * 0.5  # half\n',
        b'"""Module."""\n\n\ndef step(v):\n    return v * 0.5\n',
    ],
    ids=["docstrings", "comments", "a docstring removed"],
)
def test_python_documentation_and_comments_are_not_code(new: bytes) -> None:
    """A changed or removed docstring and changed comments leave the code the same.

    Parameters
    ----------
    new : bytes
        The changed version of the module.
    """
    assert same_code("src/model.py", _PYTHON, new)


@pytest.mark.parametrize(
    "new",
    [
        b'"""Module."""\n\n\ndef step(v):\n    """Advance."""\n    # leak\n    return v * 0.25\n',
        b'"""Module."""\n\n\ndef step(v):\n    """Advance."""\n    x = "not a docstring"\n    return v * 0.5\n',
    ],
    ids=["a constant", "a string statement after the docstring"],
)
def test_python_code_changes_are_code(new: bytes) -> None:
    """A changed constant, or a new string statement that is not the docstring, is a code change.

    Parameters
    ----------
    new : bytes
        The changed version of the module.
    """
    assert not same_code("src/model.py", _PYTHON, new)


def test_python_that_does_not_parse_has_no_code_form() -> None:
    """A file that does not parse cannot be compared, so it never counts as unchanged."""
    assert code_form("src/model.py", b"def broken(:\n") is None
    assert not same_code("src/model.py", b"def broken(:\n", b"def broken(:\n")


_RUST = b"""//! Partition.

/// Split the graph.
pub fn split(n: usize) -> usize {
    /* weights /* nested */ end */
    let url = "http://a"; // trailing
    let raw = r#"x // y"#;
    let slash = '/';
    n / 2 // half
}
"""


@pytest.mark.parametrize(
    "new",
    [
        _RUST.replace(b"//! Partition.", b"//! Partition of the graph, reworded."),
        _RUST.replace(b"/// Split the graph.", b"/// Split the graph in two.\n/// A second line."),
        _RUST.replace(b"/* weights /* nested */ end */", b"/* other /* nested */ words */"),
        _RUST.replace(b"// trailing", b"// another trailing comment"),
        _RUST.replace(b"n / 2 // half", b"n / 2"),
    ],
    ids=["inner doc", "outer doc", "nested block comment", "line comment", "a comment removed"],
)
def test_rust_documentation_and_comments_are_not_code(new: bytes) -> None:
    """Changed doc comments, line comments and nested block comments leave the code the same.

    Parameters
    ----------
    new : bytes
        The changed version of the file.
    """
    assert same_code("engine/src/partition.rs", _RUST, new)


@pytest.mark.parametrize(
    "new",
    [
        _RUST.replace(b'"http://a"', b'"http://b"'),
        _RUST.replace(b'r#"x // y"#', b'r#"x // z"#'),
        _RUST.replace(b"let slash = '/';\n    n / 2", b"let slash = '/';\n    n / 3"),
        _RUST.replace(b"n / 2", b"n / 2 + 1"),
    ],
    ids=[
        "after // in a string",
        "after // in a raw string",
        "after a '/' character",
        "an expression",
    ],
)
def test_rust_code_changes_are_code(new: bytes) -> None:
    """A change inside a string or a raw string, or after a character literal, is a code change.

    Parameters
    ----------
    new : bytes
        The changed version of the file.
    """
    assert not same_code("engine/src/partition.rs", _RUST, new)


def test_rust_lifetimes_and_escaped_characters_are_read_as_code() -> None:
    """A lifetime is not the start of a character literal, and an escaped quote does not end one."""
    old = b"fn f<'a>(x: &'a str) -> char { let q = '\\''; 'x' } // end\n"
    assert same_code("src/lib.rs", old, old.replace(b"// end", b"// the end"))
    assert not same_code("src/lib.rs", old, old.replace(b"'x'", b"'y'"))


def test_an_unterminated_comment_or_string_has_no_code_form() -> None:
    """A Rust file whose comment or string never ends cannot be compared."""
    assert code_form("src/lib.rs", b"fn f() { /* open\n}\n") is None
    assert code_form("src/lib.rs", b'fn f() { let s = "open; }\n') is None
    assert code_form("src/lib.rs", b'fn f() { let s = r#"open"; }\n') is None


def test_a_language_without_a_code_form_is_never_unchanged() -> None:
    """A TOML or JSON source has no code form here, so a review of it is never accepted."""
    assert code_form("pyproject.toml", b"a = 1\n") is None
    assert not same_code("pyproject.toml", b"a = 1\n", b"a = 1\n")


def test_bytes_that_are_not_utf8_have_no_code_form() -> None:
    """A file that is not UTF-8 text cannot be compared."""
    assert code_form("src/lib.rs", b"\xff\xfe") is None


def test_a_byte_raw_string_is_a_literal_and_an_identifier_ending_in_r_is_not_its_prefix() -> None:
    """``br"..."`` is a raw string, while ``xr"..."`` is an identifier and a string: neither hides a ``//``."""
    assert code_form("src/lib.rs", b'let s = br"a // b";') == 'let s = br"a // b";'
    assert code_form("src/lib.rs", b'xr"a // b"') == 'xr"a // b"'


@pytest.mark.parametrize("hashes", ["", "#"], ids=["cr", "cr#"])
def test_a_raw_c_string_ends_at_its_own_quote_so_a_later_string_stays_code(hashes: str) -> None:
    r"""``cr"\"`` is a raw C string that ends at its second quote; a backslash in it escapes nothing.

    Read as an escaped string, it would run on to the next quote, and the ``//`` inside the string after it would be
    taken for a comment, so a change of that string would look like no change.

    Parameters
    ----------
    hashes : str
        The raw string's hashes.
    """
    old = f'let a = cr{hashes}"\\"{hashes}; let b = "// x";\n'.encode()
    assert code_form("src/lib.rs", old) == old.decode().strip()
    assert not same_code("src/lib.rs", old, old.replace(b"// x", b"// y"))
