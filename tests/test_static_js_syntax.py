"""Validate JavaScript syntax of static frontend files.

Why: the backend test suite runs in a Python process and has no natural
JS syntax check.  Missing commas, unmatched braces, or other JS syntax
errors in app.js silently break the entire admin UI — including login —
without any test noticing.

This test uses esprima (pure Python JS parser) to catch such errors at
pytest time.  esprima is a hard test dependency (requirements-dev.txt):
the check must fail loudly, never skip silently, or it is not a guard rail.
"""

import pathlib

import pytest
import esprima  # requires requirements-dev.txt

STATIC_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "web" / "static"


def test_all_js_files_have_valid_syntax():
    """Every .js file in the static directory must parse as valid ES.

    Catches: missing commas between object properties, unmatched braces,
    stray characters, or other syntax errors that would break the frontend
    at load time.
    """
    errors = []
    for path in sorted(STATIC_DIR.glob("*.js")):
        src = path.read_text(encoding="utf-8")
        try:
            esprima.parseScript(src)
        except esprima.Error as exc:
            errors.append(f"{path.name}: {exc}")
    assert not errors, "\n".join(errors)