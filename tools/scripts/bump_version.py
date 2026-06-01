#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Bump the LLM AIO Gateway project version.

Single source of truth: app/__init__.py.__version__

Usage:
  python tools/scripts/bump_version.py 0.2.0
  python tools/scripts/bump_version.py 0.2.0 --dry-run
  python tools/scripts/bump_version.py --current
  python tools/scripts/bump_version.py --check
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = REPO_ROOT / "app" / "__init__.py"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
# Match __version__ = "..."  (double or single quoted)
VERSION_LINE_RE = re.compile(
    r"(__version__\s*=\s*)"  # group 1: prefix
    r"([" + chr(34) + chr(39) + "])"              # group 2: opening quote
    r"([^" + chr(34) + chr(39) + "]+)"            # group 3: version
    r"([" + chr(34) + chr(39) + "])"              # group 4: closing quote
)


def read_version():
    text = VERSION_FILE.read_text(encoding="utf-8")
    m = VERSION_LINE_RE.search(text)
    if not m:
        sys.exit("ERROR: cannot find __version__ in " + str(VERSION_FILE))
    return m.group(3)


def validate(version):
    if not SEMVER_RE.match(version):
        sys.exit("ERROR: " + repr(version) + " is not a valid semver")


def write_version(new):
    text = VERSION_FILE.read_text(encoding="utf-8")
    new_text, n = VERSION_LINE_RE.subn(
        lambda m: m.group(1) + m.group(2) + new + m.group(4),
        text, count=1,
    )
    if n != 1:
        sys.exit("ERROR: failed to rewrite __version__")
    VERSION_FILE.write_text(new_text, encoding="utf-8", newline="\n")


def update_doc_examples(old, new, dry_run):
    candidates = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "README_en.md",
        REPO_ROOT / "使用说明书.md",
        REPO_ROOT / "tools" / "dist_tools" / "README.md",
        REPO_ROOT / "tools" / "dist_tools" / "USER_GUIDE.md",
    ]
    patterns = [
        (re.compile("v" + re.escape(old) + r"\.zip"), "v" + new + ".zip"),
        (re.compile("v" + re.escape(old) + r"\.tar\.gz"), "v" + new + ".tar.gz"),
        (re.compile(r'"version"\s*:\s*"' + re.escape(old) + '"'), '"version": "' + new + '"'),
    ]
    touched = []
    for f in candidates:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        new_text = text
        for pat, repl in patterns:
            new_text = pat.sub(repl, new_text)
        if new_text != text:
            if not dry_run:
                f.write_text(new_text, encoding="utf-8", newline="\n")
            touched.append(f)
    return touched


def cmd_current():
    print(read_version())
    return 0


def cmd_check():
    cur = read_version()
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()
    except FileNotFoundError:
        tag = ""
    if tag and tag.startswith("v"):
        tag_v = tag[1:]
        if tag_v != cur:
            print("__version__=" + cur + " != git tag " + tag, file=sys.stderr)
            return 1
    return 0


def cmd_bump(new, dry_run):
    new = new.lstrip("v")
    validate(new)
    old = read_version()
    if old == new:
        print("already at " + new)
        return 0
    if new.split(".")[:2] != old.split(".")[:2]:
        print("WARNING: bumping " + old + " -> " + new + " crosses major.minor")
    print("bump " + old + " -> " + new + " (dry-run=" + str(dry_run) + ")")
    if not dry_run:
        write_version(new)
    touched = update_doc_examples(old, new, dry_run)
    if touched:
        print("doc snippets updated:")
        for f in touched:
            print("  " + str(f.relative_to(REPO_ROOT)))
    print()
    print("next steps:")
    extra = " ".join(str(f.relative_to(REPO_ROOT)) for f in touched)
    print("  git add app/__init__.py " + extra)
    print('  git commit -m "Bump version to ' + new + '"')
    print("  git tag v" + new)
    print("  git push origin main --tags")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("version", nargs="?", help="new version (e.g. 0.2.0)")
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    parser.add_argument("--current", action="store_true", help="print current version")
    parser.add_argument("--check", action="store_true", help="exit 0 if version matches tag")
    args = parser.parse_args()

    if args.current:
        return cmd_current()
    if args.check:
        return cmd_check()
    if not args.version:
        parser.print_help()
        return 1
    return cmd_bump(args.version, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
