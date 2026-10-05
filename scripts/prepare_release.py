#!/usr/bin/env python3
"""Replace provisional GitHub URLs once the owner/repo slug has been chosen.

Usage: python3 scripts/prepare_release.py OWNER/handoff-router
       python3 scripts/prepare_release.py --check
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "your-org" + "/handoff-router"


def tracked_files() -> list[Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    return [ROOT / name.decode() for name in names if name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", help="GitHub owner/repository, e.g. dao/handoff-router")
    parser.add_argument("--check", action="store_true", help="fail if any provisional URLs remain")
    args = parser.parse_args()
    if not args.check and (not args.repository or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository)):
        parser.error("provide a valid OWNER/REPOSITORY, or use --check")
    found = []
    for path in tracked_files():
        if not path.is_file() or path == Path(__file__).resolve():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PLACEHOLDER not in content:
            continue
        found.append(path.relative_to(ROOT))
        if not args.check:
            path.write_text(content.replace(PLACEHOLDER, args.repository), encoding="utf-8")
    if args.check:
        if found:
            parser.exit(1, "Provisional GitHub URLs remain in: " + ", ".join(map(str, found)) + "\n")
        print("Release URLs: no placeholders found")
    else:
        print(f"Updated {len(found)} files for https://github.com/{args.repository}")


if __name__ == "__main__":
    main()
