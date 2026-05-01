from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dashboard_app = repo_root / "src" / "observability" / "dashboard" / "app.py"
    if not dashboard_app.exists():
        print("dashboard app not found: " + str(dashboard_app), file=sys.stderr)
        return 1
    cmd = ["streamlit", "run", str(dashboard_app)]
    try:
        return subprocess.call(cmd, cwd=str(repo_root))
    except FileNotFoundError:
        print("streamlit command not found", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
