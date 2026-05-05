"""Launch the WaferCNN Streamlit demo.

Usage:
    python scripts/run_wafer_demo.py

Equivalent to:
    streamlit run src/semiconductor_yield/dashboard/wafer_demo.py
"""

import subprocess
import sys
from pathlib import Path

_DEMO_PAGE = (
    Path(__file__).parent.parent
    / "src" / "semiconductor_yield" / "dashboard" / "wafer_demo.py"
)


def main() -> int:
    if not _DEMO_PAGE.exists():
        print(f"Error: demo page not found at {_DEMO_PAGE}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(_DEMO_PAGE)]
    print(f"Starting: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
