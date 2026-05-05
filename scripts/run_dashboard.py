"""Launch the unified Semiconductor Yield Analytics Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py

Equivalent to:
    streamlit run src/semiconductor_yield/dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

_APP = (
    Path(__file__).parent.parent
    / "src" / "semiconductor_yield" / "dashboard" / "app.py"
)


def main() -> int:
    if not _APP.exists():
        print(f"Error: dashboard not found at {_APP}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP)]
    print(f"Starting: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
