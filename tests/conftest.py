"""pytest configuration shared across all tests.

Sets matplotlib to the non-interactive Agg backend before any test module
imports matplotlib.pyplot. This prevents "no display" errors in headless CI
environments and on Windows systems without a GUI.
"""

import matplotlib

matplotlib.use("Agg")
