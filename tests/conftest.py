"""
Make the project root importable as a set of top-level packages
(scripts, analysis, ml, dashboard) regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
