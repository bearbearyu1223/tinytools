"""Test-wide settings."""

import os

# Report lines over 74 columns raise in tests (a real run folds them instead: report.py).
os.environ.setdefault("TINYTOOLS_STRICT_WIDTH", "1")
