"""Test configuration and fixtures."""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def has_dll() -> bool:
    """Check if atlas.dll is available in project root."""
    dll_path = os.path.join(PROJECT_ROOT, "atlas.dll")
    return os.path.exists(dll_path)


def has_model() -> str | None:
    """Check for .atlas model files in project root."""
    for f in os.listdir(PROJECT_ROOT):
        if f.endswith(".atlas"):
            return os.path.join(PROJECT_ROOT, f)
    return None
