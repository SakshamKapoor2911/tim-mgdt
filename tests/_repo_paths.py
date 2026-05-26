"""Shared repo-root path setup for tests run as scripts or via pytest."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_repo_paths():
    """Ensure imports and relative output paths resolve from repository root."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    code_dir = str(REPO_ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    os.chdir(REPO_ROOT)
