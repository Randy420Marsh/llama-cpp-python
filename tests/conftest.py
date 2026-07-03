"""
Shared test configuration and fixtures.

This conftest.py applies module-level patches to allow importing llama_cpp
without requiring the compiled shared libraries to be present.  The patches
run at collection time (before any test module is imported) so that every
test in the suite can import llama_cpp freely.

The patches are only applied when no compiled library is found on disk, so a
built tree (or CI) still runs the integration tests against the real library.
"""

import ctypes
import os
import pathlib
from unittest.mock import MagicMock

_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent / "llama_cpp" / "lib"
_HAS_REAL_LIB = _LIB_DIR.is_dir() and any(
    path.suffix in (".so", ".dylib", ".dll")
    for path in _LIB_DIR.iterdir()
    if path.is_file()
)

# ---------------------------------------------------------------------------
# Patch ctypes.CDLL and pathlib.Path.exists so that load_shared_library()
# succeeds without needing actual .so / .dylib / .dll files on disk.
# ---------------------------------------------------------------------------

_mock_cdll = MagicMock()
_original_path_exists = pathlib.Path.exists
_original_ctypes_cdll = ctypes.CDLL


def _patched_path_exists(self: pathlib.Path) -> bool:
    """Return True for paths that look like the llama / mtmd shared library."""
    name = str(self)
    if "lib" in name and any(token in name for token in ("llama", "mtmd")):
        return True
    return _original_path_exists(self)


def _patched_cdll(path: str, **kwargs) -> MagicMock:
    """Return a mock CDLL object instead of actually loading a shared library."""
    return _mock_cdll


if not _HAS_REAL_LIB:
    pathlib.Path.exists = _patched_path_exists  # type: ignore[method-assign]
    ctypes.CDLL = _patched_cdll  # type: ignore[assignment]

    # On Windows, load_shared_library() also registers the (possibly
    # nonexistent) llama_cpp/lib directory in the DLL search path before
    # loading. Tolerate a missing directory so imports succeed on a
    # source-only checkout.
    if hasattr(os, "add_dll_directory"):
        _original_add_dll_directory = os.add_dll_directory

        def _patched_add_dll_directory(path: str):
            try:
                return _original_add_dll_directory(path)
            except (FileNotFoundError, OSError):
                return MagicMock()

        os.add_dll_directory = _patched_add_dll_directory  # type: ignore[assignment]
