from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

_INVALID_PORTABLE_PATH_CHARS = set('<>:"|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def safe_path_segment(value: str, label: str) -> str:
    """Validate a single path segment destined for an external repository.

    Rejects traversal (`.`, `..`), `.git`, separators, control characters, characters that are
    invalid on Windows, Windows reserved device names, and trailing dots/spaces. Returns the
    segment unchanged; raises ValueError otherwise.
    """
    windows_stem = value.split(".", 1)[0].upper()
    if (
        not value
        or value in {".", ".."}
        or value.casefold() == ".git"
        or "/" in value
        or "\\" in value
        or value != value.rstrip(" .")
        or windows_stem in _WINDOWS_RESERVED_NAMES
        or any(ord(char) < 32 or char in _INVALID_PORTABLE_PATH_CHARS for char in value)
    ):
        raise ValueError(f"{label} is not a safe path segment: {value!r}")

    return value


def safe_relative_subdir(value: str, label: str) -> str:
    """Validate a relative multi-segment path and return it normalized to forward slashes.

    Rejects absolute paths (POSIX and Windows), drive letters, and any segment that fails
    `safe_path_segment`. Raises ValueError on violation.
    """
    portable = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if portable.is_absolute() or windows.is_absolute() or windows.drive or not portable.parts:
        raise ValueError(f"{label} must be a safe relative path: {value!r}")

    parts = [safe_path_segment(part, label) for part in portable.parts]
    return "/".join(parts)
