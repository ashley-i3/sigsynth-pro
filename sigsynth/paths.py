from __future__ import annotations

from pathlib import Path
import re

# Whitelisted absolute paths for large dataset storage
ALLOWED_DATA_DIRS = [Path("/data1"), Path("/data2"), Path("/data3")]


def sanitize_macro_name(macro_name: str) -> str:
    candidate = Path(macro_name)
    if candidate.is_absolute():
        raise ValueError("Absolute paths are not allowed for macro names.")
    if len(candidate.parts) != 1:
        raise ValueError("Macro names must not include directory separators.")
    if ".." in candidate.parts:
        raise ValueError("Parent directory traversal is not allowed for macro names.")

    normalized = candidate.name
    if not normalized:
        raise ValueError("Macro name cannot be empty.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", normalized):
        raise ValueError("Macro name contains unsupported characters.")
    return normalized


def sanitize_output_dir(output_dir: str | Path, base_dir: str | Path = "output") -> Path:
    base_path = Path(base_dir)
    user_path = Path(output_dir)
    resolved_base = base_path.resolve()

    if user_path.is_absolute():
        resolved = user_path.resolve()

        # Check if path is in allowed data directories
        for allowed_dir in ALLOWED_DATA_DIRS:
            if resolved == allowed_dir or allowed_dir in resolved.parents:
                # Path is in allowed directory, return it
                return resolved

        # Otherwise check sandbox
        if resolved != resolved_base and resolved_base not in resolved.parents:
            raise ValueError(
                "Dataset output directory must stay inside the output sandbox "
                "or be within /data1, /data2, or /data3."
            )
        return resolved

    if ".." in user_path.parts:
        raise ValueError("Parent directory traversal is not allowed for dataset output directories.")

    if user_path.parts and user_path.parts[0] == base_path.name:
        user_path = Path(*user_path.parts[1:])

    resolved = (resolved_base / user_path).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError("Dataset output directory must stay inside the output sandbox.")
    return resolved
