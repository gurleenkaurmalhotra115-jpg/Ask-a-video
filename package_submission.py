"""Create a clean, standardized submission ZIP bundle for university / evaluation portals."""

from __future__ import annotations

import os
import zipfile

IGNORED_DIRS = {
    "venv",
    ".venv",
    "env",
    ".git",
    "qdrant_data",
    "__pycache__",
    ".pytest_cache",
    "scratch",
    "logs",
    "cache",
}

IGNORED_FILES = {
    ".env",
    "VidQuery_Submission.zip",
}


def package_project(output_zip: str = "VidQuery_Submission.zip") -> str:
    root_dir = os.path.abspath(".")
    print(f"Creating submission package: {output_zip} ...")

    file_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune ignored directories
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

            for filename in filenames:
                if filename in IGNORED_FILES or filename.endswith((".pyc", ".pyo")):
                    continue

                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, root_dir)
                zf.write(abs_path, rel_path)
                file_count += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {file_count} files into {output_zip} ({size_mb:.2f} MB)")
    return output_zip


if __name__ == "__main__":
    package_project()
