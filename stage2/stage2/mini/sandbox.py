"""Materialize mini repos and execute shell commands in an isolated working directory."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from stage2.mini.catalog import MiniInstance


def materialize_repo(instance: MiniInstance, work_root: Path) -> Path:
    """Copy a fresh buggy repo into work_root/<instance_id>/."""
    repo_dir = work_root / instance.instance_id
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in instance.files.items():
        dest = repo_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return repo_dir


def run_command(
    command: str,
    *,
    cwd: Path,
    timeout_s: int = 60,
    max_output_chars: int = 12_000,
) -> tuple[int, str]:
    """Run a shell command in cwd; return (exit_code, combined_output)."""
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout_s}s."

    output = (proc.stdout or "") + (proc.stderr or "")
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + "\n...[truncated]..."
    return proc.returncode, output
