"""Thực thi Manim local hoặc trong container cô lập theo runtime policy."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import get_runtime_policy


@dataclass(slots=True)
class ManimExecutionResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _run(command: list[str], *, cwd: Path | None, timeout: int, env: dict[str, str] | None = None) -> ManimExecutionResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return ManimExecutionResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return ManimExecutionResult(
            124,
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            f"Execution timed out after {timeout} seconds.",
            True,
        )
    except FileNotFoundError as exc:
        return ManimExecutionResult(127, "", str(exc))


def execute_manim_script(
    script: Path,
    media_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    timeout: int,
    local_env: dict[str, str] | None = None,
) -> ManimExecutionResult:
    """Render script theo backend; container là bắt buộc khi APP_ENV=production."""
    policy = get_runtime_policy().manim
    media_dir.mkdir(parents=True, exist_ok=True)
    source_dir = script.parent.resolve()
    media_dir = media_dir.resolve()
    actual_timeout = min(int(timeout), int(policy.timeout_seconds))

    if policy.backend == "local":
        command = [
            sys.executable, "-m", "manim", "render", "-a", "--format", "mp4",
            "-r", f"{int(width)},{int(height)}", "--fps", str(int(fps)),
            "--media_dir", str(media_dir), script.name,
        ]
        return _run(command, cwd=source_dir, timeout=actual_timeout, env=local_env)

    # Chỉ thư mục output tạm được mở quyền ghi cho UID không đặc quyền trong container.
    try:
        media_dir.chmod(0o777)
    except OSError:
        pass
    network = "bridge" if policy.network else "none"
    command = [
        "docker", "run", "--rm",
        "--network", network,
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=768m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(policy.pids_limit),
        "--memory", f"{policy.memory_mb}m",
        "--cpus", str(policy.cpus),
        "--user", "1000:1000",
        "-e", "HOME=/tmp",
        "-e", "MPLCONFIGDIR=/tmp/matplotlib",
        "-v", f"{source_dir}:/workspace:ro",
        "-v", f"{media_dir}:/output:rw",
        "-w", "/workspace",
        policy.image,
        "manim", "render", "-a", "--format", "mp4",
        "-r", f"{int(width)},{int(height)}", "--fps", str(int(fps)),
        "--media_dir", "/output", script.name,
    ]
    return _run(command, cwd=None, timeout=actual_timeout)
