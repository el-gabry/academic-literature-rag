from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI as a real Python module."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "academic_literature_rag.cli",
            *args,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_root_help_displays_available_commands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "Academic Literature RAG" in result.stdout
    assert "{demo}" in result.stdout
    assert "demo" in result.stdout
    assert result.stderr == ""


def test_demo_help_displays_demo_options() -> None:
    result = run_cli("demo", "--help")

    assert result.returncode == 0
    assert "--format" in result.stdout
    assert "{text,json,markdown}" in result.stdout
    assert "--output-file" in result.stdout
    assert "--log-level" in result.stdout
    assert "--log-format" in result.stdout
    assert "{text,json}" in result.stdout
    assert result.stderr == ""


def test_demo_rejects_invalid_output_format() -> None:
    result = run_cli(
        "demo",
        "--format",
        "xml",
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "xml" in result.stderr


def test_demo_rejects_invalid_log_format() -> None:
    result = run_cli(
        "demo",
        "--log-format",
        "xml",
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "xml" in result.stderr


def test_demo_rejects_invalid_log_level_before_loading_app_config() -> None:
    result = run_cli(
        "demo",
        "--log-level",
        "VERBOSE",
    )

    assert result.returncode == 2
    assert "Unsupported log level: VERBOSE" in result.stderr