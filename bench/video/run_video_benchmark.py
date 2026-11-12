from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CUDA video pipeline benchmark and save metadata."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=16)
    return parser.parse_args()


def optional_command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run() -> dict[str, object]:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    if min(args.width, args.height, args.frames, args.warmup_frames) < 1:
        raise ValueError("dimensions and frame counts must be positive")
    completed = subprocess.run(
        [
            str(args.binary),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--frames",
            str(args.frames),
            "--warmup-frames",
            str(args.warmup_frames),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    result: dict[str, object] = json.loads(completed.stdout)
    metadata = result["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("video benchmark output is missing metadata")
    metadata.update(
        {
            "git_hash": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repository_root,
            ).stdout.strip(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "nvcc": optional_command_output(["nvcc", "--version"]),
            "nvidia_smi": optional_command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
