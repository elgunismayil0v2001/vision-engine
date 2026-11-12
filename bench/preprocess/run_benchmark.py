from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the C++ preprocessing benchmark and save reproducible metadata."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def git_hash(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def optional_command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run() -> dict[str, object]:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    if min(args.width, args.height, args.iterations, args.threads) < 1:
        raise ValueError("width, height, iterations, and threads must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")

    command = [
        str(args.binary),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--threads",
        str(args.threads),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    result: dict[str, object] = json.loads(completed.stdout)
    metadata = result["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("benchmark output is missing metadata")
    metadata.update(
        {
            "git_hash": git_hash(repository_root),
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
