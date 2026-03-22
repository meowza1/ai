from __future__ import annotations

import subprocess
import sys


def main() -> None:
    train_cmd = [
        sys.executable,
        "-m",
        "minmax_lhrm.train",
        "--data",
        "data/english_seed.md",
        "--steps",
        "450",
        "--out-dir",
        "artifacts/minmax-v1",
    ]
    subprocess.check_call(train_cmd)

    chat_cmd = [
        sys.executable,
        "-m",
        "minmax_lhrm.chat",
        "--model-dir",
        "artifacts/minmax-v1",
        "--temperature",
        "0.7",
        "--session-minutes",
        "1.5",
    ]
    subprocess.check_call(chat_cmd)


if __name__ == "__main__":
    main()
