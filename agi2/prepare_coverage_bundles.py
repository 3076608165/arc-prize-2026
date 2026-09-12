"""Rebuild the matched diagnostic pair and full-budget audit bundles locally."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for augment in (16, 8):
        for diagnostic in (True, False):
            folder = f"coverage-benchmark{augment}" if diagnostic else f"audited{augment}"
            arguments = [sys.executable, str(ROOT / "agi2/repro_nvarc.py"),
                "--hours", "1" if diagnostic else "11.83", "--augment", str(augment),
                "--audit-coverage", "--worker-hash-seed", "0",
                "--out", str(ROOT / "agi2/out" / folder)]
            if diagnostic:
                arguments.extend(["--full-eval", "--kernel-id",
                    f"uts3125025522741/arc-agi-2-coverage-{augment}",
                    "--title", f"ARC AGI 2 Coverage {augment}"])
            subprocess.run(arguments, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
