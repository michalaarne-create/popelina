from __future__ import annotations
import subprocess
from pathlib import Path

def main() -> None:
    # --binary: jeśli kiedykolwiek będziesz patchował pliki binarne, git to ogarnie
    proc = subprocess.run(
        ["git", "diff", "--binary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode not in (0, 1):  # 1 bywa dla "differences found" w niektórych gitach
        raise SystemExit(proc.stderr.decode("utf-8", errors="replace"))

    Path("fix.patch").write_bytes(proc.stdout)
    print("Wrote fix.patch")

if __name__ == "__main__":
    main()
