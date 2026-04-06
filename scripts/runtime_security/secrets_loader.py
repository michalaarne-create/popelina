from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable


def load_named_secrets(*, names: Iterable[str], env: Dict[str, str] | None = None, env_file: str | Path | None = None) -> Dict[str, str]:
    source_env = dict(os.environ if env is None else env)
    loaded: Dict[str, str] = {}
    if env_file:
        path = Path(env_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = line.strip()
                if not row or row.startswith("#") or "=" not in row:
                    continue
                key, value = row.split("=", 1)
                source_env.setdefault(key.strip(), value.strip())
    for name in names:
        key = str(name or "").strip()
        if key and key in source_env:
            loaded[key] = str(source_env[key])
    return loaded
