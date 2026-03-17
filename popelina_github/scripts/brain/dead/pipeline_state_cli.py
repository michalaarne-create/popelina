from __future__ import annotations

# Compatibility wrapper. Real implementation moved during brain refactor.
from .fallback.compat.pipeline_state_cli import *  # noqa: F401,F403

if __name__ == "__main__":
    from .fallback.compat.pipeline_state_cli import main as _main

    _main()
