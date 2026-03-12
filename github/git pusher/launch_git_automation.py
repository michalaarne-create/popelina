from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COUNTER_FILE_NAME = ".main_launch_counter.json"
COUNTER_DIR = Path(__file__).resolve().parent
TOKEN_FILE = COUNTER_DIR.parent / "github_token" / "token.txt"
DEFAULT_PUSH_EVERY = 10
DEFAULT_NEW_REPO_EVERY = 50
DEFAULT_BASE_REPO_NAME = "popelina"
DEFAULT_OWNER = "michalaarne-create"


@dataclass
class LaunchState:
    launch_count: int = 0
    last_push_launch: int = 0
    last_new_repo_launch: int = 0
    created_repos: List[Dict[str, Any]] = field(default_factory=list)
    last_error: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LaunchState":
        return cls(
            launch_count=int(raw.get("launch_count") or 0),
            last_push_launch=int(raw.get("last_push_launch") or 0),
            last_new_repo_launch=int(raw.get("last_new_repo_launch") or 0),
            created_repos=list(raw.get("created_repos") or []),
            last_error=str(raw.get("last_error") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "launch_count": int(self.launch_count),
            "last_push_launch": int(self.last_push_launch),
            "last_new_repo_launch": int(self.last_new_repo_launch),
            "created_repos": list(self.created_repos),
            "last_error": str(self.last_error or ""),
            "updated_at": int(time.time()),
        }


def _read_state(path: Path) -> LaunchState:
    try:
        if not path.exists():
            return LaunchState()
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(raw, dict):
            return LaunchState.from_dict(raw)
    except Exception:
        pass
    return LaunchState()


def _write_state(path: Path, state: LaunchState) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_git(repo_root: Path, *args: str) -> Tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _is_git_repo(repo_root: Path) -> bool:
    rc, out, _ = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    return rc == 0 and out.strip().lower() == "true"


def _stage_python_only(repo_root: Path) -> None:
    _run_git(repo_root, "add", "--all", "--", "*.py")


def _has_staged_python_changes(repo_root: Path) -> bool:
    rc, out, _ = _run_git(repo_root, "diff", "--cached", "--name-only", "--", "*.py")
    if rc != 0:
        return False
    return bool(out.strip())


def _current_branch(repo_root: Path) -> str:
    rc, out, _ = _run_git(repo_root, "branch", "--show-current")
    if rc == 0 and out.strip():
        return out.strip()
    return "main"


def _remote_origin_exists(repo_root: Path) -> bool:
    rc, _, _ = _run_git(repo_root, "remote", "get-url", "origin")
    return rc == 0


def _git_commit_push_python(repo_root: Path, launch_count: int) -> None:
    _stage_python_only(repo_root)

    if _has_staged_python_changes(repo_root):
        _run_git(repo_root, "commit", "-m", f"auto: python checkpoint after {launch_count} main.py launches")

    if _remote_origin_exists(repo_root):
        branch = _current_branch(repo_root)
        _run_git(repo_root, "push", "origin", branch)


def _origin_owner_from_url(repo_root: Path) -> str:
    rc, out, _ = _run_git(repo_root, "remote", "get-url", "origin")
    if rc != 0:
        return DEFAULT_OWNER
    url = out.strip()
    if not url:
        return DEFAULT_OWNER

    # Examples:
    # git@github.com:owner/repo.git
    # https://github.com/owner/repo.git
    try:
        if "@github.com:" in url:
            rhs = url.split("@github.com:", 1)[1]
            owner = rhs.split("/", 1)[0].strip()
            return owner or DEFAULT_OWNER
        parsed = urllib.parse.urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return parts[0].strip() or DEFAULT_OWNER
    except Exception:
        pass
    return DEFAULT_OWNER


def _api_create_github_repo(owner: str, repo_name: str, token: str) -> Dict[str, Any]:
    payload = json.dumps({"name": repo_name, "private": False, "auto_init": False}).encode("utf-8")

    # For user accounts this endpoint is enough; for orgs token may need org permissions.
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "main-launch-counter",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("GitHub API returned non-dict response")

    # If token user differs from origin owner, still continue with created repo owner.
    return obj


def _add_unique_remote(repo_root: Path, base_name: str, url: str) -> str:
    name = base_name
    i = 1
    while True:
        rc, _, _ = _run_git(repo_root, "remote", "get-url", name)
        if rc != 0:
            break
        i += 1
        name = f"{base_name}_{i}"
    _run_git(repo_root, "remote", "add", name, url)
    return name


def _load_github_token() -> str:
    # Prefer token file in github/github_token/token.txt, then fallback to env.
    try:
        if TOKEN_FILE.exists():
            token = str(TOKEN_FILE.read_text(encoding="utf-8", errors="replace") or "").strip()
            if token:
                return token
    except Exception:
        pass
    return str(os.environ.get("GITHUB_TOKEN") or "").strip()


def _create_and_push_new_repo(repo_root: Path, launch_count: int) -> Optional[Dict[str, Any]]:
    token = _load_github_token()
    if not token:
        raise RuntimeError(f"Brak tokena GitHub: {TOKEN_FILE} lub GITHUB_TOKEN")

    owner = _origin_owner_from_url(repo_root)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    repo_name = f"{DEFAULT_BASE_REPO_NAME}-auto-{launch_count}-{ts}"

    created = _api_create_github_repo(owner=owner, repo_name=repo_name, token=token)
    clone_url = str(created.get("ssh_url") or created.get("clone_url") or "").strip()
    html_url = str(created.get("html_url") or "").strip()
    if not clone_url:
        raise RuntimeError("Nowe repo utworzone, ale brak URL do push")

    remote_name = _add_unique_remote(repo_root, f"auto_repo_{launch_count}", clone_url)
    branch = _current_branch(repo_root)
    rc, _, err = _run_git(repo_root, "push", remote_name, f"HEAD:{branch}")
    if rc != 0:
        raise RuntimeError(f"Push do nowego repo nieudany: {err.strip()}")

    return {
        "launch_count": int(launch_count),
        "owner_hint": owner,
        "repo_name": repo_name,
        "remote_name": remote_name,
        "remote_url": clone_url,
        "html_url": html_url,
        "created_at": int(time.time()),
    }


def register_main_launch(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    # Keep launch counter state next to this automation script so it persists
    # in the dedicated git-pusher directory, regardless of repo_root.
    state_path = COUNTER_DIR / COUNTER_FILE_NAME
    state = _read_state(state_path)

    state.launch_count = int(state.launch_count) + 1
    current_count = int(state.launch_count)

    try:
        if _is_git_repo(root):
            if (current_count % DEFAULT_PUSH_EVERY == 0) and (state.last_push_launch < current_count):
                _git_commit_push_python(root, current_count)
                state.last_push_launch = current_count

            if (current_count % DEFAULT_NEW_REPO_EVERY == 0) and (state.last_new_repo_launch < current_count):
                created = _create_and_push_new_repo(root, current_count)
                if created:
                    state.created_repos.append(created)
                state.last_new_repo_launch = current_count
        state.last_error = ""
    except Exception as exc:
        state.last_error = str(exc)

    _write_state(state_path, state)
    return state.to_dict()
