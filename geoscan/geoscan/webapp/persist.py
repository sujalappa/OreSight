"""Permanent storage for a hosted app whose own disk is wiped on restart (e.g. a free Hugging Face Space).

Enabled when both environment variables are set:
    GEOSCAN_STORE_REPO   a private Hugging Face *dataset* repo, e.g. "username/geoscan-store"
    HF_TOKEN             an access token with write permission to that repo

The dataset mirrors the workspace: data/<area>/{aoi,raw,ground_truth}, outputs/<run>/ and _admin/
(user accounts with hashed passwords, activity log).
  restore()        at start-up: download everything into the workspace
  save_area(area)  after an upload or a data download
  save_run(name)   after an analysis (results folder and its validation folder)
Without the variables every function does nothing, so the local app is unchanged.
"""

from __future__ import annotations

import logging
import os
import threading

from . import catalog

log = logging.getLogger("geoscan.persist")
IGNORE = ["**/_webapp_cache/**", "**/processed/**", "**/__pycache__/**", "**/_upload_*"]
_lock = threading.Lock()          # one upload at a time (the hub rejects concurrent commits)


def repo() -> str | None:
    return os.environ.get("GEOSCAN_STORE_REPO") if os.environ.get("HF_TOKEN") else None


def enabled() -> bool:
    return repo() is not None


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def restore() -> int:
    """Download the stored workspace. Returns the number of files present afterwards (0 if disabled)."""
    if not enabled():
        return 0
    from huggingface_hub import snapshot_download
    catalog.ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo(), repo_type="dataset", local_dir=str(catalog.ROOT), token=os.environ["HF_TOKEN"],
                      allow_patterns=["data/**", "outputs/**", "_admin/**"])
    n = sum(1 for p in catalog.ROOT.rglob("*") if p.is_file())
    log.info("Restored workspace from %s (%d files)", repo(), n)
    return n


def _upload(folder, message: str, mirror: bool = False) -> bool:
    """mirror=True also deletes stored files that no longer exist locally (results folders are rebuilt whole)."""
    if not enabled() or not folder.exists():
        return False
    rel = folder.relative_to(catalog.ROOT).as_posix()
    with _lock:
        _api().upload_folder(repo_id=repo(), repo_type="dataset", folder_path=str(folder), path_in_repo=rel,
                             commit_message=message, ignore_patterns=IGNORE,
                             delete_patterns=["*"] if mirror else None)
    log.info("Saved %s to %s", rel, repo())
    return True


def save_area(area: str) -> bool:
    return _upload(catalog.DATA / area, f"data: {area}")


def save_run(output_name: str) -> bool:
    ok = _upload(catalog.OUTPUTS / output_name, f"results: {output_name}", mirror=True)
    _upload(catalog.OUTPUTS / f"{output_name}_validation", f"validation: {output_name}", mirror=True)
    return ok


def save_admin() -> bool:
    """User accounts (hashed passwords) and the activity log."""
    return _upload(catalog.ROOT / "_admin", "accounts / activity log")


def save_async(fn, *args) -> None:
    """Run a save in the background; failures are logged, never raised into the request."""
    def run():
        try:
            fn(*args)
        except Exception as e:      # storage problems must not break the app
            log.warning("Could not save to permanent storage: %s", e)
    threading.Thread(target=run, daemon=True).start()
