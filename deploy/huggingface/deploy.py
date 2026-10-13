"""Deploy the OreSight web app to a free Hugging Face Docker Space.

    python deploy/huggingface/deploy.py --password "choose-a-password" --gee-key path/to/service-account.json

What it does
  1. Creates a PRIVATE dataset (<you>/geoscan-store) and uploads the workspace: data/<area>/{aoi,raw,ground_truth}
     and every result the app lists (outputs/...). This is the app's permanent storage.
  2. Creates the Space (<you>/OreSight) and uploads the code + Dockerfile.
  3. Sets the Space secrets / variables (password, storage repo, token, Earth Engine key).

Needs a Hugging Face access token with WRITE permission: set HF_TOKEN, or run `hf auth login` first.
Run with --dry-run to see what would be uploaded without contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # phase-1-copper-exploration
sys.path.insert(0, str(ROOT / "geoscan"))
from geoscan.webapp import catalog  # noqa: E402
from geoscan.webapp.persist import IGNORE  # noqa: E402

SCRIPTS = ("validate.py", "compare_lithology.py", "compare_methods.py")


def size_mb(folder: Path) -> float:
    skip = ("_webapp_cache", "processed", "__pycache__")
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and not any(s in f.parts for s in skip)) / 1e6


def workspace_folders() -> list[Path]:
    """data/<area> for every area, and outputs/<run> (+ validation) for every run the app lists."""
    out = [d for d in sorted(catalog.DATA.iterdir()) if (d / "aoi" / "aoi.geojson").exists()]
    for r in catalog.runs():
        out.append(r.path)
        if r.validation:
            out.append(r.validation)
    return out


def stage_code(dst: Path, sdk: str = "gradio") -> None:
    if sdk == "gradio":         # free on ZeroGPU hardware; app.py starts the OreSight server
        for f in ("app.py", "README.md", "requirements.txt", "packages.txt"):
            shutil.copy2(HERE / "gradio" / f, dst / f)
    else:                       # Docker Space (needs a paid Hugging Face plan)
        for f in ("Dockerfile", "README.md", "requirements-web.txt"):
            shutil.copy2(HERE / f, dst / f)
    app = dst / "geoscan"
    shutil.copytree(ROOT / "geoscan" / "geoscan", app / "geoscan", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "geoscan" / "config", app / "config")
    (app / "scripts").mkdir()
    for s in SCRIPTS:
        shutil.copy2(ROOT / "geoscan" / "scripts" / s, app / "scripts" / s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="Space id, default <you>/OreSight")
    ap.add_argument("--store", help="storage dataset id, default <you>/geoscan-store")
    ap.add_argument("--password", default=os.environ.get("GEOSCAN_PASSWORD"), help="login password for the app")
    ap.add_argument("--gee-key", help="Earth Engine service-account JSON key file")
    ap.add_argument("--public", action="store_true",
                    help="make the Space public (anyone can open the address and see the code; the app still "
                         "asks for the password, and the data stay private). Default: private Space.")
    ap.add_argument("--sdk", choices=["gradio", "docker"], default="gradio",
                    help="gradio = free Space on ZeroGPU hardware (default); docker = paid plan")
    ap.add_argument("--skip-data", action="store_true", help="do not upload the workspace (code update only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    folders = workspace_folders()
    print("Workspace to store permanently:")
    for f in folders:
        print(f"  {f.relative_to(ROOT).as_posix():55s} {size_mb(f):7.1f} MB")
    print(f"  total {sum(size_mb(f) for f in folders):.0f} MB")
    with tempfile.TemporaryDirectory() as tmp:
        stage_code(Path(tmp), a.sdk)
        n = sum(1 for p in Path(tmp).rglob("*") if p.is_file())
        print(f"Code to upload to the Space: {n} files, {size_mb(Path(tmp)):.1f} MB")
        if a.dry_run:
            print("\nDry run: nothing was uploaded.")
            return
        if not a.password:
            raise SystemExit("Give --password (the login password for the app).")

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        me = api.whoami()["name"]
        from huggingface_hub import get_token
        token = os.environ.get("HF_TOKEN") or get_token()      # saved by `hf auth login`
        if not token:
            raise SystemExit("No Hugging Face token found. Run: hf auth login")
        space, store = a.space or f"{me}/OreSight", a.store or f"{me}/geoscan-store"

        api.create_repo(store, repo_type="dataset", private=True, exist_ok=True)
        if not a.skip_data:
            for f in folders:
                rel = f.relative_to(ROOT).as_posix()
                print("uploading", rel, flush=True)
                api.upload_folder(repo_id=store, repo_type="dataset", folder_path=str(f), path_in_repo=rel,
                                  commit_message=f"workspace: {rel}", ignore_patterns=IGNORE)

        if a.sdk == "gradio":
            api.create_repo(space, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                            private=not a.public, exist_ok=True)
        else:
            api.create_repo(space, repo_type="space", space_sdk="docker", private=not a.public, exist_ok=True)
        api.add_space_variable(space, "GEOSCAN_STORE_REPO", store)
        api.add_space_secret(space, "HF_TOKEN", token)
        api.add_space_secret(space, "GEOSCAN_PASSWORD", a.password)
        if a.gee_key:
            api.add_space_secret(space, "GEE_SERVICE_ACCOUNT_KEY", Path(a.gee_key).read_text(encoding="utf-8"))
        else:
            print("NOTE: no --gee-key given. Existing results work, but NEW analyses need the Earth Engine key.")
        print("uploading code to the Space", flush=True)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp, commit_message="OreSight web app",
                          delete_patterns=["*"])

    user, name = space.split("/")
    print(f"\nDone. The Space builds for about 5-10 minutes the first time.")
    print(f"  Build log / settings : https://huggingface.co/spaces/{space}")
    print(f"  App address          : https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")
    print(f"  Login                : user name 'geoscan', the password you chose")
    if not a.public:
        print("  The Space is private: only Hugging Face accounts you give access to can open it.")


if __name__ == "__main__":
    main()
