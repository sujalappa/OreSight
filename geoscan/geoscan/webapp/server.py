"""FastAPI server for the OreSight web app (local use: http://127.0.0.1:8000)."""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..aoi import load_aoi
from . import accounts, catalog, exports, geochem, geophysics, layers, persist
from .help_text import LAYER_INFO
from .jobs import JobManager

STATIC = Path(__file__).parent / "static"
MAX_AREA_KM2 = 1000.0
app = FastAPI(title="OreSight")


@app.middleware("http")
async def login_gate(request: Request, call_next):
    """HTTP Basic login for the hosted app (GEOSCAN_PASSWORD set); the local app runs as user 'local'.

    Sets request.state.user and logs the start of each session.
    """
    if not accounts.login_required():
        request.state.user = accounts.admin_name()
        return await call_next(request)
    if request.url.path == "/api/logout":
        # Sign-out helper: the page calls this with deliberately wrong credentials so the browser forgets the
        # real ones. No WWW-Authenticate header, so no password box pops up during the call.
        return Response("Signed out", status_code=401)
    user = None
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            u, _, p = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            if accounts.check(u, p):
                user = u
        except Exception:
            user = None
    if user is None:
        return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="OreSight"'})
    request.state.user = user
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0].strip()
    accounts.seen(user, ip, request.headers.get("user-agent", ""))
    return await call_next(request)


def _user(request: Request) -> str:
    return getattr(request.state, "user", accounts.admin_name())


def _admin(request: Request) -> str:
    u = _user(request)
    if not accounts.is_admin(u):
        raise HTTPException(403, "Administrator only")
    return u


app.mount("/static", StaticFiles(directory=STATIC), name="static")
jobs = JobManager()


def _run(run_id: str) -> catalog.Run:
    try:
        return catalog.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"No results '{run_id}'")


def _safe(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if base.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404, "file not found")
    return p


@app.get("/", response_class=HTMLResponse)
def index():
    """The page, with a version stamp on its script and stylesheet so browsers never run an outdated copy.

    The stamp is the files' modification time: after every update the address changes, and the browser
    fetches the new code without anyone having to force a refresh.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "style.css"):
        html = html.replace(f'/static/{name}"', f'/static/{name}?v={int((STATIC / name).stat().st_mtime)}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.post("/api/signout")
def signout(request: Request):
    u = _user(request)
    accounts.record(u, "signed_out")
    accounts.forget(u)
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    u = _user(request)
    return {"user": u, "admin": accounts.is_admin(u), "login": accounts.login_required()}


class NewUser(BaseModel):
    user: str
    password: str
    note: str = ""


@app.get("/api/admin/users")
def admin_users(request: Request):
    _admin(request)
    return accounts.list_users()


@app.post("/api/admin/users")
def admin_add_user(req: NewUser, request: Request):
    a = _admin(request)
    try:
        accounts.add_user(req.user, req.password, req.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    accounts.record(a, "user_added", account=req.user.strip(), note=req.note)
    return accounts.list_users()


@app.delete("/api/admin/users/{name}")
def admin_remove_user(name: str, request: Request):
    a = _admin(request)
    if not accounts.remove_user(name):
        raise HTTPException(404, "No such user")
    accounts.record(a, "user_removed", account=name)
    return accounts.list_users()


@app.get("/api/admin/log")
def admin_log(request: Request, limit: int = 500):
    _admin(request)
    return list(reversed(accounts.read_log(limit)))


@app.get("/api/admin/log.csv")
def admin_log_csv(request: Request):
    _admin(request)
    return Response(accounts.log_csv(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="geoscan_activity_log.csv"'})


@app.get("/api/config")
def config():
    return {"options": catalog.public_options(), "params": catalog.public_params(), "layers": {k: v["label"] for k, v in layers.RASTER_LAYERS.items()},
            "layer_info": LAYER_INFO,
            "formats": exports.FORMATS, "max_area_km2": MAX_AREA_KM2, "permanent_storage": persist.enabled()}


@app.get("/api/blocks")
def blocks():
    return catalog.blocks()


@app.post("/api/blocks")
async def new_block(request: Request, file: UploadFile = File(...), name: str = Form("")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".kml", ".kmz", ".geojson", ".json"):
        raise HTTPException(400, "Upload a .kml, .kmz or .geojson polygon")
    block = catalog.slug(name or Path(file.filename).stem)
    aoi_dir = catalog.DATA / block / "aoi"
    tmp = catalog.DATA / f"_upload_{block}{suffix}"
    catalog.DATA.mkdir(exist_ok=True)
    content = await file.read()
    tmp.write_bytes(content)
    who, upload = _user(request), {"file": file.filename, "bytes": len(content), "area": block}
    try:
        aoi = load_aoi(tmp, block)
        area = aoi.area_km2()
    except Exception as e:
        tmp.unlink(missing_ok=True)
        accounts.record(who, "upload_area", **upload, result="rejected: not a readable polygon")
        raise HTTPException(400, f"Could not read a polygon from the file: {e}")
    if area > MAX_AREA_KM2:
        tmp.unlink(missing_ok=True)
        accounts.record(who, "upload_area", **upload, area_km2=round(area, 2), result="rejected: too large")
        raise HTTPException(400, f"Area is {area:.0f} km²; the limit is {MAX_AREA_KM2:.0f} km²")
    if (aoi_dir / "aoi.geojson").exists():
        old = load_aoi(aoi_dir / "aoi.geojson", block)
        diff = old.geometry.symmetric_difference(aoi.geometry).area / max(old.geometry.area, 1e-12)
        tmp.unlink(missing_ok=True)
        accounts.record(who, "upload_area", **upload, area_km2=round(area, 2),
                        result="name already used with a different outline" if diff > 0.01 else "same as existing area")
        if diff > 0.01:
            raise HTTPException(409, f"An area called '{block}' already exists with a different outline. "
                                     "Choose another name.")
        return {"block": block, "existing": True, "area_km2": round(area, 2)}
    aoi_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), aoi_dir / f"{block}{suffix}")
    aoi.to_geojson(aoi_dir / "aoi.geojson")
    (aoi_dir / "uploaded_by.json").write_text(json.dumps({"user": who, "file": file.filename}), encoding="utf-8")
    c = aoi.geometry.centroid
    accounts.record(who, "upload_area", **upload, area_km2=round(area, 2), result="new area",
                    centre_lat=round(c.y, 5), centre_lon=round(c.x, 5))
    persist.save_async(persist.save_area, block)
    return {"block": block, "existing": False, "area_km2": round(area, 2)}


def _ref(block: str, key: str) -> None:
    if not (catalog.DATA / block / "aoi" / "aoi.geojson").exists() or key not in geophysics.LAYERS \
            or geophysics.stations(block, key) is None:
        raise HTTPException(404, "no such reference layer for this area")


@app.get("/api/blocks/{block}/reference/{key}/overlay.png")
def reference_png(block: str, key: str):
    _ref(block, key)
    geophysics.overlay(block, key)
    return FileResponse(geophysics._cache(block) / f"{key}.png", media_type="image/png")


@app.get("/api/blocks/{block}/reference/{key}/overlay")
def reference_overlay(block: str, key: str):
    _ref(block, key)
    return {**geophysics.overlay(block, key), "url": f"/api/blocks/{block}/reference/{key}/overlay.png"}


@app.get("/api/blocks/{block}/reference/{key}/stations.geojson")
def reference_stations(block: str, key: str):
    _ref(block, key)
    return Response(layers.geojson(geophysics.stations(block, key)), media_type="application/geo+json")


@app.get("/api/blocks/{block}/reference/{key}/contours.geojson")
def reference_contours(block: str, key: str):
    _ref(block, key)
    c = geophysics.contours(block, key)
    if c is None:
        raise HTTPException(404, "too few stations for contours")
    return Response(layers.geojson(c), media_type="application/geo+json")


@app.get("/api/blocks/{block}/reference/{key}/map.png")
def reference_map(block: str, key: str):
    _ref(block, key)
    return FileResponse(geophysics.figure(block, key), media_type="image/png")


def _geochem(block: str):
    if not (catalog.DATA / block / "aoi" / "aoi.geojson").exists():
        raise HTTPException(404, "unknown area")
    g = geochem.samples(block)
    if g is None:
        raise HTTPException(404, geochem.info(block)["reason"])
    return g


@app.get("/api/blocks/{block}/geochem/samples.geojson")
def geochem_samples(block: str):
    return Response(layers.geojson(_geochem(block)), media_type="application/geo+json")


@app.get("/api/blocks/{block}/geochem/{element}/map.png")
def geochem_map(block: str, element: str):
    _geochem(block)
    path = geochem.figure(block, element)
    if path is None:
        raise HTTPException(404, "element not analysed")
    return FileResponse(path, media_type="image/png")


class RunRequest(BaseModel):
    block: str
    options: dict = {}
    params: dict = {}


def _full_values(opts: dict, params: dict) -> dict:
    full = catalog.default_values()
    for k, v in opts.items():
        if k not in catalog.OPTIONS or not catalog.OPTIONS[k]["public"]:
            raise HTTPException(400, f"Unknown option {k}")
        v = catalog._norm(v)
        if v not in [catalog._norm(c) for c in catalog.OPTIONS[k]["choices"]]:
            raise HTTPException(400, f"Invalid value {v} for {k}")
        full[k] = v
    for k, v in params.items():
        p = catalog.PARAMS.get(k)
        if p is None or not p["public"]:
            raise HTTPException(400, f"Unknown parameter {k}")
        try:
            v = bool(v) if p["kind"] == "bool" else int(v) if p["kind"] == "int" else catalog._norm(float(v))
        except (TypeError, ValueError):
            raise HTTPException(400, f"{p['label']}: not a number")
        if p["kind"] != "bool" and not (p["min"] <= v <= p["max"]):
            raise HTTPException(400, f"{p['label']} must be between {p['min']} and {p['max']}")
        full[k] = v
    problems = rule_problems(full)
    if problems:
        raise HTTPException(400, " ".join(problems))
    return full


def rule_problems(v: dict) -> list[str]:
    out = []
    w = sum(v[f"targets.w_{m}"] for m in ("muscovite", "quartz", "gossan", "chlorite"))
    if abs(w - 1) > 0.005:
        out.append(f"The four mineral weights add up to {w:.2f}; they must add up to 1.00.")
    if not v["targets.tier1"] > v["targets.tier2"] > v["targets.tier3"]:
        out.append("Tier thresholds must decrease: Tier 1 > Tier 2 > Tier 3.")
    if not v["alteration.sigma_halo_m"] > v["alteration.sigma_core_m"]:
        out.append("The halo kernel must be larger than the core kernel.")
    if not v["alteration.sericite_max_nm"] > v["alteration.sericite_min_nm"]:
        out.append("The white-mica position window must run from a lower to a higher wavelength.")
    if not v["structure.canny_high_quantile"] > v["structure.canny_low_quantile"]:
        out.append("The high edge quantile must be above the low one.")
    if not v["structure.min_length_m"] >= v["structure.min_segment_m"]:
        out.append("The minimum lineament length must be at least the minimum Hough segment.")
    return out


@app.post("/api/runs")
def start_run(req: RunRequest, request: Request):
    if not (catalog.DATA / req.block / "aoi" / "aoi.geojson").exists():
        raise HTTPException(404, f"Unknown area {req.block}")
    values = _full_values(req.options, req.params)
    run = catalog.find_run(req.block, values)
    if run:
        return {"run": run.to_dict()}
    job = jobs.submit(req.block, values, _user(request))
    return {"job": job.to_dict(), "first_step": catalog.first_step(values)}


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict() for j in jobs.jobs.values()]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in jobs.jobs:
        raise HTTPException(404, "unknown job")
    return jobs.jobs[job_id].to_dict()


@app.get("/api/runs/{run_id}")
def run_info(run_id: str, request: Request):
    r = _run(run_id)
    accounts.record(_user(request), "open_results", area=r.block, results=r.run_id, settings=r.to_dict()["label"])
    maps = [{"file": f, "label": lab} for f, lab in exports.MAPS.items() if (r.path / "maps" / f).exists()]
    vmaps = [f"maps/{p.name}" for p in sorted((r.validation / "maps").glob("*.png"))] if r.validation else []
    return {**r.to_dict(), "maps": maps, "validation_maps": vmaps, "downloads": exports.items(r), "reference": geophysics.info(r.block, include_missing=True) + [geochem.info(r.block)],
            "layers": [k for k, v in layers.RASTER_LAYERS.items() if (r.path / v["file"]).exists()]}


@app.get("/api/runs/{run_id}/overlay/{key}.png")
def overlay_png(run_id: str, key: str):
    r = _run(run_id)
    layers.overlay(r.path, key)
    return FileResponse(r.path / "_webapp_cache" / f"{key}.png", media_type="image/png")


@app.get("/api/runs/{run_id}/overlay/{key}")
def overlay(run_id: str, key: str):
    r = _run(run_id)
    if key not in layers.RASTER_LAYERS and not (key.startswith("na_") and key[3:] in layers.RASTER_LAYERS):
        raise HTTPException(404, "unknown layer")
    info = layers.overlay(r.path, key)
    return {**info, "url": f"/api/runs/{run_id}/overlay/{key}.png"}


@app.get("/api/runs/{run_id}/contours/{key}.geojson")
def contours(run_id: str, key: str):
    r = _run(run_id)
    if key not in layers.RASTER_LAYERS or not layers.RASTER_LAYERS[key]["levels"]:
        raise HTTPException(404, "no contours for this layer")
    return Response(layers.geojson(layers.contours(r.path, key)), media_type="application/geo+json")


@app.get("/api/runs/{run_id}/vector/{key}.geojson")
def vector(run_id: str, key: str):
    r = _run(run_id)
    try:
        return Response(layers.geojson(layers.vector(r.path, key)), media_type="application/geo+json")
    except Exception:
        raise HTTPException(404, "vector not found")


@app.get("/api/runs/{run_id}/map/{name}")
def map_png(run_id: str, name: str):
    r = _run(run_id)
    if name not in exports.MAPS:
        raise HTTPException(404, "map not found")
    return FileResponse(_safe(r.path / "maps", name), media_type="image/png")


@app.get("/api/runs/{run_id}/targets")
def targets(run_id: str):
    r = _run(run_id)
    path = r.path / "tables" / "field_check_locations.csv"
    df = pd.read_csv(path if path.exists() else r.path / "tables" / "top50_targets.csv")
    df = df.drop(columns=[c for c in df.columns if c in layers.DROP_COLUMNS])
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/runs/{run_id}/zones")
def zones(run_id: str):
    """Ranked prospective zones (empty for results made before zones existed)."""
    r = _run(run_id)
    path = r.path / "tables" / "prospective_zones.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/runs/{run_id}/validation")
def validation(run_id: str):
    r = _run(run_id)
    if not r.validation:
        raise HTTPException(404, "no validation for this area")
    return {"summary": (r.validation / "summary.md").read_text(encoding="utf-8")}


@app.get("/api/runs/{run_id}/validation/{rel:path}")
def validation_file(run_id: str, rel: str):
    r = _run(run_id)
    if not r.validation:
        raise HTTPException(404, "no validation")
    return FileResponse(_safe(r.validation, rel))


class DownloadRequest(BaseModel):
    items: list[str]
    formats: list[str] = ["shp"]


@app.post("/api/runs/{run_id}/download")
def download(run_id: str, req: DownloadRequest, request: Request):
    r = _run(run_id)
    if not req.items:
        raise HTTPException(400, "nothing selected")
    data = exports.build_zip(r, req.items, req.formats)
    accounts.record(_user(request), "download", area=r.block, results=r.run_id, items=req.items, formats=req.formats,
                    megabytes=round(len(data) / 1e6, 2))
    name = f"oresight_{r.block}_{catalog.variant_id(r.values) or 'recommended'}.zip"
    return StreamingResponse(io.BytesIO(data), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
