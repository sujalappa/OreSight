"""Background analysis jobs: one at a time (Earth Engine downloads are the bottleneck)."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..pipeline import STEPS, Pipeline
from . import accounts, catalog

VALIDATED_BLOCKS = ("kehalpur", "thanewasna")       # blocks with ground truth handled by scripts/validate.py
STEP_LABELS = {"acquire": "Downloading satellite data (EMIT, DEM, ASTER)", "emit_l2a": "EMIT mineral absorption features",
               "emit_l2b": "EMIT L2B mineral identification", "aster": "ASTER thermal indices (quartz)",
               "structure": "Lineaments", "alteration": "Alteration layers", "targets": "CPI, zones and field-check locations",
               "maps": "Contour maps", "validation": "Validation against ground truth"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    job_id: str
    block: str
    options: dict            # all option and parameter values
    output_name: str
    status: str = "queued"          # queued | running | done | failed
    step: str = ""
    log: list = field(default_factory=list)
    error: str = ""
    run_id: str = ""
    created: str = field(default_factory=_now)
    user: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        steps = list(self.extra.get("steps", STEPS)) + (["validation"] if self.block in VALIDATED_BLOCKS else [])
        idx = steps.index(self.step) if self.step in steps else -1
        return {"job_id": self.job_id, "block": self.block, "label": catalog.option_label(self.options),
                "options": {k: v for k, v in self.options.items() if k in catalog.OPTIONS},
                "params": {k: v for k, v in self.options.items() if k in catalog.PARAMS}, "status": self.status, "step": self.step,
                "step_label": STEP_LABELS.get(self.step, self.step), "progress": (idx + (self.status == "done")) / len(steps)
                if idx >= 0 else (1.0 if self.status == "done" else 0.0),
                "log": self.log[-200:], "error": self.error, "run_id": self.run_id, "created": self.created, "user": self.user}


class _LogHandler(logging.Handler):
    def __init__(self, job: Job):
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record):
        if record.name.startswith("geoscan"):
            self.job.log.append(f"{datetime.now():%H:%M:%S}  {record.getMessage()}")


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, block: str, options: dict, user: str = "") -> Job:
        for j in self.jobs.values():
            if j.block == block and j.options == options and j.status in ("queued", "running"):
                return j
        vid = catalog.variant_id(options)
        out = block if not vid else f"{block}__{vid}"
        job = Job(uuid.uuid4().hex[:10], block, options, out, user=user)
        accounts.record(user, "run_started", area=block, settings=catalog.option_label(options),
                        changed_parameters=catalog.changed(options), recompute_from=catalog.first_step(options))
        self.jobs[job.job_id] = job
        self.q.put(job.job_id)
        return job

    def _worker(self):
        while True:
            job = self.jobs[self.q.get()]
            job.status = "running"
            h = _LogHandler(job)
            logging.getLogger("geoscan").addHandler(h)
            logging.getLogger("geoscan").setLevel(logging.INFO)
            try:
                self._run(job)
                self._save(job, results=True)
                job.status = "done"
                accounts.record(job.user, "run_finished", area=job.block, results=job.run_id)
            except Exception as e:                           # report, keep the worker alive
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                accounts.record(job.user, "run_failed", area=job.block, step=job.step, error=job.error[:300])
                job.log.extend(traceback.format_exc().splitlines()[-16:])   # innermost frames
                self._save(job, results=False)               # keep the downloaded satellite data at least
            finally:
                logging.getLogger("geoscan").removeHandler(h)

    def _save(self, job: Job, results: bool):
        """Copy new files to permanent storage when the app runs on a host with a temporary disk."""
        from . import persist
        if not persist.enabled():
            return
        try:
            job.log.append(f"{datetime.now():%H:%M:%S}  Saving to permanent storage ...")
            persist.save_area(job.block)
            if results:
                persist.save_run(job.output_name)
            job.log.append(f"{datetime.now():%H:%M:%S}  Saved.")
        except Exception as e:                               # never fail an analysis because of storage
            job.log.append(f"WARNING: could not save to permanent storage: {e}")

    def _run(self, job: Job):
        def progress(step, msg):
            job.step = step
            job.log.append(f"{datetime.now():%H:%M:%S}  [{STEP_LABELS.get(step, step)}] {msg}")

        settings = catalog.settings_for(job.options)
        aoi = catalog.DATA / job.block / "aoi" / "aoi.geojson"
        p = Pipeline(aoi, job.block, catalog.ROOT, settings, progress=progress, output_name=job.output_name)
        start = catalog.first_step(job.options)
        base = catalog.base_run(job.block, job.options)
        out = catalog.OUTPUTS / job.output_name
        if base is not None and base.path.resolve() != out.resolve():
            # re-use everything computed before the first changed step
            steps = STEPS[STEPS.index(start):]
            job.extra["steps"] = steps
            job.log.append(f"Re-using results up to '{start}' from {base.run_id}; running {', '.join(steps)}")
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(base.path, out, ignore=shutil.ignore_patterns("_webapp_cache", "maps"))
            if STEPS.index(start) <= STEPS.index("structure"):
                from ..acquire import gee
                src = settings.structure.dem_source
                path = gee.dem_path(p.ws.raw(""), src)
                if not path.exists():
                    progress("structure", f"downloading {gee.dem_label(src)} DEM")
                    gee.initialize(settings.acquire.ee_project)
                    gee.download_dem(src, p.grid_dem, path, settings.acquire.max_request_mb, settings.acquire.retries)
            p.run(tuple(steps), clean=False)
        else:
            p.run(STEPS, clean=True)
        if job.block in VALIDATED_BLOCKS:
            progress("validation", "started")
            script = Path(__file__).resolve().parents[2] / "scripts" / "validate.py"
            args = [sys.executable, str(script), "--name", job.block]
            if job.output_name != job.block:
                args += ["--output", job.output_name]
            r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                job.log.append(r.stderr[-2000:])
                raise RuntimeError("validation failed")
            progress("validation", "done")
        (catalog.OUTPUTS / job.output_name / "created_by.json").write_text(
            json.dumps({"user": job.user, "time": _now()}), encoding="utf-8")
        run = catalog.find_run(job.block, job.options)
        if run is None:
            raise RuntimeError("run finished but its results were not found")
        job.run_id = run.run_id
