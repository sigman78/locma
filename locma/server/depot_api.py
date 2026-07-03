"""/api/depot router — thin HTTP layer over locma.depot for the web panel.

push/pull hit the configured remote (GitHub Releases by default); they run as
plain sync endpoints (FastAPI executes them on its threadpool, so the event
loop is not blocked while gh transfers).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from locma import depot as dep


class PinBody(BaseModel):
    version: int


class VersionBody(BaseModel):
    version: int | None = None  # None = the pin


class PublishBody(BaseModel):
    name: str
    files: list[str]  # server-side paths, e.g. runs/my_s0.zip
    kind: str = "model"
    note: str = ""
    parents: list[str] = []
    meta: dict = {}


class GcBody(BaseModel):
    dry_run: bool = True


def _selector(version: int | None) -> str:
    return "" if version is None else str(version)


def _record_with_status(name: str) -> dict:
    rec = dep.load_record(name)
    for vrec in rec["versions"]:
        vrec["status"] = dep.version_status(rec, vrec)
        vrec["size"] = sum(e["size"] for e in vrec["files"].values())
    return rec


def depot_router() -> APIRouter:
    router = APIRouter(prefix="/api/depot")

    @router.get("")
    def list_artifacts() -> list[dict]:
        return [_record_with_status(name) for name in dep.artifact_names()]

    @router.get("/remote")
    def remote() -> dict:
        return {"remote": dep.remote_spec()}

    @router.get("/{name}")
    def show(name: str) -> dict:
        try:
            return _record_with_status(name)
        except dep.DepotError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @router.post("/{name}/pin")
    def pin(name: str, body: PinBody) -> dict:
        try:
            dep.pin(name, body.version)
        except dep.DepotError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _record_with_status(name)

    @router.post("/{name}/pull")
    def pull(name: str, body: VersionBody) -> dict:
        try:
            fetched = dep.pull(name, _selector(body.version))
        except dep.DepotError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"fetched": fetched, "record": _record_with_status(name)}

    @router.post("/{name}/push")
    def push(name: str, body: VersionBody) -> dict:
        try:
            locator = dep.push(name, _selector(body.version))
        except dep.DepotError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"locator": locator, "record": _record_with_status(name)}

    @router.post("/publish")
    def publish(body: PublishBody) -> dict:
        try:
            vrec = dep.publish(
                body.name,
                body.files,
                kind=body.kind,
                note=body.note,
                parents=body.parents,
                meta=body.meta or None,
                command=f"web: publish {body.name}",
            )
        except dep.DepotError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"version": vrec["version"], "record": _record_with_status(body.name)}

    @router.post("/gc")
    def gc(body: GcBody) -> dict:
        removed, freed = dep.gc(dry_run=body.dry_run)
        return {"removed": len(removed), "freed": freed, "dry_run": body.dry_run}

    @router.get("/resolve/{ref:path}")
    def resolve(ref: str) -> dict:
        try:
            return {"ref": ref, "path": dep.resolve_path(ref)}
        except dep.DepotError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return router
