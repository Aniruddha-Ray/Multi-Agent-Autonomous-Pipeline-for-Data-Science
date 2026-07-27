import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.api.schemas import HealthResponse, PipelineRunResponse
from src.api.dependencies import get_deps
from src.api.services import execute_pipeline_run, get_memory_snapshot, _RUN_STORE

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy")

@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    cfg, deps = get_deps()
    try:
        result = execute_pipeline_run(df, file.filename, cfg, deps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")
    return PipelineRunResponse(**{k: v for k, v in result.items() if k != "report_text"})

@router.get("/runs/{run_id}")
def get_run(run_id: str):
    run = _RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {k: v for k, v in run.items() if k != "report_text"}

@router.get("/report/{run_id}")
def get_report(run_id: str):
    run = _RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return Response(content=run["report_text"], media_type="text/markdown")

@router.get("/memory")
def get_memory(limit: int | None = None):
    cfg, deps = get_deps()
    try:
        return JSONResponse(content=get_memory_snapshot(deps, limit=limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory read failed: {exc}")