import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
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
async def run_pipeline_endpoint(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    target_column: str | None = Form(None),
):
    for f in (train_file, test_file):
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"'{f.filename}' is not a CSV file")

    try:
        train_df = pd.read_csv(io.BytesIO(await train_file.read()))
        test_df = pd.read_csv(io.BytesIO(await test_file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    if target_column is not None and target_column not in train_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"'{target_column}' is not a column in train_file. Available: {list(train_df.columns)}",
        )

    cfg, deps = get_deps()
    try:
        result = execute_pipeline_run(
            train_df, test_df, train_file.filename, target_column, cfg, deps
        )
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