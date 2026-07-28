import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse
import sys
from pathlib import Path 
import os
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.api.schemas import HealthResponse, PipelineRunResponse
from src.api.dependencies import get_deps
from src.api.services import execute_pipeline_run, get_memory_snapshot, _RUN_STORE
from src.core.metadata import resolve_problem_type
import logging
logger = logging.getLogger(__name__)


router = APIRouter()

@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def health():
    cfg, deps = get_deps()

    db_status = "unknown"
    try:
        deps["structured"].conn.cursor()
        db_status = "connected"
    except Exception:
        db_status = "unavailable"
        logger.warning("Health check: database unreachable")

    llm_status = "real" if deps["llm_client"]._real_llm_available else "mock"
    groq_configured = bool(os.environ.get("GROQ_API_KEY"))

    memory_initialized = False
    try:
        memory_initialized = deps["structured"] is not None and deps["semantic"] is not None
    except Exception:
        logger.warning("Health check: memory subsystem not initialized")

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version=os.environ.get("APP_VERSION", "unknown"),
        database=db_status,
        llm=llm_status,
        groq_configured=groq_configured,
        memory_initialized=memory_initialized,
    )

@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline_endpoint(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    target_column: str | None = Form(None),
    problem_type: str | None = Form(None),
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

    try:
        resolved_problem_type = resolve_problem_type(problem_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg, deps = get_deps()

    try:
        result = execute_pipeline_run(
            train_df, test_df, train_file.filename, target_column, cfg, deps,
            problem_type=resolved_problem_type
        )
    except Exception as exc:
        logger.exception(f"Pipeline execution failed for train_file={train_file.filename}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")

    logger.info(f"Run {result['run_id']} completed — best_model={result['best_model']}, "
                f"execution_time={result['execution_time']:.2f}s")
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

@router.get("/predictions/{run_id}")
def get_predictions(run_id: str):
    run = _RUN_STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    pred_path = run.get("prediction_path")
    if not pred_path or not os.path.isfile(pred_path):
        raise HTTPException(status_code=404, detail="Prediction file not found")
    with open(pred_path, "rb") as f:
        content = f.read()
    return Response(
        content=content, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prediction_{run_id}.csv"'},
    )

@router.get("/memory")
def get_memory(limit: int | None = None):
    cfg, deps = get_deps()
    try:
        return JSONResponse(content=get_memory_snapshot(deps, limit=limit))
    except Exception as exc:
        logger.exception("Memory read failed")
        raise HTTPException(status_code=500, detail=f"Memory read failed: {exc}")