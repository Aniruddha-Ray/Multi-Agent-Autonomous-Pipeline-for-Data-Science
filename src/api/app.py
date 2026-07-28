import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import asyncio
import logging
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.api.dependencies import get_deps
from src.logging_config import configure_logging
from src.api.routes import router
from src.config.settings import CFG

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive Multi-Agent Data Science Pipeline API",
    docs_url="/docs" if CFG.debug else None,
    redoc_url="/redoc" if CFG.debug else None,
    openapi_url="/openapi.json" if CFG.debug else None,
)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

trusted_hosts = os.environ.get("TRUSTED_HOSTS", "*").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s: {request.method} {request.url.path}")
        return JSONResponse(status_code=504, content={"detail": "Request timed out."})
    
app.include_router(router)


@app.on_event("startup")
async def on_startup():
    logger.info("Backend starting up")

    required_env = ["POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        raise RuntimeError(f"Startup aborted — missing environment variables: {missing}")

    try:
        cfg, deps = get_deps()
    except Exception:
        logger.exception("Startup failed — could not initialize dependencies (DB/memory/LLM)")
        raise

    for path_attr in ("artifacts_dir", "uploads_dir"):
        path = getattr(cfg, path_attr)
        if not os.path.isdir(path) or not os.access(path, os.W_OK):
            logger.error(f"Startup validation failed — '{path}' ({path_attr}) is not a writable directory")
            raise RuntimeError(f"Startup aborted — '{path}' is not writable")

    logger.info("Startup validation passed — database reachable, directories writable")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs or try again."},
    )