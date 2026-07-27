from fastapi import FastAPI
import sys
import os
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.api.routes import router

app = FastAPI(title="Adaptive Multi-Agent Data Science Pipeline API")
app.include_router(router)