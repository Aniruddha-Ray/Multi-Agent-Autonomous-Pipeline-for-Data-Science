import requests
import sys
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from streamlit_app.utils.constants import API_BASE_URL, ENDPOINTS


def health() -> dict:
    resp = requests.get(API_BASE_URL + ENDPOINTS["health"], timeout=5)
    resp.raise_for_status()
    return resp.json()


def run_pipeline(train_file_bytes: bytes, train_filename: str,
                  test_file_bytes: bytes, test_filename: str,
                  target_column: str | None = None,
                  problem_type: str | None = None) -> dict:
    files = {
        "train_file": (train_filename, train_file_bytes, "text/csv"),
        "test_file": (test_filename, test_file_bytes, "text/csv"),
    }
    data = {}
    if target_column:
        data["target_column"] = target_column
    if problem_type and problem_type != "Auto-detect":
        data["problem_type"] = problem_type.lower()
    resp = requests.post(API_BASE_URL + ENDPOINTS["run_pipeline"], files=files, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json()


def get_run(run_id: str) -> dict:
    resp = requests.get(API_BASE_URL + ENDPOINTS["get_run"].format(run_id=run_id), timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_report(run_id: str) -> str:
    resp = requests.get(API_BASE_URL + ENDPOINTS["get_report"].format(run_id=run_id), timeout=10)
    resp.raise_for_status()
    return resp.text


def get_memory(limit: int | None = None) -> list:
    params = {"limit": limit} if limit else {}
    resp = requests.get(API_BASE_URL + ENDPOINTS["get_memory"], params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()