API_BASE_URL = "http://localhost:8000"

ENDPOINTS = {
    "health": "/health",
    "run_pipeline": "/pipeline/run",
    "get_run": "/runs/{run_id}",
    "get_report": "/report/{run_id}",
    "get_predictions": "/predictions/{run_id}",
    "get_memory": "/memory",
}

THEME = {
    "primary": "#4361EE",
    "secondary": "#64748B",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "bg": "#F7F8FA",
    "card": "#FFFFFF",
    "text": "#1E293B",
    "text_muted": "#64748B",
    "border": "#E5E7EB",
    "radius": "18px",
    "font": "'Inter', sans-serif",
}

APP_VERSION = "v1.2.0"
AUTHOR_NAME = "Aniruddha"
AUTHOR_GITHUB = "https://github.com/Aniruddha-Ray"