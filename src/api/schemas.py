from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    llm: str
    groq_configured: bool
    memory_initialized: bool

class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    best_model: str
    metrics: dict[str, float]
    report_path: str
    execution_time: float
    prediction_path: str | None = None
    target_column: str
    problem_type: str
    details: dict | None = None