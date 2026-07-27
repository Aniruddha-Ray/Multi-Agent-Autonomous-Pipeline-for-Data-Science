from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str

class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    best_model: str
    metrics: dict[str, float]
    report_path: str
    execution_time: float