from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.domain import SimulationRequest
from app.llm.client import LLMError
from app.simulation.engine import run_linear_forecast


router = APIRouter(prefix="/api/llm", tags=["forecasts"])


def _forecast_failure(exc: Exception) -> HTTPException:
    while isinstance(exc, ExceptionGroup):
        exc = exc.exceptions[0]
    return HTTPException(status_code=502 if isinstance(exc, LLMError) else 422, detail=str(exc))


@router.post("/simulate")
async def simulate(request: SimulationRequest) -> dict:
    try:
        result = await run_linear_forecast(request, 1)
        return result.model_dump()
    except Exception as exc:
        raise _forecast_failure(exc) from exc


@router.post("/multi-step-simulate")
async def multi_step_simulate(request: SimulationRequest) -> dict:
    try:
        result = await run_linear_forecast(request, request.steps)
        return result.model_dump()
    except Exception as exc:
        raise _forecast_failure(exc) from exc


@router.post("/multi-branch-simulate", deprecated=True)
async def multi_branch_simulate() -> dict:
    raise HTTPException(
        status_code=410,
        detail="Branch forecasting was replaced by /multi-step-simulate with prediction_repeats and event_threshold_percent.",
    )
