from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backtest import get_holdout_window, run_strategy

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/spy-backtest")
async def spy_backtest(req: Request):
    """
    Body: {"threshold":0.55, "start": "2019-01-01", "end": "2025-10-01"}
    Returns: {"equity":[...], "total_ret": %, "cagr": %}
    """
    payload = await req.json()
    try:
        threshold = float(payload.get("threshold", 0.55))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid threshold: {exc}") from exc

    start = payload.get("start")
    end = payload.get("end")

    try:
        result = run_strategy(
            threshold=threshold,
            start=start,
            end=end,
            allow_remote_download=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch for API entrypoint
        raise HTTPException(status_code=500, detail="Backtest failed.") from exc

    return result


@app.get("/api/spy-window")
def spy_window():
    try:
        start, end = get_holdout_window()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Holdout window unavailable. Seed the price cache before serving requests.",
        ) from exc
    return {
        "ticker": "SPY",
        "start": start,
        "end": end,
    }
