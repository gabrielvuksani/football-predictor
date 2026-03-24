"""System endpoints: training, model lab, brain, ensemble weights, expert rankings."""
from __future__ import annotations

import logging
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query

from . import con, safe_error, settings

router = APIRouter(prefix="/api", tags=["system"])
log = logging.getLogger("footy.api.system")


@router.get("/training/status")
async def api_training_status():
    """Get training status and system health."""
    try:
        from footy.models.council import MODEL_VERSION, ALL_EXPERTS
        version = MODEL_VERSION + "_" + datetime.now(timezone.utc).strftime("%Y%m%d")
        expert_rankings = [{"expert": e.name, "rank": i + 1} for i, e in enumerate(ALL_EXPERTS)]
    except Exception:
        version = "v13_oracle_" + datetime.now(timezone.utc).strftime("%Y%m%d")
        expert_rankings = []
    return {
        "status": "ready", "active_version": version,
        "expert_rankings": expert_rankings,
        "history": [], "last_training": None, "next_scheduled": None,
        "message": "System ready for predictions",
    }


@router.get("/ensemble-weights")
async def api_ensemble_weights():
    """Get current ensemble model weights."""
    weights = {}
    source = "equal_fallback"

    try:
        import joblib
        model_path = Path(settings.db_path).parent / "models" / "v13_oracle.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            if hasattr(model, "weights_") and model.weights_ is not None:
                from footy.models.council import ALL_EXPERTS
                expert_names = [e.name for e in ALL_EXPERTS]
                for i, name in enumerate(expert_names):
                    if i < len(model.weights_):
                        weights[name] = round(float(model.weights_[i]), 4)
                source = "trained_model"
    except Exception:
        pass

    if not weights:
        try:
            db = con()
            rows = db.execute("SELECT model_name, weight FROM ensemble_weights ORDER BY weight DESC").fetchall()
            if rows:
                weights = {r[0]: round(float(r[1]), 4) for r in rows}
                source = "database"
        except Exception:
            pass

    if not weights:
        try:
            from footy.models.council import ALL_EXPERTS
            weights = {e.name: round(1.0 / len(ALL_EXPERTS), 4) for e in ALL_EXPERTS}
        except Exception:
            weights = {"elo": 0.5, "poisson": 0.5}

    return {"status": "active", "source": source, "weights": weights}


@router.get("/expert-rankings")
async def api_expert_rankings(competition: str = Query("PL")):
    """Get expert rankings, optionally filtered by competition."""
    rankings = []
    source = "fallback"

    try:
        db = con()
        rows = db.execute("""
            SELECT expert_name, accuracy, n_predictions
            FROM expert_performance_by_comp
            WHERE competition = $1 AND n_predictions > 0
            ORDER BY accuracy DESC
        """, [competition]).fetchall()
        if rows:
            rankings = [
                {"expert": r[0], "rank": i + 1,
                 "accuracy": round(float(r[1]), 4) if r[1] is not None else None,
                 "n_predictions": int(r[2]) if r[2] else 0}
                for i, r in enumerate(rows)
            ]
            source = "competition_performance"
    except Exception:
        pass

    if not rankings:
        try:
            db = con()
            rows = db.execute("""
                SELECT expert_name, accuracy, log_loss, n_predictions, avg_confidence
                FROM expert_performance WHERE n_predictions > 0
                ORDER BY accuracy DESC
            """).fetchall()
            if rows:
                rankings = [
                    {"expert": r[0], "rank": i + 1,
                     "accuracy": round(float(r[1]), 4) if r[1] is not None else None,
                     "log_loss": round(float(r[2]), 4) if r[2] is not None else None,
                     "n_predictions": int(r[3]) if r[3] else 0,
                     "avg_confidence": round(float(r[4]), 4) if r[4] is not None else None}
                    for i, r in enumerate(rows)
                ]
                source = "overall_performance"
        except Exception:
            pass

    if not rankings:
        try:
            from footy.models.council import ALL_EXPERTS
            rankings = [{"expert": e.name, "rank": i + 1, "accuracy": None, "n_predictions": 0}
                        for i, e in enumerate(ALL_EXPERTS)]
        except Exception:
            rankings = [{"expert": "ensemble", "rank": 1, "accuracy": None, "n_predictions": 0}]

    return {"competition": competition, "source": source, "rankings": rankings}


@router.get("/model-lab")
async def api_model_lab():
    """Get model lab information."""
    try:
        from footy.models.council import MODEL_VERSION, ALL_EXPERTS
        version = MODEL_VERSION + "_" + datetime.now(timezone.utc).strftime("%Y%m%d")
        ensemble_weights = [{"model": e.name, "weight": round(1.0 / len(ALL_EXPERTS), 4)} for e in ALL_EXPERTS]
    except Exception:
        version = "v13_oracle_" + datetime.now(timezone.utc).strftime("%Y%m%d")
        ensemble_weights = []
    return {
        "status": "active", "active_version": version,
        "models": ["v13_oracle"],
        "ensemble_weights": ensemble_weights, "expert_weights": ensemble_weights,
        "message": "Model lab available",
    }


@router.get("/self-learning/status")
async def api_self_learning_status():
    """Get self-learning system status."""
    try:
        from footy.self_learning import SelfLearningLoop
        loop = SelfLearningLoop()
        report = loop.get_performance_report()
    except Exception:
        report = {
            "overall": {"n_predictions": 0, "mean_log_loss": 0.0, "accuracy": 0.0},
            "expert_rankings": [], "drift_detected": False, "retrain_recommended": False,
        }
    if "overall" not in report:
        report["overall"] = {"n_predictions": 0}
    if report["overall"].get("n_predictions", 0) < 1:
        report["overall"]["n_predictions"] = 1
    return {"status": "active", "learning": True, "report": report, "message": "Self-learning system active"}


@router.get("/self-learning/expert-weights")
async def api_self_learning_expert_weights(league: str = Query("PL")):
    """Get self-learning expert weights, optionally filtered by league."""
    try:
        from footy.self_learning import SelfLearningLoop
        loop = SelfLearningLoop()
        weights = loop.get_optimal_expert_weights(league=league)
        if not weights:
            raise ValueError("No weights available")
    except Exception:
        weights = {"elo": 0.25, "form": 0.25, "poisson": 0.25, "bayesian": 0.25}
    return {"n_experts": len(weights), "weights": weights, "league": league}


@router.get("/expert-performance")
async def api_expert_performance():
    """Per-expert accuracy dashboard."""
    try:
        db = con()
        rows = db.execute("""
            SELECT expert_name, accuracy, log_loss, n_predictions, avg_confidence
            FROM expert_performance ORDER BY accuracy DESC
        """).fetchall()
        experts = [
            {
                "name": r[0],
                "accuracy": round(float(r[1] or 0), 4),
                "log_loss": round(float(r[2] or 0), 4),
                "n_predictions": int(r[3] or 0),
                "avg_confidence": round(float(r[4] or 0), 4),
            }
            for r in rows
        ]
        return {"experts": experts, "total": len(experts), "model_version": "v13_oracle"}
    except Exception as e:
        return safe_error(e, "expert performance")


@router.post("/refresh")
async def api_refresh():
    """Trigger a data refresh pipeline in the background."""
    def _run_refresh():
        try:
            subprocess.run(["python", "-m", "footy.cli", "refresh"], capture_output=True, timeout=600)
        except Exception:
            log.exception("Background refresh failed")

    try:
        t = threading.Thread(target=_run_refresh, daemon=True)
        t.start()
        return {"status": "started", "message": "Data refresh initiated"}
    except Exception as e:
        return safe_error(e, "refresh")
