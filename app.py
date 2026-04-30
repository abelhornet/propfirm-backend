from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SIMULATIONS = 1000
MAX_TRADES = 300


# ==============================
# REQUEST MODEL
# ==============================

class FreeRequest(BaseModel):
    winrate: float = Field(..., gt=0, lt=1.0)
    rr: float = Field(..., gt=0)
    risk: float = Field(..., gt=0, lt=3)

    rr_std: float = 0.3

    initial_balance: float = 10000
    target_pct: float = 8

    max_dd_pct: Optional[float] = None
    daily_dd_pct: Optional[float] = None
    min_days: Optional[int] = None
    max_days: Optional[int] = None


# ==============================
# NORMALIZATION
# ==============================

def normalize_config(cfg):
    return {
        "initial_balance": cfg["initial_balance"],
        "target_pct": (cfg["target_pct"] / 100) * 1.1,
        "max_dd_pct": cfg["max_dd_pct"] / 100 if cfg.get("max_dd_pct") else None,
        "daily_dd_pct": cfg["daily_dd_pct"] / 100 if cfg.get("daily_dd_pct") else None,
        "min_days": cfg.get("min_days"),
        "max_days": cfg.get("max_days") if cfg.get("max_days") else 999,
    }


def normalize_winrate(w):
    return w / 100 if w > 1 else w


# ==============================
# CORE SIMULATION
# ==============================

def run_simulation(winrate, rr, risk, rr_std, cfg):
    balance = cfg["initial_balance"]
    peak = balance
    target = balance * (1 + cfg["target_pct"])

    days = 0
    trades_today = 0
    day_start = balance

    dd_track = []

    for _ in range(MAX_TRADES):

        r = rr if np.random.rand() < winrate else -1
        r += np.random.normal(0, rr_std)

        risk_factor = (risk / 100)
        balance *= (1 + risk_factor * r)

        # fricción
        balance *= 0.9995

        # penalización riesgo alto
        if risk > 1.5:
            balance *= (1 - (risk - 1.5) * 0.004)

        if balance <= 0:
            return "fail", days, dd_track, balance

        peak = max(peak, balance)
        dd = (balance - peak) / peak
        dd_track.append(dd)

        if cfg["max_dd_pct"] and dd <= -cfg["max_dd_pct"]:
            return "fail", days, dd_track, balance

        if cfg["daily_dd_pct"]:
            if (balance - day_start) / day_start <= -cfg["daily_dd_pct"]:
                return "fail", days, dd_track, balance

        trades_today += 1

        if trades_today >= 3:
            trades_today = 0
            days += 1
            day_start = balance

        if balance >= target:
            if not cfg["min_days"] or days >= cfg["min_days"]:
                return "pass", days, dd_track, balance

        if cfg["max_days"] and days >= cfg["max_days"]:
            return "timeout", days, dd_track, balance

    return "timeout", days, dd_track, balance


# ==============================
# MONTE CARLO
# ==============================

def monte_carlo(winrate, rr, risk, rr_std, cfg):
    outcomes = []
    days_list = []
    dd_all = []
    end_balances = []

    target = cfg["initial_balance"] * (1 + cfg["target_pct"])

    for _ in range(SIMULATIONS):
        o, d, dd, final_balance = run_simulation(winrate, rr, risk, rr_std, cfg)

        outcomes.append(o)
        days_list.append(d)
        dd_all.append(min(dd) if dd else 0)
        end_balances.append(final_balance)

    s = pd.Series(outcomes)

    return {
        "pass_rate": float((s == "pass").mean()),
        "avg_days": float(np.mean(days_list)),
        "max_dd_p95": float(np.percentile(dd_all, 95)),
        "profitable_rate": float(
            sum(1 for b in end_balances if b > cfg["initial_balance"]) / SIMULATIONS
        ),
        "expected_return": float(
            np.mean(end_balances) / cfg["initial_balance"] - 1
        ),
        "avg_progress": float(np.mean(end_balances) / target),
        "dd_fail_rate": float((s == "fail").mean()),
    }


# ==============================
# FREE
# ==============================

@app.post("/simulate/free")
def simulate_free(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    cfg = normalize_config(req.dict())

    stats = monte_carlo(winrate, req.rr, req.risk, req.rr_std, cfg)

    edge = (winrate * req.rr) - (1 - winrate)
    kelly = edge / req.rr if req.rr != 0 else 0

    risk_amount = req.initial_balance * (req.risk / 100)

    return {
        "profiles": [
            {
                "name": "standard",
                "pass_rate": round(stats["pass_rate"] * 100, 2),
                "risk": req.risk,
                "risk_amount": round(risk_amount, 2),
                "edge": round(edge * 100, 2),
                "kelly": round(kelly, 3),
                "stats": stats,
            }
        ]
    }


# ==============================
# OPTIMIZE (PRO)
# ==============================

@app.post("/optimize")
def optimize(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    cfg = normalize_config(req.dict())

    risk_grid = np.linspace(0.25, 2.0, 10)

    raw_results = []

    for r in risk_grid:
        stats = monte_carlo(winrate, req.rr, r, req.rr_std, cfg)

        raw_results.append({
            "risk": r,
            "pass_rate": stats["pass_rate"],
            "avg_days": stats["avg_days"],
            "max_dd": stats["max_dd_p95"],
            "dd_fail_rate": stats["dd_fail_rate"],
        })

    def score(x):
        return (
            x["pass_rate"] * 0.6
            - x["dd_fail_rate"] * 0.9
            - (x["avg_days"] / 60)
            - abs(x["max_dd"]) * 0.4
        )

    best = max(raw_results, key=score)
    optimal_risk = best["risk"]

    profile_risks = {
        "low": max(0.25, optimal_risk * 0.6),
        "mid": optimal_risk,
        "high": min(2.5, optimal_risk * 1.4),
    }

    final_profiles = []

    for name, r in profile_risks.items():
        stats = monte_carlo(winrate, req.rr, r, req.rr_std, cfg)

        final_profiles.append({
            "name": name,
            "risk": round(r, 2),
            "risk_amount": round(req.initial_balance * (r / 100), 2),
            "pass_rate": round(stats["pass_rate"] * 100, 2),
        })

    edge = (winrate * req.rr) - (1 - winrate)
    kelly = edge / req.rr if req.rr != 0 else 0

    return {
        "optimal": {
            "risk": round(optimal_risk, 2),
            "risk_amount": round(req.initial_balance * (optimal_risk / 100), 2),
            "pass_rate": round(best["pass_rate"] * 100, 2),
            "avg_days": round(best["avg_days"], 1),
            "max_dd": round(best["max_dd"] * 100, 2),
            "edge": round(edge * 100, 2),
            "kelly": round(kelly, 3),
        },
        "all_results": final_profiles
    }
