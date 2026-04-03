from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# 🔥 AUMENTADO (ANTES 50)
SIMULATIONS = 1000
MAX_TRADES = 1000


# ==============================
# REQUEST MODEL
# ==============================

class FreeRequest(BaseModel):
    winrate: float
    rr: float
    risk: float

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
        "target_pct": cfg["target_pct"] / 100,
        "max_dd_pct": cfg["max_dd_pct"] / 100 if cfg.get("max_dd_pct") else None,
        "daily_dd_pct": cfg["daily_dd_pct"] / 100 if cfg.get("daily_dd_pct") else None,
        "min_days": cfg.get("min_days"),
        "max_days": cfg.get("max_days"),
    }


def normalize_winrate(w):
    return w / 100 if w > 1 else w


# ==============================
# CORE SIMULATION (FIXED)
# ==============================

def run_simulation(winrate, rr, risk, cfg):
    balance = cfg["initial_balance"]
    peak = balance
    target = balance * (1 + cfg["target_pct"])

    days = 0
    trades_today = 0
    day_start = balance

    dd_track = []

    for _ in range(MAX_TRADES):

        if np.random.rand() < winrate:
            r = rr
        else:
            r = -1

        # 🔥 ruido realista
        noise = np.random.normal(0, 0.15)
        r = r + noise

        # aplicar retorno
        balance *= (1 + (risk / 100) * r)

        if balance <= 0:
            return "fail", days, dd_track, balance

        peak = max(peak, balance)

        dd = (balance - peak) / peak
        dd_track.append(dd)

        # MAX DD
        if cfg["max_dd_pct"] and dd <= -cfg["max_dd_pct"]:
            return "fail", days, dd_track, balance

        # DAILY DD
        if cfg["daily_dd_pct"]:
            if (balance - day_start) / day_start <= -cfg["daily_dd_pct"]:
                return "fail", days, dd_track, balance

        trades_today += 1

        if trades_today >= 3:
            trades_today = 0
            days += 1
            day_start = balance

        # PASS
        if balance >= target:
            if not cfg["min_days"] or days >= cfg["min_days"]:
                return "pass", days, dd_track, balance

        # TIMEOUT
        if cfg["max_days"] and days >= cfg["max_days"]:
            return "timeout", days, dd_track, balance

    return "timeout", days, dd_track, balance


# ==============================
# MONTE CARLO (FIXED)
# ==============================

def monte_carlo(winrate, rr, risk, cfg):
    outcomes = []
    days_list = []
    dd_all = []
    end_balances = []

    target = cfg["initial_balance"] * (1 + cfg["target_pct"])

    for _ in range(SIMULATIONS):
        o, d, dd, final_balance = run_simulation(winrate, rr, risk, cfg)

        outcomes.append(o)
        days_list.append(d)
        dd_all.append(min(dd) if dd else 0)
        end_balances.append(final_balance)

    s = pd.Series(outcomes)

    pass_rate = float((s == "pass").mean())
    profitable_rate = float(
        sum(1 for b in end_balances if b > cfg["initial_balance"]) / SIMULATIONS
    )
    dd_fail_rate = float((s == "fail").mean())
    avg_days = float(np.mean(days_list))

    return {
        "pass_rate": pass_rate,
        "avg_days": avg_days,
        "max_dd_p95": float(np.percentile(dd_all, 95)),
        "profitable_rate": profitable_rate,
        "expected_return": float(
            np.mean(end_balances) / cfg["initial_balance"] - 1
        ),
        "avg_progress": float(np.mean(end_balances) / target),
        "dd_fail_rate": dd_fail_rate,
    }


# ==============================
# INTELLIGENCE
# ==============================

def generate_message(stats):

    if stats["pass_rate"] > 0.6:
        return "Favorable conditions to pass challenge"

    if stats["pass_rate"] < 0.1 and stats["profitable_rate"] > 0.6:
        return "Profitable strategy but unlikely to pass challenge"

    if stats["profitable_rate"] < 0.5:
        return "Strategy lacks edge"

    if stats["avg_progress"] > 0.75:
        return "Good progress but constraints are limiting performance"

    return "Moderate performance under current constraints"


# ==============================
# FREE ENDPOINT
# ==============================

@app.post("/simulate/free")
def simulate_free(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    cfg = normalize_config(req.dict())

    stats = monte_carlo(winrate, req.rr, req.risk, cfg)
    message = generate_message(stats)

    edge = (winrate * req.rr) - (1 - winrate)

    # 🔥 curva simple para UI
    balance = req.initial_balance
    curve = [balance]

    for _ in range(150):
        if np.random.rand() < winrate:
            r = req.rr
        else:
            r = -1

        balance *= (1 + (req.risk / 100) * r)
        curve.append(balance)

    return {
        "profiles": [
            {
                "name": "quick",
                "pass_rate": round(stats["pass_rate"] * 100, 2),
                "risk": req.risk,
                "message": message,
                "edge": round(edge * 100, 2),
                "stats": stats,
                "mc_curves": {
                    "p50": curve
                }
            }
        ],
        "optimal_risk": req.risk
    }


# ==============================
# OPTIMIZE ENDPOINT 🔥
# ==============================

@app.post("/optimize")
def optimize(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    base_cfg = normalize_config(req.dict())

    risks = np.arange(0.25, 2.25, 0.25)

    results = []

    for r in risks:
        stats = monte_carlo(winrate, req.rr, r, base_cfg)

        results.append({
            "risk": round(r, 2),
            "pass_rate": stats["pass_rate"],
            "avg_days": stats["avg_days"],
            "max_dd": stats["max_dd_p95"],
            "dd_fail_rate": stats["dd_fail_rate"],
            "stats": stats
        })

    def score(x):
        return (
            x["pass_rate"] * 0.7
            - x["dd_fail_rate"] * 0.5
            - (x["avg_days"] / 100)
        )

    best = max(results, key=score)

    winrate = normalize_winrate(req.winrate)
    edge = (winrate * req.rr) - (1 - winrate)

    return {
        "optimal": {
            "risk": best["risk"],
            "pass_rate": round(best["pass_rate"] * 100, 2),
            "avg_days": round(best["avg_days"], 1),
            "max_dd": round(best["max_dd"] * 100, 2),
            "dd_fail_rate": round(best["dd_fail_rate"] * 100, 2),
            "edge": round(edge * 100, 2)
        },
        "all_results": [
            {
                "risk": r["risk"],
                "pass_rate": round(r["pass_rate"] * 100, 2),
                "avg_days": round(r["avg_days"], 1)
            }
            for r in results
        ]
    }
