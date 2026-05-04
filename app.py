from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
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

SIMULATIONS = 300
MAX_TRADES = 300
TRADES_PER_DAY = 3


# ==============================
# MODELS
# ==============================

class OptimizeRequest(BaseModel):
    returns: List[float]

    initial_balance: float = 10000
    target_pct: float = 8

    max_dd_pct: Optional[float] = None
    daily_dd_pct: Optional[float] = None
    min_days: Optional[int] = None
    max_days: Optional[int] = None


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
        "target_pct": cfg["target_pct"] / 100,
        "max_dd_pct": cfg["max_dd_pct"] / 100 if cfg.get("max_dd_pct") else None,
        "daily_dd_pct": cfg["daily_dd_pct"] / 100 if cfg.get("daily_dd_pct") else None,
        "min_days": cfg.get("min_days"),
        "max_days": cfg.get("max_days") if cfg.get("max_days") else 999,
    }


def normalize_winrate(w):
    return w / 100 if w > 1 else w


# ==============================
# CONFIDENCE SCORE
# ==============================

def compute_confidence(returns):
    n = len(returns)

    if n == 0:
        return 0.0

    # 📊 sample size component (log scaling)
    sample_score = min(1.0, np.log10(n + 1) / 2)

    # 📉 variance penalty
    std = np.std(returns)
    variance_score = 1 / (1 + std)

    # 📈 signal quality (edge strength)
    mean = np.mean(returns)
    signal_score = abs(mean) / (std + 1e-6)

    signal_score = min(signal_score / 2, 1.0)

    # 🔥 combinación
    confidence = (
        0.5 * sample_score +
        0.3 * variance_score +
        0.2 * signal_score
    )

    return float(round(confidence, 3))


# ==============================
# CORE ENGINE
# ==============================

def run_simulation(returns, risk, cfg):
    balance = cfg["initial_balance"]
    peak = balance
    target = balance * (1 + cfg["target_pct"])

    days = 0
    trades_today = 0
    day_start = balance

    dd_track = []

    for _ in range(MAX_TRADES):

        r = np.random.choice(returns)

        balance *= (1 + (risk / 100) * r)
        balance *= 0.9998

        peak = max(peak, balance)
        dd = (balance - peak) / peak
        dd_track.append(dd)

        if balance <= 0:
            return "fail", days, dd_track, balance

        if cfg["max_dd_pct"] and dd <= -cfg["max_dd_pct"]:
            return "fail", days, dd_track, balance

        if cfg["daily_dd_pct"]:
            if (balance - day_start) / day_start <= -cfg["daily_dd_pct"]:
                return "fail", days, dd_track, balance

        trades_today += 1

        if trades_today >= TRADES_PER_DAY:
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

def monte_carlo(returns, risk, cfg):
    outcomes = []
    days_list = []
    dd_all = []

    for _ in range(SIMULATIONS):
        o, d, dd, _ = run_simulation(returns, risk, cfg)

        outcomes.append(o)
        days_list.append(d)
        dd_all.append(min(dd) if dd else 0)

    s = pd.Series(outcomes)

    return {
        "pass_rate": float((s == "pass").mean()),
        "avg_days": float(np.mean(days_list)),
        "max_dd_p95": float(np.percentile(dd_all, 95)),
        "dd_fail_rate": float((s == "fail").mean()),
    }


# ==============================
# EQUITY CURVE
# ==============================

def generate_equity_curve(returns, risk, cfg):
    balance = cfg["initial_balance"]
    curve = [balance]

    for _ in range(150):
        r = np.random.choice(returns)
        balance *= (1 + (risk / 100) * r)
        curve.append(balance)

    return curve


# ==============================
# FREE
# ==============================

@app.post("/simulate/free")
def simulate_free(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    cfg = normalize_config(req.dict())

    returns = []

    for _ in range(1000):
        r = req.rr if np.random.rand() < winrate else -1
        r += np.random.normal(0, req.rr_std)
        returns.append(r)

    returns = np.array(returns)

    stats = monte_carlo(returns, req.risk, cfg)

    edge = (winrate * req.rr) - (1 - winrate)
    kelly = edge / req.rr if req.rr != 0 else 0

    return {
        "profiles": [
            {
                "name": "standard",
                "pass_rate": round(stats["pass_rate"] * 100, 2),
                "risk": req.risk,
                "risk_amount": round(req.initial_balance * (req.risk / 100), 2),
                "edge": round(edge * 100, 2),
                "kelly": round(kelly, 3),
            }
        ]
    }


# ==============================
# OPTIMIZE
# ==============================

@app.post("/optimize")
def optimize(req: OptimizeRequest):

    cfg = normalize_config(req.dict())

    returns = np.array(req.returns)

    returns = returns[np.isfinite(returns)]
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {"error": "No valid trades"}

    low_sample = len(returns) < 20

    # 🔥 clipping solo si hay muestra suficiente
    if len(returns) >= 20:
        returns = np.clip(returns, -3, 3)

    confidence = compute_confidence(returns)

    print("========== BACKEND DEBUG ==========")
    print("Trades:", len(returns))
    print("Mean:", np.mean(returns))
    print("Confidence:", confidence)
    print("===================================")

    risk_grid = np.linspace(0.25, 2.0, 10)
    results = []

    for r in risk_grid:
        stats = monte_carlo(returns, r, cfg)

        results.append({
            "risk": r,
            "pass_rate": stats["pass_rate"],
            "avg_days": stats["avg_days"],
            "max_dd": stats["max_dd_p95"],
            "dd_fail_rate": stats["dd_fail_rate"],
        })

    def score(x):
        return (
            x["pass_rate"] * 0.7
            - x["dd_fail_rate"] * 0.9
            - (x["avg_days"] / 60)
            - abs(x["max_dd"]) * 0.4
        )

    best = max(results, key=score)
    optimal_risk = best["risk"]

    profiles = {
        "low": max(0.25, optimal_risk * 0.6),
        "mid": optimal_risk,
        "high": min(2.5, optimal_risk * 1.4),
    }

    final_profiles = []

    for name, r in profiles.items():
        stats = monte_carlo(returns, r, cfg)

        final_profiles.append({
            "name": name,
            "risk": round(r, 2),
            "risk_amount": round(cfg["initial_balance"] * (r / 100), 2),
            "pass_rate": round(stats["pass_rate"] * 100, 2),
        })

    equity_curve = generate_equity_curve(returns, optimal_risk, cfg)

    return {
        "optimal": {
            "risk": round(optimal_risk, 2),
            "risk_amount": round(cfg["initial_balance"] * (optimal_risk / 100), 2),
            "pass_rate": round(best["pass_rate"] * 100, 2),
            "avg_days": round(best["avg_days"], 1),
            "max_dd": round(best["max_dd"] * 100, 2),
        },
        "all_results": final_profiles,
        "equity_curve": equity_curve,

        # 🔥 NUEVO
        "confidence": confidence,
        "low_sample": low_sample,
        "trades_count": int(len(returns)),
    }
