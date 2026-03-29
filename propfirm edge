from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd

app = FastAPI()

# ==============================
# CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SIMULATIONS = 500


# ==============================
# MODEL
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
# HELPERS
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
# CORE
# ==============================

def run_simulation(returns, cfg):
    balance = cfg["initial_balance"]
    peak = balance
    target = balance * (1 + cfg["target_pct"])

    days = 0
    trades_today = 0
    day_start = balance

    dd_track = []

    for _ in range(500):

        r = np.random.choice(returns)
        balance *= (1 + r)

        peak = max(peak, balance)

        dd = (balance - peak) / peak
        dd_track.append(dd)

        # MAX DD
        if cfg["max_dd_pct"]:
            if dd <= -cfg["max_dd_pct"]:
                return "fail", days, dd_track

        # DAILY DD
        if cfg["daily_dd_pct"]:
            if (balance - day_start) / day_start <= -cfg["daily_dd_pct"]:
                return "fail", days, dd_track

        trades_today += 1

        if trades_today >= 3:
            trades_today = 0
            days += 1
            day_start = balance

        # TARGET
        if balance >= target:
            if not cfg["min_days"] or days >= cfg["min_days"]:
                return "pass", days, dd_track

        # MAX DAYS
        if cfg["max_days"] and days >= cfg["max_days"]:
            return "timeout", days, dd_track

    return "timeout", days, dd_track


def monte_carlo(returns, cfg):
    outcomes = []
    days_list = []
    dd_all = []

    for _ in range(SIMULATIONS):
        o, d, dd = run_simulation(returns, cfg)
        outcomes.append(o)
        days_list.append(d)
        dd_all.append(min(dd))

    s = pd.Series(outcomes)

    return {
        "pass_rate": float((s == "pass").mean()),
        "avg_days": float(np.mean([d for o, d in zip(outcomes, days_list) if o == "pass"]) if "pass" in s.values else 0),
        "max_dd_p95": float(np.percentile(dd_all, 95))
    }


def generate_returns(winrate, rr):
    return np.array([
        np.random.normal(rr * 0.01, 0.01) if np.random.rand() < winrate else np.random.normal(-0.01, 0.005)
        for _ in range(1000)
    ])


def generate_curve(returns, balance):
    eq = [balance]
    for r in returns[:200]:
        balance *= (1 + r)
        eq.append(balance)
    return eq


# ==============================
# AUTO RISK LOGIC
# ==============================

def score(r, cfg):
    return (
        r["pass_rate"] * 0.6
        - (abs(r["max_dd_p95"]) / (cfg["max_dd_pct"] or 1)) * 0.25
        - (r["avg_days"] / (cfg["max_days"] or 30)) * 0.15
    )


def build_profiles(results, cfg):
    for r in results:
        r["score"] = score(r, cfg)

    safe = min(results, key=lambda x: abs(x["max_dd_p95"]))
    balanced = max(results, key=lambda x: x["score"])
    aggressive = max(results, key=lambda x: x["pass_rate"])

    return {
        "safe": safe,
        "balanced": balanced,
        "aggressive": aggressive
    }


# ==============================
# FREE (SIN CAMBIOS)
# ==============================

@app.post("/simulate/free")
def simulate_free(req: FreeRequest):

    winrate = normalize_winrate(req.winrate)
    cfg = normalize_config(req.dict())

    returns = generate_returns(winrate, req.rr) * (req.risk / 100)

    stats = monte_carlo(returns, cfg)

    return {
        "profiles": [
            {
                "name": "quick",
                "pass_rate": round(stats["pass_rate"] * 100, 2),
                "risk": req.risk,
                "mc_curves": {
                    "p50": generate_curve(returns, req.initial_balance)
                }
            }
        ],
        "optimal_risk": req.risk
    }


# ==============================
# OPTIMIZER (PRO)
# ==============================

@app.post("/optimize")
async def optimize(
    file: UploadFile = File(...),
    base_risk_pct: float = 1.0,
    initial_balance: float = 10000,
    target_pct: float = 8,
    max_dd_pct: Optional[float] = None,
    daily_dd_pct: Optional[float] = None,
    min_days: Optional[int] = None,
    max_days: Optional[int] = None
):

    df = pd.read_csv(file.file)
    returns = df["return"].values

    cfg = normalize_config({
        "initial_balance": initial_balance,
        "target_pct": target_pct,
        "max_dd_pct": max_dd_pct,
        "daily_dd_pct": daily_dd_pct,
        "min_days": min_days,
        "max_days": max_days
    })

    risks = np.linspace(0.2, 5, 15)
    results = []

    for r in risks:
        scaled = returns * (r / 100)
        stats = monte_carlo(scaled, cfg)

        results.append({
            "risk": float(r),
            "pass_rate": stats["pass_rate"],
            "avg_days": stats["avg_days"],
            "max_dd_p95": stats["max_dd_p95"]
        })

    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res["pass_rate"].idxmax()]

    return {
        "profiles": [
            {
                "name": "optimized",
                "risk": round(best["risk"], 2),
                "pass_rate": round(best["pass_rate"] * 100, 2),
                "mc_curves": {
                    "p50": generate_curve(returns * (best["risk"] / 100), initial_balance)
                }
            }
        ],
        "optimal_risk": round(best["risk"], 2),
        "heatmap": results
    }


# ==============================
# AUTO RISK (NUEVO PRO)
# ==============================

@app.post("/auto-risk")
async def auto_risk(
    file: UploadFile = File(...),
    initial_balance: float = 10000,
    target_pct: float = 8,
    max_dd_pct: Optional[float] = None,
    max_days: Optional[int] = None
):

    df = pd.read_csv(file.file)
    returns = df["return"].values

    cfg = normalize_config({
        "initial_balance": initial_balance,
        "target_pct": target_pct,
        "max_dd_pct": max_dd_pct,
        "max_days": max_days
    })

    risks = np.linspace(0.2, 5, 15)
    results = []

    for r in risks:
        scaled = returns * (r / 100)
        stats = monte_carlo(scaled, cfg)

        results.append({
            "risk": float(r),
            "pass_rate": stats["pass_rate"],
            "avg_days": stats["avg_days"],
            "max_dd_p95": stats["max_dd_p95"]
        })

    profiles = build_profiles(results, cfg)

    return {
        "profiles": profiles,
        "heatmap": results
    }

