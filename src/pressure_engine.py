"""
Pressure Engine module for SupeRR Selector Analysis System.
Reconstructs the exact match-state prior to each delivery using only information available
before the ball was delivered (zero data leakage). Evaluates configurable pressure definitions.
"""

import numpy as np
import pandas as pd
from src.config import PRESSURE_CONFIG

def compute_predelivery_state(dfs):
    """
    Computes pre-delivery context for every ball in ball_by_ball DataFrame.
    Strictly preserves causality: no ball's outcome determines its pre-delivery state.
    """
    bbb = dfs["bbb"].copy()
    matches = dfs["matches"]
    
    print("[PRESSURE ENGINE] Reconstructing pre-delivery match states...")
    
    # 1. Innings 1 totals per match (to determine Innings 2 chase target)
    inn1_summary = bbb[bbb["innings"] == 1].groupby("match_id").agg(
        inn1_total_runs=("total_ball_runs", "sum")
    ).reset_index()
    
    # 2. Merge match context
    m_meta = matches[[
        "match_id", "comp_id", "match_date", "knock_out", "win_team_id", "result"
    ]].copy()
    m_meta = m_meta.merge(inn1_summary, on="match_id", how="left")
    
    # Target formula: For standard 20-over T20 matches, target is Innings 1 Total + 1
    m_meta["target"] = m_meta["inn1_total_runs"] + 1
    
    # Merge metadata into bbb
    cols_to_merge = ["match_id", "comp_id", "match_date", "target", "knock_out", "win_team_id"]
    if "comp_id" in bbb.columns:
        bbb = bbb.drop(columns=["comp_id"])
    if "match_date" in bbb.columns:
        bbb = bbb.drop(columns=["match_date"])
    bbb = bbb.merge(m_meta[cols_to_merge], on="match_id", how="left")
    
    # 3. Cumulative state BEFORE each delivery (shift by 1 within match_id, innings)
    bbb["cum_runs_before"] = bbb.groupby(["match_id", "innings"])["total_ball_runs"].shift(1).fillna(0)
    bbb["cum_runs_before"] = bbb.groupby(["match_id", "innings"])["cum_runs_before"].cumsum()
    
    bbb["cum_wickets_before"] = bbb.groupby(["match_id", "innings"])["is_wicket"].shift(1).fillna(0).astype(int)
    bbb["cum_wickets_before"] = bbb.groupby(["match_id", "innings"])["cum_wickets_before"].cumsum()
    
    bbb["cum_legal_before"] = bbb.groupby(["match_id", "innings"])["is_legal"].shift(1).fillna(0).astype(int)
    bbb["cum_legal_before"] = bbb.groupby(["match_id", "innings"])["cum_legal_before"].cumsum()
    
    # Standard 20 overs = 120 legal balls
    bbb["balls_remaining"] = np.maximum(0, 120 - bbb["cum_legal_before"])
    bbb["wickets_in_hand"] = np.maximum(0, 10 - bbb["cum_wickets_before"])
    
    # Innings 2 Chasing Metrics
    bbb["runs_required"] = np.where(
        bbb["innings"] == 2,
        np.maximum(0, bbb["target"] - bbb["cum_runs_before"]),
        0
    )
    
    bbb["rrr"] = np.where(
        (bbb["innings"] == 2) & (bbb["balls_remaining"] > 0),
        (bbb["runs_required"] / bbb["balls_remaining"]) * 6.0,
        0.0
    )
    # Clip extreme RRR for mathematical stability
    bbb["rrr"] = np.where(
        (bbb["innings"] == 2) & (bbb["balls_remaining"] == 0),
        36.0,
        bbb["rrr"].clip(0.0, 36.0)
    )
    
    # 4. Classify Pressure Deliveries across Standard, Strict, and Broad thresholds
    for p_type, cfg in PRESSURE_CONFIG.items():
        col_name = f"is_pressure_{p_type}"
        
        # Rule A: Chasing Pressure (Innings 2)
        cond_chase_death = (
            (bbb["innings"] == 2) &
            (bbb["balls_remaining"] <= cfg["chase_death_balls"]) &
            (bbb["rrr"] >= cfg["chase_min_rrr"]) &
            (bbb["rrr"] <= bbb["wickets_in_hand"] * cfg["chase_max_rrr_mult"])
        )
        cond_chase_close = (
            (bbb["innings"] == 2) &
            (bbb["balls_remaining"] <= cfg["chase_close_balls"]) &
            (bbb["runs_required"] <= cfg["chase_close_max_runs"])
        )
        
        # Rule B: Batting Collapse / Top-Order Crisis (Innings 1 or 2)
        cond_pp_collapse = (
            (bbb["cum_wickets_before"] >= cfg["crisis_pp_wickets"]) &
            (bbb["balls_remaining"] >= cfg["crisis_pp_balls_rem"])
        )
        cond_mid_collapse = (
            (bbb["cum_wickets_before"] >= cfg["crisis_mid_wickets"]) &
            (bbb["balls_remaining"] >= cfg["crisis_mid_balls_rem"])
        )
        
        # Rule C: Setting a Target Death Overs (Innings 1)
        cond_inn1_death = (
            (bbb["innings"] == 1) &
            (
                ((bbb["balls_remaining"] <= cfg["death_inn1_balls"]) & (bbb["cum_wickets_before"] >= cfg["death_inn1_wickets"])) |
                (bbb["balls_remaining"] <= cfg["death_inn1_final_balls"])
            )
        )
        
        bbb[col_name] = (cond_chase_death | cond_chase_close | cond_pp_collapse | cond_mid_collapse | cond_inn1_death)
        
    # Default pressure column
    bbb["is_pressure"] = bbb["is_pressure_std"]
    
    std_count = bbb["is_pressure_std"].sum()
    strict_count = bbb["is_pressure_strict"].sum()
    broad_count = bbb["is_pressure_broad"].sum()
    total_balls = len(bbb)
    
    print(f"[PRESSURE ENGINE] Pressure classifications assigned:")
    print(f"  -> Standard Pressure deliveries: {std_count:,} ({std_count/total_balls:.2%})")
    print(f"  -> Strict Pressure deliveries:   {strict_count:,} ({strict_count/total_balls:.2%})")
    print(f"  -> Broad Pressure deliveries:    {broad_count:,} ({broad_count/total_balls:.2%})")
    
    dfs["bbb"] = bbb
    return dfs
