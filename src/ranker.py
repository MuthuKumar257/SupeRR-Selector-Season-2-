"""
Ranking Engine module for SupeRR Selector Analysis System.
Constructs the composite Pressure Reliability Index (PRI) for batters and bowlers,
applies qualification rules, handles tie-breaking, formats display names, and outputs
submission-ready rankings.
"""

import numpy as np
import pandas as pd
from src.config import QUALIFICATION_RULES

def rank_batters(bat_df, pop_stats, min_balls=None, min_matches=None):
    """
    Ranks batters based on the enhanced Batting Pressure Reliability Index (BPRI).
    Synthesizes context-adjusted value added, personal clutch elevation, survival capacity,
    and dot-ball avoidance with effective sample size shrinkage.
    """
    if min_balls is None:
        min_balls = QUALIFICATION_RULES["batter_min_balls"]
    if min_matches is None:
        min_matches = QUALIFICATION_RULES["batter_min_matches"]
        
    pop_api = np.sqrt(pop_stats["pop_avg_p"] * pop_stats["pop_sr_p"])
    
    # 1. Normalized Absolute Pressure Index (API) - Scaled around 50.0
    norm_api = (np.sqrt(bat_df["avg_shrunk"] * bat_df["sr_shrunk"]) / pop_api) * 50.0
    
    # 2. Context Value-Added Component (Situational execution above ML expectation)
    ctx_score = bat_df["w_sr"] * np.clip(bat_df["delta_sr_ctx"] * 0.40, -20.0, 25.0)
    
    # 3. Personal Clutch Elevation Component (Shrunk difference vs player's own baseline)
    sr_clutch = (bat_df["sr_p"] - bat_df["sr_b"]) / pop_stats["pop_sr_b"]
    avg_clutch = (bat_df["avg_p"] - bat_df["avg_b"]) / pop_stats["pop_avg_b"]
    clutch_score = bat_df["w_sr"] * np.clip(sr_clutch * 15.0 + avg_clutch * 10.0, -20.0, 20.0)
    
    # 4. Execution & Dot Ball Avoidance Factor
    exec_score = ((bat_df["bound_pct_p"] - pop_stats["pop_bound_p"]) * 0.40 - 
                  (bat_df["dot_pct_p"] - pop_stats["pop_dot_p"]) * 0.25)
    exec_shrunk = bat_df["w_sr"] * np.clip(exec_score, -10.0, 15.0)
    
    # Composite BPRI
    bat_df["bpri"] = (norm_api + ctx_score + clutch_score + exec_shrunk).round(2)
    
    # Qualification filter
    bat_df["is_qualified"] = (bat_df["balls_p"] >= min_balls) & (bat_df["matches_p"] >= min_matches)
    
    ranked = bat_df[bat_df["is_qualified"]].sort_values(
        by=["bpri", "balls_p", "sr_p"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    
    return ranked

def rank_bowlers(bowl_df, pop_stats, min_balls=None, min_matches=None):
    """
    Ranks bowlers based on the enhanced Bowling Pressure Reliability Index (BPRI_bowl).
    Synthesizes context-adjusted runs saved, personal clutch elevation, wicket-taking threat,
    and dot-ball control with effective sample size shrinkage.
    """
    if min_balls is None:
        min_balls = QUALIFICATION_RULES["bowler_min_balls"]
    if min_matches is None:
        min_matches = QUALIFICATION_RULES["bowler_min_matches"]
        
    pop_econ = pop_stats["pop_econ_p"]
    pop_wkt = pop_stats["pop_wkt_p"]
    
    # 1. Base Absolute Pressure Index (scaled around 50 pts base + wicket score)
    econ_base = 50.0 + (pop_econ - bowl_df["econ_shrunk"]) * 5.0
    wkt_base = (bowl_df["wkt_rate_shrunk"] / pop_wkt) * 20.0
    
    # 2. Context Value-Added Component (Runs saved below ML expectation given exact match state)
    ctx_score = bowl_df["w_econ"] * np.clip(bowl_df["delta_econ_ctx"] * 3.5, -15.0, 20.0)
    
    # 3. Personal Clutch Elevation Component (vs player's own non-pressure baseline)
    econ_clutch = (bowl_df["econ_b"] - bowl_df["econ_p"])
    wkt_clutch = (bowl_df["wkt_rate_p"] - bowl_df["wkt_rate_b"])
    clutch_score = bowl_df["w_econ"] * np.clip(econ_clutch * 2.5 + wkt_clutch * 80.0, -15.0, 15.0)
    
    # 4. Dot Ball Dominance and Boundary Concession Control
    control_score = ((bowl_df["dot_pct_p"] - pop_stats["pop_dot_p"]) * 0.35 - 
                     (bowl_df["bound_pct_p"] - pop_stats["pop_bound_p"]) * 0.35)
    exec_shrunk = bowl_df["w_econ"] * np.clip(control_score, -8.0, 10.0)
    
    # Composite BPRI
    bowl_df["bpri"] = (econ_base + wkt_base + ctx_score + clutch_score + exec_shrunk).round(2)
    
    # Qualification filter
    bowl_df["is_qualified"] = (bowl_df["balls_p"] >= min_balls) & (bowl_df["matches_p"] >= min_matches)
    
    ranked = bowl_df[bowl_df["is_qualified"]].sort_values(
        by=["bpri", "balls_p", "econ_p"], ascending=[False, False, True]
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    
    return ranked

def format_top10_output(ranked_df, role="batter"):
    """
    Formats the submission-ready Top 10 table with exact column labels and rounding.
    """
    top10 = ranked_df.head(10).copy()
    
    if role == "batter":
        cols = [
            "rank", "striker_name", "bpri", "balls_p", "matches_p", "neff_p",
            "sr_p", "sr_exp", "delta_sr_ctx", "avg_p", "sr_b", "avg_b", "dot_pct_p", "bound_pct_p"
        ]
        col_names = {
            "rank": "Rank",
            "striker_name": "Player Name",
            "bpri": "BPRI Score",
            "balls_p": "Pressure Balls",
            "matches_p": "Matches",
            "neff_p": "Effective Balls (N_eff)",
            "sr_p": "Pressure SR",
            "sr_exp": "Expected SR",
            "delta_sr_ctx": "Context Delta SR",
            "avg_p": "Pressure Avg",
            "sr_b": "Baseline SR",
            "avg_b": "Baseline Avg",
            "dot_pct_p": "Dot %",
            "bound_pct_p": "Boundary %"
        }
        res = top10[cols].rename(columns=col_names)
        res["Effective Balls (N_eff)"] = res["Effective Balls (N_eff)"].round(1)
        res["Pressure SR"] = res["Pressure SR"].round(1)
        res["Expected SR"] = res["Expected SR"].round(1)
        res["Context Delta SR"] = res["Context Delta SR"].round(1)
        res["Pressure Avg"] = res["Pressure Avg"].round(1)
        res["Baseline SR"] = res["Baseline SR"].round(1)
        res["Baseline Avg"] = res["Baseline Avg"].round(1)
        res["Dot %"] = res["Dot %"].round(1)
        res["Boundary %"] = res["Boundary %"].round(1)
    else:
        cols = [
            "rank", "bowler_name", "bpri", "balls_p", "matches_p", "neff_p",
            "econ_p", "econ_exp", "delta_econ_ctx", "wkt_rate_p", "econ_b", "wkt_rate_b", "dot_pct_p", "bound_pct_p"
        ]
        col_names = {
            "rank": "Rank",
            "bowler_name": "Player Name",
            "bpri": "BPRI Score",
            "balls_p": "Pressure Balls",
            "matches_p": "Matches",
            "neff_p": "Effective Balls (N_eff)",
            "econ_p": "Pressure Econ",
            "econ_exp": "Expected Econ",
            "delta_econ_ctx": "Context Delta Econ",
            "wkt_rate_p": "Wicket %",
            "econ_b": "Baseline Econ",
            "wkt_rate_b": "Baseline Wkt %",
            "dot_pct_p": "Dot %",
            "bound_pct_p": "Boundary %"
        }
        res = top10[cols].rename(columns=col_names)
        res["Effective Balls (N_eff)"] = res["Effective Balls (N_eff)"].round(1)
        res["Pressure Econ"] = res["Pressure Econ"].round(2)
        res["Expected Econ"] = res["Expected Econ"].round(2)
        res["Context Delta Econ"] = res["Context Delta Econ"].round(2)
        res["Wicket %"] = (res["Wicket %"] * 100).round(1)
        res["Baseline Econ"] = res["Baseline Econ"].round(2)
        res["Baseline Wkt %"] = (res["Baseline Wkt %"] * 100).round(1)
        res["Dot %"] = res["Dot %"].round(1)
        res["Boundary %"] = res["Boundary %"].round(1)
        
    return res
