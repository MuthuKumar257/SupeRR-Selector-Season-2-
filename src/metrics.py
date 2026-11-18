"""
Performance Metrics module for SupeRR Selector Analysis System.
Calculates role-appropriate metrics for batters and bowlers under pressure and baseline,
trains pre-delivery expected-outcome models, calculates match-level Intra-Cluster Correlation (ICC)
and design effects (DEFF), applies Empirical Bayes shrinkage, and derives dual-baseline differentials.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from src.config import SHRINKAGE_PARAMS, IPL_COMP_IDS, ICC_PARAMS, PRE_DELIVERY_FEATURES, TRAIN_DATE_CUTOFF

def fit_expected_outcome_models(bbb, train_cutoff=TRAIN_DATE_CUTOFF):
    """
    Fits pre-delivery expected-outcome models strictly on training matches (before train_cutoff).
    Zero data leakage: features are computed prior to each delivery, and holdout data is unseen.
    """
    print(f"[METRICS ENGINE] Fitting pre-delivery expected outcome models on training set (< {train_cutoff})...")
    df = bbb.copy()
    
    # Fill feature nulls
    for col in PRE_DELIVERY_FEATURES:
        df[col] = df[col].fillna(0)
        
    train_mask = df["match_date"] < train_cutoff
    
    # 1. Batting Models
    df_bat = df[df["is_batter_ball"]].copy()
    tr_bat = df_bat[df_bat["match_date"] < train_cutoff]
    
    bat_runs_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    bat_runs_model.fit(tr_bat[PRE_DELIVERY_FEATURES], tr_bat["runs"])
    
    bat_wkt_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    bat_wkt_model.fit(tr_bat[PRE_DELIVERY_FEATURES], tr_bat["is_striker_out"].astype(int))
    
    # 2. Bowling Models
    df_bowl = df[df["is_legal"]].copy()
    tr_bowl = df_bowl[df_bowl["match_date"] < train_cutoff]
    
    bowl_runs_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    bowl_runs_model.fit(tr_bowl[PRE_DELIVERY_FEATURES], tr_bowl["bowler_runs_conceded"])
    
    bowl_wkt_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    bowl_wkt_model.fit(tr_bowl[PRE_DELIVERY_FEATURES], tr_bowl["is_bowler_wicket"].astype(int))
    
    models = {
        "bat_runs": bat_runs_model,
        "bat_wkt": bat_wkt_model,
        "bowl_runs": bowl_runs_model,
        "bowl_wkt": bowl_wkt_model,
        "features": PRE_DELIVERY_FEATURES,
        "train_cutoff": train_cutoff
    }
    print("[METRICS ENGINE] Expected-outcome models successfully trained.")
    return models

def attach_expected_outcomes(bbb, models):
    """
    Applies fitted models to generate context-expected runs and wicket probabilities
    for every delivery in the dataset.
    """
    df = bbb.copy()
    for col in PRE_DELIVERY_FEATURES:
        df[col] = df[col].fillna(0)
        
    # Batting expectations
    df["xruns_bat"] = np.nan
    df["xwkt_bat"] = np.nan
    bat_idx = df[df["is_batter_ball"]].index
    if len(bat_idx) > 0:
        df.loc[bat_idx, "xruns_bat"] = models["bat_runs"].predict(df.loc[bat_idx, PRE_DELIVERY_FEATURES])
        df.loc[bat_idx, "xwkt_bat"] = models["bat_wkt"].predict_proba(df.loc[bat_idx, PRE_DELIVERY_FEATURES])[:, 1]
        
    # Bowling expectations
    df["xruns_bowl"] = np.nan
    df["xwkt_bowl"] = np.nan
    bowl_idx = df[df["is_legal"]].index
    if len(bowl_idx) > 0:
        df.loc[bowl_idx, "xruns_bowl"] = models["bowl_runs"].predict(df.loc[bowl_idx, PRE_DELIVERY_FEATURES])
        df.loc[bowl_idx, "xwkt_bowl"] = models["bowl_wkt"].predict_proba(df.loc[bowl_idx, PRE_DELIVERY_FEATURES])[:, 1]
        
    return df

def calculate_batting_metrics(bbb, pressure_col="is_pressure_std", models=None):
    """
    Computes batting metrics under pressure vs baseline (non-pressure) for all batters.
    Incorporates context-expected predictions, match-level ICC / DEFF effective sample size,
    and Empirical Bayes shrinkage.
    """
    df_bat = bbb[bbb["is_batter_ball"]].copy()
    df_bat["p"] = df_bat[pressure_col]
    df_bat["is_ipl"] = df_bat["comp_id"].isin(IPL_COMP_IDS)
    
    if models is not None and "xruns_bat" not in df_bat.columns:
        df_bat = attach_expected_outcomes(df_bat, models)
    elif "xruns_bat" not in df_bat.columns:
        models = fit_expected_outcome_models(bbb)
        df_bat = attach_expected_outcomes(df_bat, models)
        
    pop_p = df_bat[df_bat["p"]]
    pop_b = df_bat[~df_bat["p"]]
    
    pop_stats = {
        "pop_sr_p": (pop_p["runs"].sum() / len(pop_p)) * 100,
        "pop_avg_p": pop_p["runs"].sum() / max(pop_p["is_striker_out"].sum(), 1),
        "pop_dot_p": (pop_p["runs"] == 0).mean() * 100,
        "pop_bound_p": ((pop_p["is_four"] | pop_p["is_six"]).mean()) * 100,
        "pop_sr_b": (pop_b["runs"].sum() / len(pop_b)) * 100,
        "pop_avg_b": pop_b["runs"].sum() / max(pop_b["is_striker_out"].sum(), 1)
    }
    
    # 1. Under Pressure Aggregations
    bp = df_bat[df_bat["p"]].groupby(["striker_id", "striker_name"]).agg(
        balls_p=("is_batter_ball", "count"),
        matches_p=("match_id", "nunique"),
        runs_p=("runs", "sum"),
        xruns_p=("xruns_bat", "sum"),
        outs_p=("is_striker_out", "sum"),
        xwkt_p=("xwkt_bat", "sum"),
        dots_p=("runs", lambda x: (x == 0).sum()),
        fours_p=("is_four", "sum"),
        sixes_p=("is_six", "sum"),
        runs_std=("runs", "std"),
        ipl_balls_p=("is_ipl", lambda x: x.sum()),
        ipl_runs_p=("runs", lambda x: x[df_bat.loc[x.index, "is_ipl"]].sum())
    ).reset_index()
    
    # 2. Baseline (Non-Pressure) Aggregations
    bb = df_bat[~df_bat["p"]].groupby(["striker_id", "striker_name"]).agg(
        balls_b=("is_batter_ball", "count"),
        matches_b=("match_id", "nunique"),
        runs_b=("runs", "sum"),
        outs_b=("is_striker_out", "sum"),
        dots_b=("runs", lambda x: (x == 0).sum()),
        fours_b=("is_four", "sum"),
        sixes_b=("is_six", "sum")
    ).reset_index()
    
    bat_df = bp.merge(bb, on=["striker_id", "striker_name"], how="left").fillna(0)
    
    # Compute observed rates
    bat_df["sr_p"] = (bat_df["runs_p"] / bat_df["balls_p"]) * 100
    bat_df["sr_exp"] = (bat_df["xruns_p"] / bat_df["balls_p"]) * 100
    bat_df["delta_sr_ctx"] = bat_df["sr_p"] - bat_df["sr_exp"]  # Context-adjusted differential
    bat_df["avg_p"] = bat_df["runs_p"] / bat_df["outs_p"].clip(lower=1)
    bat_df["dot_pct_p"] = (bat_df["dots_p"] / bat_df["balls_p"]) * 100
    bat_df["bound_pct_p"] = ((bat_df["fours_p"] + bat_df["sixes_p"]) / bat_df["balls_p"]) * 100
    
    bat_df["sr_b"] = np.where(bat_df["balls_b"] > 0, (bat_df["runs_b"] / bat_df["balls_b"]) * 100, pop_stats["pop_sr_b"])
    bat_df["avg_b"] = np.where(bat_df["balls_b"] > 0, bat_df["runs_b"] / bat_df["outs_b"].clip(lower=1), pop_stats["pop_avg_b"])
    bat_df["dot_pct_b"] = np.where(bat_df["balls_b"] > 0, (bat_df["dots_b"] / bat_df["balls_b"]) * 100, pop_stats["pop_dot_p"])
    bat_df["bound_pct_b"] = np.where(bat_df["balls_b"] > 0, ((bat_df["fours_b"] + bat_df["sixes_b"]) / bat_df["balls_b"]) * 100, pop_stats["pop_bound_p"])
    
    # Match-level clustering & Effective sample size
    bat_icc = ICC_PARAMS["batting_icc"]
    bat_df["m_bar"] = bat_df["balls_p"] / bat_df["matches_p"].clip(lower=1)
    bat_df["deff"] = 1.0 + (bat_df["m_bar"] - 1.0) * bat_icc
    bat_df["neff_p"] = bat_df["balls_p"] / bat_df["deff"]
    
    # Empirical Bayes Shrinkage
    k_sr = SHRINKAGE_PARAMS["batter_k_balls"]
    k_avg = SHRINKAGE_PARAMS["batter_k_outs"]
    bat_df["w_sr"] = bat_df["neff_p"] / (bat_df["neff_p"] + k_sr)
    bat_df["w_avg"] = bat_df["neff_p"] / (bat_df["neff_p"] + k_avg)
    
    bat_df["sr_shrunk"] = bat_df["w_sr"] * bat_df["sr_p"] + (1 - bat_df["w_sr"]) * pop_stats["pop_sr_p"]
    bat_df["avg_shrunk"] = bat_df["w_avg"] * bat_df["avg_p"] + (1 - bat_df["w_avg"]) * pop_stats["pop_avg_p"]
    
    # Standard Errors & Uncertainty
    bat_df["runs_std"] = bat_df["runs_std"].fillna(pop_p["runs"].std())
    bat_df["sr_se"] = (bat_df["runs_std"] / np.sqrt(bat_df["neff_p"].clip(lower=1))) * 100
    bat_df["sr_ci_lower"] = (bat_df["sr_shrunk"] - 1.96 * bat_df["sr_se"]).round(1)
    bat_df["sr_ci_upper"] = (bat_df["sr_shrunk"] + 1.96 * bat_df["sr_se"]).round(1)
    
    return bat_df, pop_stats

def calculate_bowling_metrics(bbb, pressure_col="is_pressure_std", models=None):
    """
    Computes bowling metrics under pressure vs baseline (non-pressure) for all bowlers.
    Incorporates context-expected predictions, match-level ICC / DEFF effective sample size,
    and Empirical Bayes shrinkage.
    """
    df_bowl = bbb[bbb["is_legal"]].copy()
    df_bowl["p"] = df_bowl[pressure_col]
    df_bowl["is_ipl"] = df_bowl["comp_id"].isin(IPL_COMP_IDS)
    
    if models is not None and "xruns_bowl" not in df_bowl.columns:
        df_bowl = attach_expected_outcomes(df_bowl, models)
    elif "xruns_bowl" not in df_bowl.columns:
        models = fit_expected_outcome_models(bbb)
        df_bowl = attach_expected_outcomes(df_bowl, models)
        
    pop_p = df_bowl[df_bowl["p"]]
    pop_b = df_bowl[~df_bowl["p"]]
    
    pop_stats = {
        "pop_econ_p": (pop_p["bowler_runs_conceded"].sum() / len(pop_p)) * 6.0,
        "pop_wkt_p": pop_p["is_bowler_wicket"].mean(),
        "pop_dot_p": (pop_p["bowler_runs_conceded"] == 0).mean() * 100,
        "pop_bound_p": ((pop_p["is_four"] | pop_p["is_six"]).mean()) * 100,
        "pop_econ_b": (pop_b["bowler_runs_conceded"].sum() / len(pop_b)) * 6.0,
        "pop_wkt_b": pop_b["is_bowler_wicket"].mean()
    }
    
    # 1. Under Pressure Aggregations
    bp = df_bowl[df_bowl["p"]].groupby(["bowler_id", "bowler_name"]).agg(
        balls_p=("is_legal", "count"),
        matches_p=("match_id", "nunique"),
        runs_p=("bowler_runs_conceded", "sum"),
        xruns_p=("xruns_bowl", "sum"),
        wkts_p=("is_bowler_wicket", "sum"),
        xwkt_p=("xwkt_bowl", "sum"),
        dots_p=("bowler_runs_conceded", lambda x: (x == 0).sum()),
        fours_p=("is_four", "sum"),
        sixes_p=("is_six", "sum"),
        runs_std=("bowler_runs_conceded", "std"),
        ipl_balls_p=("is_ipl", lambda x: x.sum()),
        ipl_runs_p=("bowler_runs_conceded", lambda x: x[df_bowl.loc[x.index, "is_ipl"]].sum())
    ).reset_index()
    
    # 2. Baseline (Non-Pressure) Aggregations
    bb = df_bowl[~df_bowl["p"]].groupby(["bowler_id", "bowler_name"]).agg(
        balls_b=("is_legal", "count"),
        matches_b=("match_id", "nunique"),
        runs_b=("bowler_runs_conceded", "sum"),
        wkts_b=("is_bowler_wicket", "sum"),
        dots_b=("bowler_runs_conceded", lambda x: (x == 0).sum()),
        fours_b=("is_four", "sum"),
        sixes_b=("is_six", "sum")
    ).reset_index()
    
    bowl_df = bp.merge(bb, on=["bowler_id", "bowler_name"], how="left").fillna(0)
    
    # Compute observed rates
    bowl_df["econ_p"] = (bowl_df["runs_p"] / bowl_df["balls_p"]) * 6.0
    bowl_df["econ_exp"] = (bowl_df["xruns_p"] / bowl_df["balls_p"]) * 6.0
    bowl_df["delta_econ_ctx"] = bowl_df["econ_exp"] - bowl_df["econ_p"] # Positive is good (runs saved per over)
    bowl_df["wkt_rate_p"] = bowl_df["wkts_p"] / bowl_df["balls_p"]
    bowl_df["wkt_exp"] = bowl_df["xwkt_p"] / bowl_df["balls_p"]
    bowl_df["delta_wkt_ctx"] = bowl_df["wkt_rate_p"] - bowl_df["wkt_exp"]
    
    bowl_df["dot_pct_p"] = (bowl_df["dots_p"] / bowl_df["balls_p"]) * 100
    bowl_df["bound_pct_p"] = ((bowl_df["fours_p"] + bowl_df["sixes_p"]) / bowl_df["balls_p"]) * 100
    
    bowl_df["econ_b"] = np.where(bowl_df["balls_b"] > 0, (bowl_df["runs_b"] / bowl_df["balls_b"]) * 6.0, pop_stats["pop_econ_b"])
    bowl_df["wkt_rate_b"] = np.where(bowl_df["balls_b"] > 0, bowl_df["wkts_b"] / bowl_df["balls_b"], pop_stats["pop_wkt_b"])
    bowl_df["dot_pct_b"] = np.where(bowl_df["balls_b"] > 0, (bowl_df["dots_b"] / bowl_df["balls_b"]) * 100, pop_stats["pop_dot_p"])
    bowl_df["bound_pct_b"] = np.where(bowl_df["balls_b"] > 0, ((bowl_df["fours_b"] + bowl_df["sixes_b"]) / bowl_df["balls_b"]) * 100, pop_stats["pop_bound_p"])
    
    # Match-level clustering & Effective sample size
    bowl_icc = ICC_PARAMS["bowling_icc"]
    bowl_df["m_bar"] = bowl_df["balls_p"] / bowl_df["matches_p"].clip(lower=1)
    bowl_df["deff"] = 1.0 + (bowl_df["m_bar"] - 1.0) * bowl_icc
    bowl_df["neff_p"] = bowl_df["balls_p"] / bowl_df["deff"]
    
    # Empirical Bayes Shrinkage
    k_econ = SHRINKAGE_PARAMS["bowler_k_balls"]
    k_wkt = SHRINKAGE_PARAMS["bowler_k_wkt"]
    bowl_df["w_econ"] = bowl_df["neff_p"] / (bowl_df["neff_p"] + k_econ)
    bowl_df["w_wkt"] = bowl_df["neff_p"] / (bowl_df["neff_p"] + k_wkt)
    
    bowl_df["econ_shrunk"] = bowl_df["w_econ"] * bowl_df["econ_p"] + (1 - bowl_df["w_econ"]) * pop_stats["pop_econ_p"]
    bowl_df["wkt_rate_shrunk"] = bowl_df["w_wkt"] * bowl_df["wkt_rate_p"] + (1 - bowl_df["w_wkt"]) * pop_stats["pop_wkt_p"]
    
    # Standard Errors & Uncertainty
    bowl_df["runs_std"] = bowl_df["runs_std"].fillna(pop_p["bowler_runs_conceded"].std())
    bowl_df["econ_se"] = (bowl_df["runs_std"] / np.sqrt(bowl_df["neff_p"].clip(lower=1))) * 6.0
    bowl_df["econ_ci_lower"] = (bowl_df["econ_shrunk"] - 1.96 * bowl_df["econ_se"]).round(2)
    bowl_df["econ_ci_upper"] = (bowl_df["econ_shrunk"] + 1.96 * bowl_df["econ_se"]).round(2)
    
    return bowl_df, pop_stats
