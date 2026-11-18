"""
SupeRR Selector Pressure-Performance Model & Pipeline
Standalone, reproducible training, evaluation, and ranking system.

Usage:
    python outputs/pressure_model.py [--data-dir <path>] [--output-dir <path>]

This script:
1. Ingests all 6 CSV files from the Data Pack and enforces MCC cricket rules.
2. Excludes revised-overs and revised-target matches to prevent post-interruption leakage.
3. Reconstructs pre-delivery match states strictly prior to each ball bowled.
4. Trains pre-delivery expected-outcome models (HistGradientBoosting) on 2022-2024 matches.
5. Saves the trained models to saved_model.pkl.
6. Evaluates alternative model architectures (Constant, Ridge, Poisson, HGB) on the 2025-2026 holdout.
7. Computes match-level Intra-Cluster Correlation (ICC) and design effects (DEFF) for effective sample sizes.
8. Applies Empirical Bayes shrinkage to dual-baseline differentials.
9. Ranks the Top 10 Batters and Top 10 Bowlers under pressure.
10. Performs sensitivity checks across Standard, Strict, and Broad pressure definitions.
11. Generates publication-quality charts and CSV exports.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, QUALIFICATION_RULES, ICC_PARAMS, SHRINKAGE_PARAMS, TRAIN_DATE_CUTOFF
from src.data_loader import load_dataset
from src.pressure_engine import compute_predelivery_state
from src.metrics import fit_expected_outcome_models, attach_expected_outcomes, calculate_batting_metrics, calculate_bowling_metrics
from src.ranker import rank_batters, rank_bowlers, format_top10_output
from src.evaluator import evaluate_temporal_model
from src.visualizer import plot_top10_rankings, plot_validation_comparison, plot_error_distribution, plot_rank_evolution, plot_metrics_scatter, plot_sensitivity_ranks

def train_and_evaluate(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Executes the complete pressure model pipeline.
    """
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 80)
    print("  SUPER-RR SELECTOR: PRESSURE-PERFORMANCE RETRAINING & EVALUATION")
    print("=" * 80)
    print(f"Data Source:       {data_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Figures Directory: {figures_dir}")
    print("-" * 80)
    
    # 1. Ingest and Clean Dataset (Excluding revised-overs/target matches)
    dfs = load_dataset(data_dir, exclude_revised=True)
    
    # 2. Pre-Delivery Match State Reconstruction (Zero Leakage)
    dfs = compute_predelivery_state(dfs)
    bbb = dfs["bbb"]
    matches = dfs["matches"]
    
    # 3. Model Training & Out-of-Sample Holdout Evaluation
    val_summary, bat_shifts, bowl_shifts, trained_models = evaluate_temporal_model(
        bbb, matches, output_dir=output_dir
    )
    
    # 4. Save Trained Models to File
    model_save_path = os.path.join(output_dir, "saved_model.pkl")
    model_artifacts = {
        "models": trained_models,
        "icc_params": ICC_PARAMS,
        "shrinkage_params": SHRINKAGE_PARAMS,
        "qualification_rules": QUALIFICATION_RULES,
        "train_date_cutoff": TRAIN_DATE_CUTOFF,
        "version": "2.0-enhanced-context-deff"
    }
    joblib.dump(model_artifacts, model_save_path)
    print(f"\n[MODEL SERIALIZATION] Successfully saved trained pressure models to: {model_save_path}")
    
    # 5. Full Dataset Scoring (2022-2026) using Fitted Context Models
    print("\n[SCORING] Calculating full-dataset batting metrics (2022-2026)...")
    bat_df, bat_pop = calculate_batting_metrics(bbb, pressure_col="is_pressure_std", models=trained_models)
    
    print("[SCORING] Calculating full-dataset bowling metrics (2022-2026)...")
    bowl_df, bowl_pop = calculate_bowling_metrics(bbb, pressure_col="is_pressure_std", models=trained_models)
    
    # 6. Generate Official Top 10 Rankings
    print("\n[RANKER] Ranking Batters...")
    ranked_batters = rank_batters(bat_df, bat_pop)
    top10_batters = format_top10_output(ranked_batters, role="batter")
    
    print("[RANKER] Ranking Bowlers...")
    ranked_bowlers = rank_bowlers(bowl_df, bowl_pop)
    top10_bowlers = format_top10_output(ranked_bowlers, role="bowler")
    
    # Export CSVs
    top10_batters_csv = os.path.join(output_dir, "top10_batters.csv")
    top10_bowlers_csv = os.path.join(output_dir, "top10_bowlers.csv")
    full_batters_csv = os.path.join(output_dir, "full_batter_rankings.csv")
    full_bowlers_csv = os.path.join(output_dir, "full_bowler_rankings.csv")
    
    top10_batters.to_csv(top10_batters_csv, index=False)
    top10_bowlers.to_csv(top10_bowlers_csv, index=False)
    ranked_batters.to_csv(full_batters_csv, index=False)
    ranked_bowlers.to_csv(full_bowlers_csv, index=False)
    
    print(f"\n[EXPORT] Saved Top 10 Batters to:  {top10_batters_csv}")
    print(f"[EXPORT] Saved Top 10 Bowlers to:  {top10_bowlers_csv}")
    print(f"[EXPORT] Saved Full Batters to:    {full_batters_csv}")
    print(f"[EXPORT] Saved Full Bowlers to:    {full_bowlers_csv}")
    
    # 7. Robustness & Sensitivity Checks Across Pressure Definitions
    print("\n[SENSITIVITY] Running robustness checks across pressure definitions...")
    # Strict
    bat_strict, bat_strict_pop = calculate_batting_metrics(bbb, pressure_col="is_pressure_strict", models=trained_models)
    bowl_strict, bowl_strict_pop = calculate_bowling_metrics(bbb, pressure_col="is_pressure_strict", models=trained_models)
    ranked_bat_strict = rank_batters(bat_strict, bat_strict_pop, min_balls=90, min_matches=12)
    ranked_bowl_strict = rank_bowlers(bowl_strict, bowl_strict_pop, min_balls=100, min_matches=12)
    
    # Broad
    bat_broad, bat_broad_pop = calculate_batting_metrics(bbb, pressure_col="is_pressure_broad", models=trained_models)
    bowl_broad, bowl_broad_pop = calculate_bowling_metrics(bbb, pressure_col="is_pressure_broad", models=trained_models)
    ranked_bat_broad = rank_batters(bat_broad, bat_broad_pop, min_balls=150, min_matches=18)
    ranked_bowl_broad = rank_bowlers(bowl_broad, bowl_broad_pop, min_balls=180, min_matches=18)
    
    # Sensitivity Summary Tables
    top10_bat_names = ranked_batters.head(10)["striker_name"].tolist()
    sens_bat_rows = []
    for pname in top10_bat_names:
        r_std = ranked_batters[ranked_batters["striker_name"] == pname]["rank"].iloc[0]
        r_strict = ranked_bat_strict[ranked_bat_strict["striker_name"] == pname]["rank"].iloc[0] if pname in ranked_bat_strict["striker_name"].values else np.nan
        r_broad = ranked_bat_broad[ranked_bat_broad["striker_name"] == pname]["rank"].iloc[0] if pname in ranked_bat_broad["striker_name"].values else np.nan
        sens_bat_rows.append({"striker_name": pname, "Rank_Standard": r_std, "Rank_Strict": r_strict, "Rank_Broad": r_broad})
    df_sens_bat = pd.DataFrame(sens_bat_rows)
    
    top10_bowl_names = ranked_bowlers.head(10)["bowler_name"].tolist()
    sens_bowl_rows = []
    for pname in top10_bowl_names:
        r_std = ranked_bowlers[ranked_bowlers["bowler_name"] == pname]["rank"].iloc[0]
        r_strict = ranked_bowl_strict[ranked_bowl_strict["bowler_name"] == pname]["rank"].iloc[0] if pname in ranked_bowl_strict["bowler_name"].values else np.nan
        r_broad = ranked_bowl_broad[ranked_bowl_broad["bowler_name"] == pname]["rank"].iloc[0] if pname in ranked_bowl_broad["bowler_name"].values else np.nan
        sens_bowl_rows.append({"bowler_name": pname, "Rank_Standard": r_std, "Rank_Strict": r_strict, "Rank_Broad": r_broad})
    df_sens_bowl = pd.DataFrame(sens_bowl_rows)
    
    sens_csv = os.path.join(output_dir, "sensitivity_analysis.csv")
    pd.concat([df_sens_bat.assign(Role="Batter"), df_sens_bowl.assign(Role="Bowler")], ignore_index=True).to_csv(sens_csv, index=False)
    print(f"[EXPORT] Saved Sensitivity Analysis to: {sens_csv}")
    
    # 8. Generate Visualizations
    print("\n[VISUALIZER] Generating publication-quality charts...")
    plot_top10_rankings(top10_batters, top10_bowlers, figures_dir)
    plot_validation_comparison(val_summary, figures_dir)
    
    # Attach predictions to full bbb for visualization
    bbb = attach_expected_outcomes(bbb, trained_models)
    
    # Error distribution plot on test set
    te_bat_balls = bbb[(bbb["is_batter_ball"]) & (bbb["match_date"] >= TRAIN_DATE_CUTOFF)].copy()
    if len(te_bat_balls) > 0 and "xruns_bat" in te_bat_balls.columns:
        plot_error_distribution(te_bat_balls["runs"], te_bat_balls["xruns_bat"], figures_dir)
        
    plot_rank_evolution(bat_shifts, bowl_shifts, figures_dir)
    plot_metrics_scatter(ranked_batters, ranked_bowlers, figures_dir)
    plot_sensitivity_ranks(df_sens_bat, df_sens_bowl, figures_dir)
    
    # Print Executive Summary in Terminal
    print("\n" + "=" * 80)
    print("  OFFICIAL SUBMISSION: TOP 10 BATTERS UNDER PRESSURE (2022-2026)")
    print("=" * 80)
    print(top10_batters.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("  OFFICIAL SUBMISSION: TOP 10 BOWLERS UNDER PRESSURE (2022-2026)")
    print("=" * 80)
    print(top10_bowlers.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("  EXACT COMMAND TO RETRAIN MODEL:")
    print("  python outputs/pressure_model.py")
    print("  (or: python run_analysis.py)")
    print("=" * 80)
    
    return {
        "top10_batters": top10_batters,
        "top10_bowlers": top10_bowlers,
        "val_summary": val_summary,
        "models": trained_models
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SupeRR Selector Pressure Model Pipeline")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to Data Pack folder")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    args = parser.parse_args()
    
    train_and_evaluate(args.data_dir, args.output_dir)
