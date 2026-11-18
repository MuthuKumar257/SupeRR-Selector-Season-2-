"""
Visualization module for SupeRR Selector Analysis System.
Generates publication-quality charts and visual artifacts for rankings,
metric distributions, out-of-sample model evaluations, and sensitivity comparisons.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_top10_rankings(top10_batters, top10_bowlers, figures_dir):
    """
    Horizontal bar charts displaying the official Top 10 Batters and Bowlers under pressure.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Batters Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df_b = top10_batters.sort_values(by="BPRI Score", ascending=True)
    bars = ax.barh(df_b["Player Name"], df_b["BPRI Score"], color="#1f77b4", edgecolor="black", height=0.65)
    
    ax.set_xlabel("Batting Pressure Reliability Index (BPRI)", fontweight="bold", fontsize=11)
    ax.set_title("Top 10 Batters Performing Most Reliably Under Pressure (2022-2026)\nContext-Adjusted, DEFF Shrunk & Leak-Free", fontweight="bold", fontsize=12)
    ax.set_xlim(0, max(df_b["BPRI Score"]) * 1.15)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}", ha="left", va="center", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    b_path = os.path.join(figures_dir, "top10_batters_ranking.png")
    plt.savefig(b_path, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {b_path}")
    
    # 2. Bowlers Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df_w = top10_bowlers.sort_values(by="BPRI Score", ascending=True)
    bars = ax.barh(df_w["Player Name"], df_w["BPRI Score"], color="#2ca02c", edgecolor="black", height=0.65)
    
    ax.set_xlabel("Bowling Pressure Reliability Index (BPRI)", fontweight="bold", fontsize=11)
    ax.set_title("Top 10 Bowlers Performing Most Reliably Under Pressure (2022-2026)\nContext-Adjusted, DEFF Shrunk & Leak-Free", fontweight="bold", fontsize=12)
    ax.set_xlim(0, max(df_w["BPRI Score"]) * 1.15)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}", ha="left", va="center", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    w_path = os.path.join(figures_dir, "top10_bowlers_ranking.png")
    plt.savefig(w_path, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {w_path}")

def plot_validation_comparison(summary_df, figures_dir):
    """
    Bar chart comparing out-of-sample RMSE across baseline, naive prior, and retrained pressure model.
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Batting
    bat_row = summary_df[summary_df["Role"].str.contains("Batting")].iloc[0]
    bars1 = ax1.bar(["Naive Pop Prior", "Simple Baseline\n(Non-Press SR)", "Retrained Model\n(Context Shrunk)"],
                    [bat_row["Naive_Prior_RMSE"], bat_row["Simple_Baseline_RMSE"], bat_row["Retrained_Model_RMSE"]],
                    color=["#95a5a6", "#e74c3c", "#2ecc71"], width=0.55, edgecolor="black")
    ax1.set_ylabel("Holdout Strike Rate RMSE (Lower is Better)", fontweight="bold")
    ax1.set_title(f"Batting: {bat_row['Error_Reduction_Pct']}% Error Reduction vs Baseline\n(Evaluated on 2025-2026 Matches)", fontweight="bold", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, max(bat_row["Simple_Baseline_RMSE"], bat_row["Naive_Prior_RMSE"]) * 1.2)
    
    # Bowling
    bowl_row = summary_df[summary_df["Role"].str.contains("Bowling")].iloc[0]
    bars2 = ax2.bar(["Naive Pop Prior", "Simple Baseline\n(Non-Press Econ)", "Retrained Model\n(Context Shrunk)"],
                    [bowl_row["Naive_Prior_RMSE"], bowl_row["Simple_Baseline_RMSE"], bowl_row["Retrained_Model_RMSE"]],
                    color=["#95a5a6", "#e74c3c", "#27ae60"], width=0.55, edgecolor="black")
    ax2.set_ylabel("Holdout Economy RMSE (Lower is Better)", fontweight="bold")
    ax2.set_title(f"Bowling: {bowl_row['Error_Reduction_Pct']}% Error Reduction vs Baseline\n(Evaluated on 2025-2026 Matches)", fontweight="bold", fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, max(bowl_row["Simple_Baseline_RMSE"], bowl_row["Naive_Prior_RMSE"]) * 1.25)
    
    plt.tight_layout()
    fname = os.path.join(figures_dir, "model_validation_comparison.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {fname}")

def plot_error_distribution(y_true, y_pred, figures_dir):
    """
    Plots the residual distribution of the ball-level expected-outcome model,
    showing why discrete variance dominates at the single-delivery level.
    """
    os.makedirs(figures_dir, exist_ok=True)
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=np.arange(-2.5, 6.5, 0.5), color="#3498db", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Mean Residual (Zero-Bias Baseline)")
    
    ax.set_title("Single-Delivery Residual Distribution of Expected-Runs Model\n(High Discrete Noise: Dots, Singles, Fours, Sixes)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Residual (Observed Runs - Expected Runs)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    
    # Annotate discrete outcome spikes
    ax.text(0 - 1.2, 0.35, "Dot Ball Residual\n(~ -1.2 to -1.5)", ha="center", fontsize=9, color="#2c3e50")
    ax.text(4 - 1.4, 0.12, "Four Boundary\n(+2.4 to +2.7)", ha="center", fontsize=9, color="#e67e22")
    ax.text(6 - 1.4, 0.06, "Six Boundary\n(+4.4 to +4.7)", ha="center", fontsize=9, color="#e74c3c")
    
    plt.tight_layout()
    fname = os.path.join(figures_dir, "expected_outcome_error_distribution.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {fname}")

def plot_rank_evolution(bat_shifts, bowl_shifts, figures_dir):
    """
    Plots rank transitions from Train Split (2022-2024) to Full Retrained Model (2022-2026).
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Batters
    for i, row in bat_shifts.iterrows():
        pname = row["striker_name (Full)"]
        r_full = row["Rank"]
        if pname in bat_shifts["striker_name (Train)"].values:
            r_train = bat_shifts[bat_shifts["striker_name (Train)"] == pname]["Rank"].iloc[0]
        else:
            r_train = 12
        ax1.plot([1, 2], [r_train, r_full], marker="o", linewidth=2.2, label=pname)
        ax1.text(2.05, r_full, f" #{r_full} {pname}", va="center", fontsize=9, fontweight="bold")
        
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Train Split (2022-2024)", "Full Retrained (2022-2026)"], fontweight="bold")
    ax1.set_ylabel("Rank (1 = Highest)", fontweight="bold")
    ax1.set_ylim(13, 0.5)
    ax1.set_xlim(0.8, 2.8)
    ax1.set_title("Batters: Rank Evolution Across Time", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Bowlers
    for i, row in bowl_shifts.iterrows():
        pname = row["bowler_name (Full)"]
        r_full = row["Rank"]
        if pname in bowl_shifts["bowler_name (Train)"].values:
            r_train = bowl_shifts[bowl_shifts["bowler_name (Train)"] == pname]["Rank"].iloc[0]
        else:
            r_train = 12
        ax2.plot([1, 2], [r_train, r_full], marker="o", linewidth=2.2, label=pname)
        ax2.text(2.05, r_full, f" #{r_full} {pname}", va="center", fontsize=9, fontweight="bold")
        
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Train Split (2022-2024)", "Full Retrained (2022-2026)"], fontweight="bold")
    ax2.set_ylabel("Rank (1 = Highest)", fontweight="bold")
    ax2.set_ylim(13, 0.5)
    ax2.set_xlim(0.8, 2.8)
    ax2.set_title("Bowlers: Rank Evolution Across Time", fontweight="bold", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fname = os.path.join(figures_dir, "rank_evolution_train_vs_full.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {fname}")

def plot_metrics_scatter(bat_df, bowl_df, figures_dir):
    """
    Scatter plots displaying metric relationships with bubble size proportional to effective sample size.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Batting: Pressure SR vs Pressure Avg
    fig, ax = plt.subplots(figsize=(9, 6))
    qual_bat = bat_df[bat_df["is_qualified"]].copy()
    scatter = ax.scatter(
        qual_bat["avg_p"], qual_bat["sr_p"],
        s=qual_bat["neff_p"] * 0.75, c=qual_bat["bpri"], cmap="viridis",
        alpha=0.85, edgecolors="black", linewidth=1.2
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("BPRI Score", fontweight="bold")
    
    for i, row in qual_bat.head(10).iterrows():
        ax.annotate(
            row["striker_name"], (row["avg_p"], row["sr_p"]),
            xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold"
        )
        
    ax.set_xlabel("Pressure Batting Average (Runs per Dismissal)", fontweight="bold")
    ax.set_ylabel("Pressure Strike Rate", fontweight="bold")
    ax.set_title("Batting: Pressure Strike Rate vs Average\n(Bubble size = Effective Sample Size N_eff)", fontweight="bold", fontsize=12)
    
    plt.tight_layout()
    b_path = os.path.join(figures_dir, "batter_metrics_scatter.png")
    plt.savefig(b_path, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {b_path}")
    
    # 2. Bowling: Pressure Economy vs Wicket Rate
    fig, ax = plt.subplots(figsize=(9, 6))
    qual_bowl = bowl_df[bowl_df["is_qualified"]].copy()
    scatter2 = ax.scatter(
        qual_bowl["econ_p"], qual_bowl["wkt_rate_p"] * 100,
        s=qual_bowl["neff_p"] * 0.75, c=qual_bowl["bpri"], cmap="plasma",
        alpha=0.85, edgecolors="black", linewidth=1.2
    )
    cbar2 = plt.colorbar(scatter2, ax=ax)
    cbar2.set_label("BPRI Score", fontweight="bold")
    
    for i, row in qual_bowl.head(10).iterrows():
        ax.annotate(
            row["bowler_name"], (row["econ_p"], row["wkt_rate_p"] * 100),
            xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold"
        )
        
    ax.set_xlabel("Pressure Economy Rate (Runs per Over - Lower is Better)", fontweight="bold")
    ax.set_ylabel("Pressure Wicket Rate (% Deliveries Resulting in Bowler Wicket)", fontweight="bold")
    ax.set_title("Bowling: Pressure Economy vs Wicket Taking Rate\n(Bubble size = Effective Sample Size N_eff)", fontweight="bold", fontsize=12)
    ax.invert_xaxis()
    
    plt.tight_layout()
    w_path = os.path.join(figures_dir, "bowler_metrics_scatter.png")
    plt.savefig(w_path, dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved {w_path}")

def plot_sensitivity_ranks(df_bat_sens, df_bowl_sens, figures_dir):
    """
    Plots stability of player ranks across Standard, Strict, and Broad pressure definitions.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Batters
    for i, row in df_bat_sens.head(8).iterrows():
        pname = row["striker_name"]
        ax1.plot([1, 2, 3], [row["Rank_Standard"], row["Rank_Strict"], row["Rank_Broad"]],
                 marker="s", linewidth=2.0, label=pname)
        ax1.text(3.05, row["Rank_Broad"], f" {pname}", va="center", fontsize=8.5, fontweight="bold")
        
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["Standard", "Strict (Ultra)", "Broad (Death)"], fontweight="bold")
    ax1.set_ylabel("Rank Position", fontweight="bold")
    ax1.set_ylim(20, 0.5)
    ax1.set_xlim(0.8, 4.0)
    ax1.set_title("Batting Rank Stability Across Pressure Thresholds", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Bowlers
    for i, row in df_bowl_sens.head(8).iterrows():
        pname = row["bowler_name"]
        ax2.plot([1, 2, 3], [row["Rank_Standard"], row["Rank_Strict"], row["Rank_Broad"]],
                 marker="s", linewidth=2.0, label=pname)
        ax2.text(3.05, row["Rank_Broad"], f" {pname}", va="center", fontsize=8.5, fontweight="bold")
        
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Standard", "Strict (Ultra)", "Broad (Death)"], fontweight="bold")
    ax2.set_ylabel("Rank Position", fontweight="bold")
    ax2.set_ylim(20, 0.5)
    ax2.set_xlim(0.8, 4.0)
    ax2.set_title("Bowling Rank Stability Across Pressure Thresholds", fontweight="bold", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.90, wspace=0.25)
    fname = os.path.join(figures_dir, "sensitivity_stability_comparison.png")
    plt.savefig(fname, dpi=300)
    
    # Also save individual files for backward compatibility
    plt.savefig(os.path.join(figures_dir, "batter_sensitivity_stability.png"), dpi=300)
    plt.savefig(os.path.join(figures_dir, "bowler_sensitivity_stability.png"), dpi=300)
    plt.close()
    print(f"[VISUALIZER] Saved sensitivity stability charts to {figures_dir}")
