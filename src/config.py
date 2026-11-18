"""
Configuration module for SupeRR Selector Analysis System.
Provides default file paths, configurable pressure parameters, qualification thresholds,
and hyperparameter settings. All paths can be overridden via CLI arguments or environment variables.
"""

import os

# Default paths (relative to project root or Model directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detect Data Pack location across local and cloud environments
ENV_DATA_DIR = os.getenv("DATA_DIR")
if ENV_DATA_DIR and os.path.exists(ENV_DATA_DIR):
    DEFAULT_DATA_DIR = ENV_DATA_DIR
elif os.path.exists(os.path.join(BASE_DIR, "Data Pack")):
    DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "Data Pack")
elif os.path.exists(os.path.join(os.path.dirname(BASE_DIR), "Data Pack")):
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data Pack")
else:
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data Pack")

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))

# Configurable Pressure Definition Parameters
PRESSURE_CONFIG = {
    # Standard Definition
    "std": {
        "chase_death_balls": 36,
        "chase_min_rrr": 8.5,
        "chase_max_rrr_mult": 3.5,
        "chase_close_balls": 18,
        "chase_close_max_runs": 40,
        "crisis_pp_wickets": 3,
        "crisis_pp_balls_rem": 84,  # First 6 overs (120 - 36 = 84)
        "crisis_mid_wickets": 4,
        "crisis_mid_balls_rem": 60, # First 10 overs
        "death_inn1_balls": 24,
        "death_inn1_wickets": 4,
        "death_inn1_final_balls": 12,
    },
    # Strict Definition (Ultra-High Pressure)
    "strict": {
        "chase_death_balls": 30,
        "chase_min_rrr": 9.5,
        "chase_max_rrr_mult": 3.0,
        "chase_close_balls": 12,
        "chase_close_max_runs": 30,
        "crisis_pp_wickets": 3,
        "crisis_pp_balls_rem": 84,
        "crisis_mid_wickets": 5,
        "crisis_mid_balls_rem": 60,
        "death_inn1_balls": 18,
        "death_inn1_wickets": 4,
        "death_inn1_final_balls": 6,
    },
    # Broad Definition (Extended Pressure / All Death Overs)
    "broad": {
        "chase_death_balls": 48,
        "chase_min_rrr": 8.0,
        "chase_max_rrr_mult": 4.0,
        "chase_close_balls": 24,
        "chase_close_max_runs": 55,
        "crisis_pp_wickets": 3,
        "crisis_pp_balls_rem": 60,
        "crisis_mid_wickets": 4,
        "crisis_mid_balls_rem": 48,
        "death_inn1_balls": 30,
        "death_inn1_wickets": 3,
        "death_inn1_final_balls": 18,
    }
}

# Qualification & Sample Size Rules
QUALIFICATION_RULES = {
    "batter_min_balls": 120,      # Minimum pressure balls faced
    "batter_min_matches": 15,     # Minimum distinct matches with pressure balls
    "bowler_min_balls": 150,      # Minimum pressure legal balls bowled
    "bowler_min_matches": 15,     # Minimum distinct matches with pressure balls
}

# Intra-Cluster Correlation (ICC) parameters across matches for pressure deliveries
ICC_PARAMS = {
    "batting_icc": 0.0233,        # Intra-cluster correlation for batting runs per delivery across matches
    "bowling_icc": 0.0234,        # Intra-cluster correlation for bowling runs conceded per delivery across matches
}

# Empirical Bayes Shrinkage Hyperparameters
SHRINKAGE_PARAMS = {
    "batter_k_balls": 60.0,       # Credibility prior weight for strike rate (effective balls)
    "batter_k_outs": 100.0,       # Credibility prior weight for average (effective balls)
    "bowler_k_balls": 75.0,       # Credibility prior weight for economy (effective legal deliveries)
    "bowler_k_wkt": 120.0,        # Credibility prior weight for wicket rate (higher shrinkage for binary events)
}

# Pre-delivery feature column names for Machine Learning models
PRE_DELIVERY_FEATURES = [
    "innings", "overid", "cum_legal_before", "balls_remaining",
    "cum_wickets_before", "wickets_in_hand", "cum_runs_before",
    "runs_required", "rrr", "is_pressure_std"
]

# Chronological split date for holdout evaluation
TRAIN_DATE_CUTOFF = "2025-01-01"

# IPL Competition IDs
IPL_COMP_IDS = [1358, 1715, 2193, 2725, 3298]
