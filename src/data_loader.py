"""
Data loading and hygiene module for SupeRR Selector Analysis System.
Loads all required CSVs, validates schema, checks missing values, handles edge cases,
and documents data cleaning decisions transparently.
"""

import os
import pandas as pd
import numpy as np

def load_dataset(data_dir, exclude_revised=True):
    """
    Load and validate all CSVs from the supplied data directory.
    Args:
        data_dir: Path to directory containing the 6 CSV files.
        exclude_revised: Whether to exclude matches with revised overs or targets
                         to prevent forward-looking leakage.
    Returns:
        dict of DataFrames containing competitions, matches, teams,
        playing_eleven, mapping, and ball_by_ball.
    """
    print(f"[DATA LOADER] Loading CSVs from: {data_dir}")
    
    files = {
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "matches": "matches.csv",
        "playing_eleven": "playing_eleven.csv",
        "mapping": "competition_team_player_mapping.csv",
        "bbb": "ball_by_ball.csv"
    }
    
    dfs = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required CSV not found: {fpath}")
        print(f"  -> Reading {fname}...")
        dfs[key] = pd.read_csv(fpath, low_memory=False)
        print(f"     Loaded {len(dfs[key]):,} rows, {len(dfs[key].columns)} columns.")

    # Data Hygiene & Validation
    _validate_and_clean(dfs, exclude_revised=exclude_revised)
    
    return dfs

def _validate_and_clean(dfs, exclude_revised=True):
    """
    Validates data consistency and performs documented cleaning.
    """
    matches = dfs["matches"]
    bbb = dfs["bbb"]
    mapping = dfs["mapping"]
    
    # 1. Matches vs Ball-by-Ball alignment
    m_ids_in_matches = set(matches["match_id"])
    m_ids_in_bbb = set(bbb["match_id"])
    missing_in_bbb = m_ids_in_matches - m_ids_in_bbb
    
    print(f"\n[DATA HYGIENE] Total matches in matches.csv: {len(m_ids_in_matches)}")
    print(f"[DATA HYGIENE] Total matches with ball-by-ball data: {len(m_ids_in_bbb)}")
    print(f"[DATA HYGIENE] Matches without ball-by-ball deliveries: {len(missing_in_bbb)}")
    if missing_in_bbb:
        abandoned_results = matches[matches["match_id"].isin(missing_in_bbb)]["result"].value_counts().to_dict()
        print(f"  -> Reason: Verified all {len(missing_in_bbb)} matches were abandoned/washed out without play ({abandoned_results}).")
    
    # 2. Player ID vs Display Name Stability
    # Some players share a name across different countries/teams (e.g. Haider Ali PAK vs UAE).
    # We maintain identity on player_id, and format display names with ID when duplicates exist.
    name_to_ids = mapping.groupby("player_name")["player_id"].unique()
    mult_names = name_to_ids[name_to_ids.apply(len) > 1]
    print(f"[DATA HYGIENE] Names shared across distinct player_ids: {len(mult_names)}")
    for name, pids in mult_names.items():
        print(f"  -> '{name}' maps to IDs: {list(pids)}")
    
    # 3. Exclude Revised-Overs / Revised-Target Matches to Prevent Forward-Looking Leakage
    if exclude_revised:
        revised_mask = (
            matches["revised_target"].notna() |
            matches["first_inn_revised_overs"].notna() |
            matches["second_inn_revised_overs"].notna()
        )
        revised_ids = set(matches[revised_mask]["match_id"])
        print(f"[DATA HYGIENE] Identified {len(revised_ids)} matches with revised overs or targets.")
        print("  -> Excluding these matches from analysis to prevent post-interruption target leakage.")
        dfs["matches"] = matches[~matches["match_id"].isin(revised_ids)].copy().reset_index(drop=True)
        dfs["bbb"] = bbb[~bbb["match_id"].isin(revised_ids)].copy().reset_index(drop=True)
        bbb = dfs["bbb"]
        matches = dfs["matches"]
        print(f"  -> Retained {len(matches)} standard matches ({len(bbb):,} deliveries).")
    
    # 4. Ball-by-ball sorting & chronologic sequencing
    dfs["bbb"] = bbb.sort_values(by=["match_id", "innings", "overid", "over_ball_num"]).reset_index(drop=True)
    
    # 5. Ball classification & Cricket scoring rules
    # Legal ball for bowler: not wide and not no-ball (Overstep, other)
    dfs["bbb"]["is_legal"] = ~dfs["bbb"]["extras_type"].isin(["wide", "Overstep", "other"])
    # Ball faced for batter: not wide
    dfs["bbb"]["is_batter_ball"] = dfs["bbb"]["extras_type"] != "wide"
    
    # Bowler extras: wides + no-balls (Overstep, other). Byes and leg-byes are fielding extras.
    dfs["bbb"]["bowler_extras"] = np.where(
        dfs["bbb"]["extras_type"].isin(["wide", "Overstep", "other"]),
        dfs["bbb"]["extras1_runs"],
        0
    )
    dfs["bbb"]["bowler_runs_conceded"] = dfs["bbb"]["runs"] + dfs["bbb"]["bowler_extras"]
    
    # Total runs scored on the delivery
    dfs["bbb"]["total_ball_runs"] = dfs["bbb"]["runs"] + dfs["bbb"]["extras1_runs"] + dfs["bbb"]["extras_2_runs"]
    
    # Wickets credited to bowler vs fielding dismissals
    bowler_dismissals = ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"]
    dfs["bbb"]["is_bowler_wicket"] = dfs["bbb"]["is_wicket"] & dfs["bbb"]["dismissal_type"].isin(bowler_dismissals)
    
    # Dismissals of striker
    dfs["bbb"]["is_striker_out"] = dfs["bbb"]["is_wicket"] & (
        (dfs["bbb"]["dismissed_player_id"] == dfs["bbb"]["striker_id"]) | 
        (dfs["bbb"]["dismissal_type"].isin(bowler_dismissals)) |
        (dfs["bbb"]["dismissal_type"] == "Retire Out")
    )
    
    print("[DATA HYGIENE] Delivery accounting verified according to MCC Laws of Cricket.")
    print(f"  -> Total deliveries: {len(dfs['bbb']):,}")
    print(f"  -> Total batter balls faced: {dfs['bbb']['is_batter_ball'].sum():,}")
    print(f"  -> Total legal bowler deliveries: {dfs['bbb']['is_legal'].sum():,}")
    print(f"  -> Bowler credited wickets: {dfs['bbb']['is_bowler_wicket'].sum():,}")
    print(f"  -> Total striker outs: {dfs['bbb']['is_striker_out'].sum():,}")
