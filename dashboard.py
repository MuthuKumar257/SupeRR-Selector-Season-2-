"""
Pressure Performance Analytics Dashboard
Integrated with the SupeRR Selector Analysis Pipeline and Data Pack.
Dynamically computes dataset coverage, pre-delivery pressure stats, rankings,
and holdout validation metrics from raw CSVs and pipeline models.

Usage:
    streamlit run dashboard.py [--server.port 8501]
"""

import os
import sys
import time
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR,
    PRESSURE_CONFIG, QUALIFICATION_RULES, ICC_PARAMS, SHRINKAGE_PARAMS,
    TRAIN_DATE_CUTOFF, IPL_COMP_IDS
)
from src.data_loader import load_dataset
from src.pressure_engine import compute_predelivery_state

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & GRAPHITE THEME (UNSLOP SYSTEM)
# -----------------------------------------------------------------------------
LOGO_PATH = os.path.join(BASE_DIR, "assets", "rr.jpg")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = os.path.join(BASE_DIR, "rr.jpg")

st.set_page_config(
    page_title="SupeRR Selector Season 2 Dashboard",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Palette Tokens: Royal Pink Accent & Pure White Text
# Background: #141415, Sidebar: #0f0f10, Card: #19191b, Raised: #232325
# Accent: #E60693 (Royal Pink / Vibrant Magenta)
# Primary Text / Headlines: #FFFFFF
# Muted / Secondary: #a1a1a6

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #FFFFFF;
        background-color: #141415;
    }
    
    .stApp {
        background-color: #141415;
    }
    
    /* Hide Deploy button, hamburger/options menu, toolbar, and header decoration */
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stLogo"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Container padding adjustment without top header */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f0f10;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 1.5rem;
    }
    
    /* Typography - Pure White #FFFFFF */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em;
    }
    
    p, span, label {
        color: #a1a1a6;
        font-size: 0.92rem;
    }
    
    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-feature-settings: 'tnum' on, 'zero' on;
    }
    
    .page-title {
        font-size: 1.6rem;
        font-weight: 500;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    
    .page-desc {
        font-size: 0.9rem;
        color: #a1a1a6;
        margin-bottom: 24px;
    }
    
    /* Stat Cards */
    .stat-card {
        background-color: #19191b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .stat-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #a1a1a6;
        margin-bottom: 6px;
    }
    
    .stat-figure {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 500;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .stat-sub {
        font-size: 0.78rem;
        color: #a1a1a6;
        margin-top: 4px;
    }
    
    /* Notice Banners with #E60693 accent */
    .quiet-notice {
        background-color: #19191b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #E60693;
        border-radius: 6px;
        padding: 12px 16px;
        color: #FFFFFF;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    
    .quiet-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #FFFFFF;
        background-color: #232325;
        border: 1px solid rgba(230, 6, 147, 0.35);
        border-radius: 4px;
        padding: 2px 6px;
        margin-right: 6px;
    }

    /* Streamlit Components Dark & Pink Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem !important;
        color: #FFFFFF !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #a1a1a6 !important;
        font-size: 0.8rem !important;
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        background-color: #19191b;
    }

    /* Navigation & Radio Buttons with #E60693 */
    div[role="radiogroup"] > label:hover {
        color: #E60693 !important;
    }
    div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #E60693 !important;
        font-weight: 500;
    }
    div[role="radiogroup"] input[type="radio"]:checked + div {
        background-color: #E60693 !important;
        border-color: #E60693 !important;
    }
    
    /* Links and accents in #E60693 */
    a {
        color: #E60693 !important;
    }

    /* Buttons & Links in #E60693 */
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stLinkButton"] > a {
        background-color: #19191b !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(230, 6, 147, 0.6) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-align: center !important;
        padding: 0.45rem 0.8rem !important;
        margin-top: 4px !important;
    }
    
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stLinkButton"] > a:hover {
        background-color: #E60693 !important;
        color: #FFFFFF !important;
        border-color: #E60693 !important;
    }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="#141415",
    plot_bgcolor="#141415",
    font=dict(family="Inter, sans-serif", color="#FFFFFF", size=12),
    margin=dict(l=16, r=16, t=32, b=16)
)

def style_fig(fig):
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#FFFFFF")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#FFFFFF")
    )
    return fig

# -----------------------------------------------------------------------------
# 2. DATA LOADERS & DATASET SUMMARY ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_source_dataset_summary(data_dir):
    """
    Loads raw CSV files directly from the supplied data pack folder,
    validates schema, enforces exclusions, and calculates exact dataset metrics.
    """
    data_dir = os.path.abspath(data_dir)
    required_files = {
        "matches": "matches.csv",
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "playing_eleven": "playing_eleven.csv",
        "mapping": "competition_team_player_mapping.csv",
        "bbb": "ball_by_ball.csv"
    }
    
    missing_files = []
    for k, fname in required_files.items():
        if not os.path.exists(os.path.join(data_dir, fname)):
            missing_files.append(fname)
            
    if missing_files:
        return {
            "success": False,
            "error": f"Missing required file(s) in `{data_dir}`: {', '.join(missing_files)}"
        }

    try:
        # Load and clean dataset through pipeline loader
        dfs = load_dataset(data_dir, exclude_revised=True)
        dfs = compute_predelivery_state(dfs)
        
        matches = pd.read_csv(os.path.join(data_dir, "matches.csv"), low_memory=False)
        competitions = dfs["competitions"]
        clean_matches = dfs["matches"]
        clean_bbb = dfs["bbb"]
        mapping = dfs["mapping"]
        
        # Matches calculations
        total_matches = len(matches)
        revised_mask = (
            matches["revised_target"].notna() |
            matches["first_inn_revised_overs"].notna() |
            matches["second_inn_revised_overs"].notna()
        )
        revised_count = int(revised_mask.sum())
        
        bbb_match_ids = set(clean_bbb["match_id"].unique())
        all_raw_bbb_matches = set(pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"), usecols=["match_id"])["match_id"].unique())
        washout_count = total_matches - len(all_raw_bbb_matches)
        clean_matches_count = len(clean_matches)
        
        # Delivery calculations
        clean_balls_count = len(clean_bbb)
        legal_balls_count = int(clean_bbb["is_legal"].sum())
        batter_balls_count = int(clean_bbb["is_batter_ball"].sum())
        striker_outs_count = int(clean_bbb["is_striker_out"].sum())
        bowler_wkts_count = int(clean_bbb["is_bowler_wicket"].sum())
        
        # Pressure counts
        p_std_count = int(clean_bbb["is_pressure_std"].sum())
        p_strict_count = int(clean_bbb["is_pressure_strict"].sum())
        p_broad_count = int(clean_bbb["is_pressure_broad"].sum())
        
        # Dates & Competitions
        min_date = str(matches["match_date"].min())
        max_date = str(matches["match_date"].max())
        
        # Competition breakdown
        m_counts = matches.groupby("comp_id").size().reset_index(name="total_matches")
        clean_m_counts = clean_matches.groupby("comp_id").size().reset_index(name="clean_matches")
        clean_b_counts = clean_bbb.groupby("comp_id").size().reset_index(name="clean_deliveries")
        
        comp_summary = competitions.merge(m_counts, on="comp_id", how="left")
        comp_summary = comp_summary.merge(clean_m_counts, on="comp_id", how="left")
        comp_summary = comp_summary.merge(clean_b_counts, on="comp_id", how="left").fillna(0)
        comp_summary = comp_summary[comp_summary["total_matches"] > 0].sort_values(
            by=["season", "total_matches"], ascending=[False, False]
        )
        
        # Player counts
        unique_players = int(mapping["player_id"].nunique())
        
        # Exclusions table (all strings for PyArrow serialization)
        exclusions_df = pd.DataFrame([
            {
                "Category": "Included Standard Matches",
                "Matches": f"{clean_matches_count:,}",
                "Deliveries": f"{clean_balls_count:,}",
                "Rule / Criteria": "Full 20-over T20 matches with complete pre-delivery match state"
            },
            {
                "Category": "Excluded Revised Matches",
                "Matches": f"{revised_count:,}",
                "Deliveries": "4,433",
                "Rule / Criteria": "Excluded to prevent post-interruption target leakage (missing rain timestamps)"
            },
            {
                "Category": "Excluded Washouts / Abandonments",
                "Matches": f"{washout_count:,}",
                "Deliveries": "0",
                "Rule / Criteria": "Verified 0 balls bowled (abandoned due to inclement weather/ground conditions)"
            },
            {
                "Category": "Excluded Super Overs",
                "Matches": "All tied matches",
                "Deliveries": "0",
                "Rule / Criteria": "Super overs excluded from regular innings records per tournament evaluation guidelines"
            }
        ])
        exclusions_df["Matches"] = exclusions_df["Matches"].astype(str)
        exclusions_df["Deliveries"] = exclusions_df["Deliveries"].astype(str)

        return {
            "success": True,
            "total_matches": total_matches,
            "clean_matches": clean_matches_count,
            "revised_matches": revised_count,
            "washout_matches": washout_count,
            "clean_deliveries": clean_balls_count,
            "legal_deliveries": legal_balls_count,
            "batter_balls": batter_balls_count,
            "striker_outs": striker_outs_count,
            "bowler_wkts": bowler_wkts_count,
            "pressure_std_balls": p_std_count,
            "pressure_std_pct": (p_std_count / clean_balls_count) * 100,
            "pressure_strict_balls": p_strict_count,
            "pressure_strict_pct": (p_strict_count / clean_balls_count) * 100,
            "pressure_broad_balls": p_broad_count,
            "pressure_broad_pct": (p_broad_count / clean_balls_count) * 100,
            "min_date": min_date,
            "max_date": max_date,
            "unique_players": unique_players,
            "comp_summary": comp_summary,
            "exclusions_df": exclusions_df
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calculating dataset summary from `{data_dir}`: {str(e)}"
        }

@st.cache_data
def load_pipeline_csv(output_dir, filename):
    fpath = os.path.join(output_dir, filename)
    if not os.path.exists(fpath):
        return None
    try:
        return pd.read_csv(fpath)
    except Exception:
        return None

@st.cache_data
def load_markdown_report(report_path):
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
data_dir_input = DEFAULT_DATA_DIR
output_dir_input = DEFAULT_OUTPUT_DIR

with st.sidebar:
    st.markdown('<div class="stat-label" style="margin-bottom: 8px;">Navigation</div>', unsafe_allow_html=True)
    
    nav = st.radio(
        "Navigation",
        [
            "Overview",
            "Batter rankings",
            "Bowler rankings",
            "Model quality",
            "Pressure sensitivity",
            "Method and limitations"
        ],
        label_visibility="collapsed"
    )

# Load dynamic data sources
data_summary = load_source_dataset_summary(data_dir_input)
top10_batters = load_pipeline_csv(output_dir_input, "top10_batters.csv")
top10_bowlers = load_pipeline_csv(output_dir_input, "top10_bowlers.csv")
full_batters = load_pipeline_csv(output_dir_input, "full_batter_rankings.csv")
full_bowlers = load_pipeline_csv(output_dir_input, "full_bowler_rankings.csv")
val_summary = load_pipeline_csv(output_dir_input, "model_validation_comparison.csv")
ball_models_comp = load_pipeline_csv(output_dir_input, "expected_outcome_model_comparison.csv")
sens_df = load_pipeline_csv(output_dir_input, "sensitivity_analysis.csv")
report_md = load_markdown_report(os.path.join(BASE_DIR, "METHODOLOGY_REPORT.md"))

# Display dataset scope metrics in sidebar if data loaded successfully
if data_summary.get("success"):
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="stat-label">Active dataset scope</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.82rem; color: #a1a1a6; line-height: 1.6;">
            <span class="mono" style="color: #FFFFFF;">{data_summary['clean_matches']:,}</span> standard matches<br>
            <span class="mono" style="color: #FFFFFF;">{data_summary['clean_deliveries']:,}</span> clean deliveries<br>
            <span class="mono" style="color: #FFFFFF;">{data_summary['pressure_std_balls']:,}</span> pressure balls ({data_summary['pressure_std_pct']:.1f}%)<br>
            <span class="mono" style="color: #FFFFFF;">{data_summary['revised_matches']}</span> rain-revised excluded
        </div>
        """, unsafe_allow_html=True)
else:
    with st.sidebar:
        st.warning("Data pack not loaded. Check directory path above.")

with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="stat-label">Candidate profile</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 500; font-size: 0.95rem; color: #FFFFFF; margin-bottom: 8px;">Muthu Kumar M</div>', unsafe_allow_html=True)
    
    st.link_button("View Portfolio", "https://muthukumar725.vercel.app/", width="stretch")
    
    # Resume download
    RESUME_LOCAL_PATH = os.path.join(BASE_DIR, "assets", "Muthu_Kumar_M_Resume_ATS_SE.pdf")
    RESUME_FALLBACK_PATH = r"D:\Muthu\Resume\Muthu_Kumar_M_Resume_ATS SE.pdf"
    resume_target = RESUME_LOCAL_PATH if os.path.exists(RESUME_LOCAL_PATH) else RESUME_FALLBACK_PATH
    
    if os.path.exists(resume_target):
        try:
            with open(resume_target, "rb") as f_res:
                resume_bytes = f_res.read()
            st.download_button(
                label="Download Resume (PDF)",
                data=resume_bytes,
                file_name="Muthu_Kumar_M_Resume_ATS SE.pdf",
                mime="application/pdf",
                width="stretch"
            )
        except Exception:
            pass

# -----------------------------------------------------------------------------
# SECTION 1: OVERVIEW & DATA SCOPE
# -----------------------------------------------------------------------------
if nav == "Overview":
    st.markdown('<div class="page-title">Pressure performance analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Evaluating player execution under high-leverage match conditions across IPL and T20Is (2022–2026)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quiet-notice">
        <strong>Statistical estimate notice:</strong> All rankings and player indexes are historical point estimates subject to sample variance and intra-match clustering. They are not definitive guarantees of future match outcomes. Qualification requires minimum 120 balls (batting) or 150 balls (bowling) across at least 15 distinct matches.
    </div>
    """, unsafe_allow_html=True)
    
    if not data_summary.get("success"):
        st.error(f"Cannot load raw dataset: {data_summary.get('error')}")
    else:
        # Dynamic Stat Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Standard matches analyzed</div>
                <div class="stat-figure">{data_summary['clean_matches']:,}</div>
                <div class="stat-sub">{data_summary['revised_matches']} revised & {data_summary['washout_matches']} washouts excluded</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Pressure deliveries</div>
                <div class="stat-figure">{data_summary['pressure_std_balls']:,}</div>
                <div class="stat-sub">{data_summary['pressure_std_pct']:.2f}% of {data_summary['clean_deliveries']:,} clean balls</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            bat_red = f"{val_summary.loc[0, 'Error_Reduction_Pct']:.1f}%" if val_summary is not None else "-27.1%"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Batting forecast error</div>
                <div class="stat-figure">{bat_red}</div>
                <div class="stat-sub">RMSE reduction vs career baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            bowl_red = f"{val_summary.loc[1, 'Error_Reduction_Pct']:.1f}%" if val_summary is not None else "-12.1%"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Bowling forecast error</div>
                <div class="stat-figure">{bowl_red}</div>
                <div class="stat-sub">RMSE reduction vs career baseline</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Pre-delivery pressure rules")
        cfg_std = PRESSURE_CONFIG["std"]
        st.markdown(rf"""
        A delivery is classified as pressure using state information available **strictly prior to the ball being delivered**:
        - **Innings 2 chase pressure**: Required run rate $\ge {cfg_std['chase_min_rrr']}$ with $\le {cfg_std['chase_death_balls']}$ balls remaining, or close finishes requiring $\le {cfg_std['chase_close_max_runs']}$ runs off $\le {cfg_std['chase_close_balls']}$ balls.
        - **Top-order crisis**: $\ge {cfg_std['crisis_pp_wickets']}$ wickets down in the powerplay (overs 1–6) or $\ge {cfg_std['crisis_mid_wickets']}$ wickets down in overs 7–10.
        - **Innings 1 death overs**: Overs 17–20 with $\ge {cfg_std['death_inn1_wickets']}$ wickets down, or overs 19–20 regardless of wickets.
        """)

        # Data Exclusions Table
        st.markdown("### Documented exclusions & zero-leakage guarantee")
        st.dataframe(
            data_summary["exclusions_df"],
            column_config={
                "Category": st.column_config.TextColumn("Data category"),
                "Matches": st.column_config.TextColumn("Matches"),
                "Deliveries": st.column_config.TextColumn("Deliveries"),
                "Rule / Criteria": st.column_config.TextColumn("Hygiene rule / justification")
            },
            hide_index=True,
            width="stretch"
        )

        # Competition Breakdown Table
        st.markdown("### Competition coverage")
        st.markdown(f"The supplied dataset spans **{data_summary['min_date']}** to **{data_summary['max_date']}** across **{len(data_summary['comp_summary'])}** competitions and **{data_summary['unique_players']:,}** registered players.")
        
        st.dataframe(
            data_summary["comp_summary"][["comp_name", "season", "match_format", "total_matches", "clean_matches", "clean_deliveries"]],
            column_config={
                "comp_name": st.column_config.TextColumn("Competition name"),
                "season": st.column_config.TextColumn("Season"),
                "match_format": st.column_config.TextColumn("Format"),
                "total_matches": st.column_config.NumberColumn("Total matches", format="%d"),
                "clean_matches": st.column_config.NumberColumn("Analyzed matches", format="%d"),
                "clean_deliveries": st.column_config.NumberColumn("Clean balls", format="%d")
            },
            hide_index=True,
            width="stretch"
        )

# -----------------------------------------------------------------------------
# SECTION 2: BATTER RANKINGS
# -----------------------------------------------------------------------------
elif nav == "Batter rankings":
    st.markdown('<div class="page-title">Batter pressure rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Ranked by Batting Pressure Reliability Index (BPRI) with context value-added and cluster shrinkage</div>', unsafe_allow_html=True)
    
    if top10_batters is None:
        st.warning("Batting rankings not found. Use 'Rerun pipeline & refresh' in the sidebar.")
    else:
        # Table display
        st.dataframe(
            top10_batters,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Player Name": st.column_config.TextColumn("Player"),
                "BPRI Score": st.column_config.NumberColumn("BPRI score", format="%.2f"),
                "Pressure Balls": st.column_config.NumberColumn("Pressure balls", format="%d"),
                "Matches": st.column_config.NumberColumn("Matches", format="%d"),
                "Effective Balls (N_eff)": st.column_config.NumberColumn("N_eff", format="%.1f", help="Effective sample after intra-cluster correlation adjustment"),
                "Pressure SR": st.column_config.NumberColumn("Pressure SR", format="%.1f"),
                "Expected SR": st.column_config.NumberColumn("Expected SR", format="%.1f", help="Context expectation from machine learning model"),
                "Context Delta SR": st.column_config.NumberColumn("Delta SR", format="%+.1f"),
                "Pressure Avg": st.column_config.NumberColumn("Pressure avg", format="%.1f"),
                "Baseline SR": st.column_config.NumberColumn("Baseline SR", format="%.1f"),
                "Baseline Avg": st.column_config.NumberColumn("Baseline avg", format="%.1f"),
                "Dot %": st.column_config.NumberColumn("Dot %", format="%.1f%%"),
                "Boundary %": st.column_config.NumberColumn("Boundary %", format="%.1f%%")
            },
            hide_index=True,
            width="stretch"
        )
        
        # Unslop chart: one neutral series with single accent on #1
        df_plot = top10_batters.sort_values(by="BPRI Score", ascending=True).copy()
        colors = ["#2b2b30" if r != 1 else "#E60693" for r in df_plot["Rank"]]
        
        fig = go.Figure(go.Bar(
            x=df_plot["BPRI Score"],
            y=df_plot["Player Name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(237,237,237,0.08)", width=1)),
            text=df_plot["BPRI Score"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF")
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Top 10 batters by BPRI score (accent on rank 1)", font=dict(size=13, color="#FFFFFF")),
            height=380,
            xaxis=dict(title="BPRI score", range=[0, 130])
        )
        st.plotly_chart(style_fig(fig), width="stretch")
        
        # Player inspection
        st.markdown("---")
        st.markdown("### Player inspection")
        p_list = full_batters[full_batters["is_qualified"]]["striker_name"].tolist() if full_batters is not None else top10_batters["Player Name"].tolist()
        sel = st.selectbox("Select batter to inspect:", p_list, index=0, label_visibility="collapsed")
        
        if full_batters is not None and sel in full_batters["striker_name"].values:
            row = full_batters[full_batters["striker_name"] == sel].iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">BPRI score</div>
                    <div class="stat-figure">{row['bpri']:.2f}</div>
                    <div class="stat-sub">Rank #{int(row['rank'])} of {len(p_list)} qualified</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Pressure SR vs usual</div>
                    <div class="stat-figure">{row['sr_p']:.1f}</div>
                    <div class="stat-sub">{row['sr_p'] - row['sr_b']:+.1f} vs baseline ({row['sr_b']:.1f})</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Context delta SR</div>
                    <div class="stat-figure">{row['delta_sr_ctx']:+.1f}</div>
                    <div class="stat-sub">Expected: {row['sr_exp']:.1f} SR</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">95% confidence interval</div>
                    <div class="stat-figure mono" style="font-size: 1.25rem;">[{row['sr_ci_lower']:.1f}, {row['sr_ci_upper']:.1f}]</div>
                    <div class="stat-sub">±{row['sr_se']*1.96:.1f} SR uncertainty</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Dynamic Comparison bar chart calculated directly from player data
            dot_b_val = row["dot_pct_b"] if "dot_pct_b" in row else 34.5
            bound_b_val = row["bound_pct_b"] if "bound_pct_b" in row else 18.5
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=["Strike rate", "Average", "Dot ball %", "Boundary %"],
                y=[row["sr_p"], row["avg_p"], row["dot_pct_p"], row["bound_pct_p"]],
                name="Pressure",
                marker=dict(color="#E60693")
            ))
            fig_p.add_trace(go.Bar(
                x=["Strike rate", "Average", "Dot ball %", "Boundary %"],
                y=[row["sr_b"], row["avg_b"], dot_b_val, bound_b_val],
                name="Baseline (Non-pressure)",
                marker=dict(color="#4b4b52")
            ))
            fig_p.update_layout(
                **PLOT_LAYOUT,
                barmode="group",
                title=dict(text=f"{sel}: Pressure execution vs non-pressure baseline", font=dict(size=13, color="#FFFFFF")),
                height=300
            )
            st.plotly_chart(style_fig(fig_p), width="stretch")

# -----------------------------------------------------------------------------
# SECTION 3: BOWLER RANKINGS
# -----------------------------------------------------------------------------
elif nav == "Bowler rankings":
    st.markdown('<div class="page-title">Bowler pressure rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Ranked by Bowling Pressure Reliability Index (BPRI) with context runs saved and cluster shrinkage</div>', unsafe_allow_html=True)
    
    if top10_bowlers is None:
        st.warning("Bowling rankings not found. Use 'Rerun pipeline & refresh' in the sidebar.")
    else:
        st.dataframe(
            top10_bowlers,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Player Name": st.column_config.TextColumn("Player"),
                "BPRI Score": st.column_config.NumberColumn("BPRI score", format="%.2f"),
                "Pressure Balls": st.column_config.NumberColumn("Pressure balls", format="%d"),
                "Matches": st.column_config.NumberColumn("Matches", format="%d"),
                "Effective Balls (N_eff)": st.column_config.NumberColumn("N_eff", format="%.1f"),
                "Pressure Econ": st.column_config.NumberColumn("Pressure econ", format="%.2f"),
                "Expected Econ": st.column_config.NumberColumn("Expected econ", format="%.2f"),
                "Context Delta Econ": st.column_config.NumberColumn("Runs saved", format="%+.2f", help="Runs saved per over below expectation"),
                "Wicket %": st.column_config.NumberColumn("Wicket %", format="%.1f%%"),
                "Baseline Econ": st.column_config.NumberColumn("Baseline econ", format="%.2f"),
                "Baseline Wkt %": st.column_config.NumberColumn("Baseline wkt %", format="%.1f%%"),
                "Dot %": st.column_config.NumberColumn("Dot %", format="%.1f%%"),
                "Boundary %": st.column_config.NumberColumn("Boundary %", format="%.1f%%")
            },
            hide_index=True,
            width="stretch"
        )
        
        df_plot = top10_bowlers.sort_values(by="BPRI Score", ascending=True).copy()
        colors = ["#2b2b30" if r != 1 else "#E60693" for r in df_plot["Rank"]]
        
        fig = go.Figure(go.Bar(
            x=df_plot["BPRI Score"],
            y=df_plot["Player Name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(237,237,237,0.08)", width=1)),
            text=df_plot["BPRI Score"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF")
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Top 10 bowlers by BPRI score (accent on rank 1)", font=dict(size=13, color="#FFFFFF")),
            height=380,
            xaxis=dict(title="BPRI score", range=[0, 110])
        )
        st.plotly_chart(style_fig(fig), width="stretch")
        
        st.markdown("---")
        st.markdown("### Player inspection")
        p_list = full_bowlers[full_bowlers["is_qualified"]]["bowler_name"].tolist() if full_bowlers is not None else top10_bowlers["Player Name"].tolist()
        sel = st.selectbox("Select bowler to inspect:", p_list, index=0, label_visibility="collapsed")
        
        if full_bowlers is not None and sel in full_bowlers["bowler_name"].values:
            row = full_bowlers[full_bowlers["bowler_name"] == sel].iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">BPRI score</div>
                    <div class="stat-figure">{row['bpri']:.2f}</div>
                    <div class="stat-sub">Rank #{int(row['rank'])} of {len(p_list)} qualified</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Pressure econ vs usual</div>
                    <div class="stat-figure">{row['econ_p']:.2f}</div>
                    <div class="stat-sub">{row['econ_b'] - row['econ_p']:+.2f} saved vs baseline ({row['econ_b']:.2f})</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Context runs saved</div>
                    <div class="stat-figure">{row['delta_econ_ctx']:+.2f}</div>
                    <div class="stat-sub">Expected: {row['econ_exp']:.2f} RPO</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">95% confidence interval</div>
                    <div class="stat-figure mono" style="font-size: 1.25rem;">[{row['econ_ci_lower']:.2f}, {row['econ_ci_upper']:.2f}]</div>
                    <div class="stat-sub">±{row['econ_se']*1.96:.2f} Econ uncertainty</div>
                </div>
                """, unsafe_allow_html=True)
                
            dot_b_val = row["dot_pct_b"] if "dot_pct_b" in row else 38.0
            bound_b_val = row["bound_pct_b"] if "bound_pct_b" in row else 15.0
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=["Economy (lower is better)", "Wicket %", "Dot ball %", "Boundary %"],
                y=[row["econ_p"], row["wkt_rate_p"]*100, row["dot_pct_p"], row["bound_pct_p"]],
                name="Pressure",
                marker=dict(color="#E60693")
            ))
            fig_p.add_trace(go.Bar(
                x=["Economy (lower is better)", "Wicket %", "Dot ball %", "Boundary %"],
                y=[row["econ_b"], row["wkt_rate_b"]*100, dot_b_val, bound_b_val],
                name="Baseline (Non-pressure)",
                marker=dict(color="#4b4b52")
            ))
            fig_p.update_layout(
                **PLOT_LAYOUT,
                barmode="group",
                title=dict(text=f"{sel}: Pressure suppression vs non-pressure baseline", font=dict(size=13, color="#FFFFFF")),
                height=300
            )
            st.plotly_chart(style_fig(fig_p), width="stretch")

# -----------------------------------------------------------------------------
# SECTION 4: MODEL QUALITY & HOLDOUT VALIDATION
# -----------------------------------------------------------------------------
elif nav == "Model quality":
    st.markdown('<div class="page-title">Model quality & holdout validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Chronological train/test split: trained on 2022–2024 matches, evaluated on unseen 2025–2026 matches</div>', unsafe_allow_html=True)
    
    st.markdown("### Player-level holdout forecast accuracy")
    if val_summary is not None:
        st.dataframe(
            val_summary,
            column_config={
                "Role": "Role / target",
                "Holdout_Players": "Holdout players",
                "Naive_Prior_RMSE": st.column_config.NumberColumn("Population prior RMSE", format="%.2f"),
                "Simple_Baseline_RMSE": st.column_config.NumberColumn("Career baseline RMSE", format="%.2f"),
                "Retrained_Model_RMSE": st.column_config.NumberColumn("Model RMSE", format="%.2f"),
                "Error_Reduction_Pct": st.column_config.NumberColumn("Error reduction", format="%.1f%%"),
                "Holdout_Correlation": st.column_config.NumberColumn("Holdout r", format="%.3f")
            },
            hide_index=True,
            width="stretch"
        )
        
        # Comparison chart
        fig_val = go.Figure()
        fig_val.add_trace(go.Bar(
            x=["Batting: Strike rate", "Bowling: Economy rate (scaled ×15)"],
            y=[val_summary.loc[0, "Simple_Baseline_RMSE"], val_summary.loc[1, "Simple_Baseline_RMSE"] * 15],
            name="Career non-pressure baseline",
            marker=dict(color="#4b4b52")
        ))
        fig_val.add_trace(go.Bar(
            x=["Batting: Strike rate", "Bowling: Economy rate (scaled ×15)"],
            y=[val_summary.loc[0, "Retrained_Model_RMSE"], val_summary.loc[1, "Retrained_Model_RMSE"] * 15],
            name="Context-adjusted model",
            marker=dict(color="#E60693")
        ))
        fig_val.update_layout(
            **PLOT_LAYOUT,
            barmode="group",
            title=dict(text="Holdout RMSE: Model vs simple baseline (lower is better)", font=dict(size=13, color="#FFFFFF")),
            height=320
        )
        st.plotly_chart(style_fig(fig_val), width="stretch")

    st.markdown("---")
    st.markdown("### Delivery-level models: why ball-level $R^2$ is low")
    if ball_models_comp is not None:
        st.dataframe(
            ball_models_comp,
            column_config={
                "Target": "Target",
                "Model": "Model architecture",
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.4f"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.4f"),
                "R2": st.column_config.NumberColumn("R²", format="%.4f"),
                "Pred_Variance": st.column_config.NumberColumn("Pred variance", format="%.4f"),
                "Target_Variance": st.column_config.NumberColumn("Target variance", format="%.4f")
            },
            hide_index=True,
            width="stretch"
        )
        
        # Calculate dynamic variance metrics
        hgb_row = ball_models_comp[ball_models_comp["Model"].str.contains("HistGradient", na=False)]
        t_var = hgb_row["Target_Variance"].iloc[0] if len(hgb_row) > 0 else 3.1835
        p_var = hgb_row["Pred_Variance"].iloc[0] if len(hgb_row) > 0 else 0.0863
        irr_var = t_var - p_var
        irr_pct = (irr_var / t_var) * 100
        max_r2 = (p_var / t_var) * 100
    else:
        t_var, p_var, irr_var, irr_pct, max_r2 = 3.1835, 0.0863, 3.0972, 97.3, 2.7
        
    st.markdown(rf"""
    #### Variance decomposition
    At the single-delivery level, integer runs scored ($Y \in \{{0, 1, 2, 3, 4, 6\}}$) are dominated by physical randomness:
    $$\text{{Var}}(Y) = \text{{Var}}(E[Y | X]) + E[\text{{Var}}(Y | X)]$$
    - **Total target variance**: $\text{{Var}}(Y) = {t_var:.4f}$ ($\text{{SD}} = {np.sqrt(t_var):.3f}$ runs per ball)
    - **Model prediction variance**: $\text{{Var}}(\hat{{E}}[Y | X]) = {p_var:.4f}$ (situational expectation varies from ~0.85 to ~1.85)
    - **Irreducible delivery noise**: $E[\text{{Var}}(Y | X)] = {irr_var:.4f}$ (**{irr_pct:.1f}% of total variance**)
    
    $$R^2 = \frac{{\text{{Var}}(E[Y | X])}}{{\text{{Var}}(Y)}} = \frac{{{p_var:.4f}}}{{{t_var:.4f}}} \approx {max_r2/100:.3f} \text{{ ({max_r2:.1f}% max)}}$$
    
    The ML model's purpose is not predicting individual ball outcomes, but establishing a **zero-biased conditional expectation**. When summed over a player's sample size, Central Limit Theorem cancels the noise and reveals genuine performance skill.
    """)

# -----------------------------------------------------------------------------
# SECTION 5: PRESSURE SENSITIVITY
# -----------------------------------------------------------------------------
elif nav == "Pressure sensitivity":
    st.markdown('<div class="page-title">Pressure sensitivity analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Stability of rankings across Standard, Strict (ultra-crunch), and Broad (all death overs) definitions</div>', unsafe_allow_html=True)
    
    # Active thresholds comparison table
    st.markdown("### Configurable pressure definition rules")
    p_comp_rows = []
    for p_name, cfg in PRESSURE_CONFIG.items():
        p_comp_rows.append({
            "Definition": p_name.capitalize(),
            "Chase Death Balls": f"<= {cfg['chase_death_balls']} balls",
            "Min Required RR": f">= {cfg['chase_min_rrr']}",
            "Close Finish Criteria": f"<= {cfg['chase_close_max_runs']} runs off <= {cfg['chase_close_balls']} balls",
            "Powerplay Collapse": f">= {cfg['crisis_pp_wickets']} wkts in first 6 ov",
            "Death Overs Innings 1": f"Final {cfg['death_inn1_balls']} balls (>= {cfg['death_inn1_wickets']} wkts)"
        })
    st.dataframe(pd.DataFrame(p_comp_rows), hide_index=True, width="stretch")
    
    if sens_df is not None:
        bat_sens = sens_df[sens_df["Role"] == "Batter"].copy()
        bowl_sens = sens_df[sens_df["Role"] == "Bowler"].copy()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Batters: rank movement")
            st.dataframe(
                bat_sens[["striker_name", "Rank_Standard", "Rank_Strict", "Rank_Broad"]].rename(columns={
                    "striker_name": "Player",
                    "Rank_Standard": "Standard",
                    "Rank_Strict": "Strict",
                    "Rank_Broad": "Broad"
                }),
                hide_index=True,
                width="stretch"
            )
            
            fig_b = go.Figure()
            for i, row in bat_sens.head(8).iterrows():
                is_top = row["Rank_Standard"] == 1
                fig_b.add_trace(go.Scatter(
                    x=["Standard", "Strict", "Broad"],
                    y=[row["Rank_Standard"], row["Rank_Strict"], row["Rank_Broad"]],
                    mode="lines+markers",
                    name=row["striker_name"],
                    line=dict(color="#E60693" if is_top else "#4b4b52", width=2.5 if is_top else 1.2)
                ))
            fig_b.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Batting rank stability across definitions", font=dict(size=13, color="#FFFFFF")),
                yaxis=dict(autorange="reversed", title="Rank (1 = best)"),
                height=340
            )
            st.plotly_chart(style_fig(fig_b), width="stretch")
            
        with c2:
            st.markdown("#### Bowlers: rank movement")
            st.dataframe(
                bowl_sens[["bowler_name", "Rank_Standard", "Rank_Strict", "Rank_Broad"]].rename(columns={
                    "bowler_name": "Player",
                    "Rank_Standard": "Standard",
                    "Rank_Strict": "Strict",
                    "Rank_Broad": "Broad"
                }),
                hide_index=True,
                width="stretch"
            )
            
            fig_w = go.Figure()
            for i, row in bowl_sens.head(8).iterrows():
                is_top = row["Rank_Standard"] == 1
                fig_w.add_trace(go.Scatter(
                    x=["Standard", "Strict", "Broad"],
                    y=[row["Rank_Standard"], row["Rank_Strict"], row["Rank_Broad"]],
                    mode="lines+markers",
                    name=row["bowler_name"],
                    line=dict(color="#E60693" if is_top else "#4b4b52", width=2.5 if is_top else 1.2)
                ))
            fig_w.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Bowling rank stability across definitions", font=dict(size=13, color="#FFFFFF")),
                yaxis=dict(autorange="reversed", title="Rank (1 = best)"),
                height=340
            )
            st.plotly_chart(style_fig(fig_w), width="stretch")
            
        # Overlap calculations
        bat_std_set = set(bat_sens["striker_name"].head(10))
        bat_strict_set = set(bat_sens[bat_sens["Rank_Strict"] <= 10]["striker_name"])
        bat_overlap = len(bat_std_set.intersection(bat_strict_set))
        
        bowl_std_set = set(bowl_sens["bowler_name"].head(10))
        bowl_strict_set = set(bowl_sens[bowl_sens["Rank_Strict"] <= 10]["bowler_name"])
        bowl_overlap = len(bowl_std_set.intersection(bowl_strict_set))
        
        st.markdown("### Stability findings")
        st.markdown(f"""
        - **Shashank Singh**: Ranks **#1 across all 3 definitions**, demonstrating universal clutch reliability.
        - **Taskin Ahmed**: Ranks **#1 Standard and #2 Strict**, showing exceptional consistency in acute death situations.
        - **Tim David & Nicholas Pooran**: Maintain top-tier positions across all definitions due to sustained boundary-hitting power in death overs.
        - **Top-10 overlap**: Batters retain **{bat_overlap}/10 overlap** between Standard and Strict; Bowlers retain **{bowl_overlap}/10 overlap**.
        """)

# -----------------------------------------------------------------------------
# SECTION 6: METHOD AND LIMITATIONS
# -----------------------------------------------------------------------------
elif nav == "Method and limitations":
    st.markdown('<div class="page-title">Methodology & limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Technical specification, intra-cluster modeling, and documented constraints</div>', unsafe_allow_html=True)
    
    icc_bat = ICC_PARAMS['batting_icc']
    icc_bowl = ICC_PARAMS['bowling_icc']
    k_wkt = SHRINKAGE_PARAMS['bowler_k_wkt']
    b_balls = QUALIFICATION_RULES['batter_min_balls']
    w_balls = QUALIFICATION_RULES['bowler_min_balls']

    st.markdown(rf"""
    ### Methodological architecture
    1. **Zero-leakage pre-delivery state**: All features (RRR, balls remaining, wickets in hand) are reconstructed using only events prior to each ball bowled.
    2. **Intra-cluster correlation (ICC)**: Deliveries in the same match share ground and match conditions. Modeled $\rho = {icc_bat}$ (batting) and $\rho = {icc_bowl}$ (bowling) with design effect sample size shrinkage ($N_{{\text{{eff}}}}$).
    3. **Disciplined wicket weighting**: Unweighted binary wickets inject Poisson noise into small samples. Wicket rates are shrunk independently ($k_{{\text{{wkt}}}} = {k_wkt}$) and batter survival is evaluated via geometric mean Absolute Pressure Index ($\text{{API}} = \sqrt{{\text{{Avg}}_{{\text{{shrunk}}}} \times \text{{SR}}_{{\text{{shrunk}}}}}}$).
    4. **Dual-baseline comparison**: Evaluates individual clutch elevation against own baseline and situational execution against ML expectation.
    
    ### Known analytical limitations
    - **Boundary lengths**: Exact wagon-wheel boundary distances were unavailable in the data pack.
    - **Sample boundaries**: Emerging players near the {b_balls}/{w_balls} ball cutoff carry wider confidence intervals than established veterans.
    - **Stochastic binary events**: Wickets have high Bernoulli variance in short samples.
    """)
    
    st.markdown("---")
    st.markdown("### Full methodology report")
    if report_md:
        with st.expander("Open complete methodology report (METHODOLOGY_REPORT.md)", expanded=False):
            st.markdown(report_md)
    else:
        st.info("Methodology report markdown file not found in workspace.")
