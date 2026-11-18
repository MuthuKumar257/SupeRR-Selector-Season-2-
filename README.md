# SupeRR Selector (Season 2 — Round 1)
## Player Reliability Under Pressure Analytics System (IPL & T20Is: 2022–2026)

**Candidate / Author**: Muthu Kumar M  
**Portfolio**: [https://muthukumar725.vercel.app/](https://muthukumar725.vercel.app/)  
**Event**: Rajasthan Royals SupeRR Selector Season 2 — Round 1 Challenge  
**Dataset Scope**: Complete IPL & T20 International (T20I) Ball-by-Ball Records (2022–2026)  
**Live Interactive Dashboard**: Runs on Streamlit (`streamlit run dashboard.py`) and deployable directly on Render / Streamlit Community Cloud.

---

## 1. Executive Summary

In high-stakes Twenty20 cricket, match outcomes are overwhelmingly decided in acute pressure moments: chasing 45 runs off the final 3 overs, stabilizing after a 25/3 top-order collapse, or defending 10 runs in the 20th over. This repository provides a **fully reproducible, leak-free, context-adjusted analytics engine** and **interactive dashboard** that identifies the **10 batters** and **10 bowlers** who perform most reliably under pressure across **849 standard matches** (196,566 clean deliveries) from 2022 to 2026.

All rankings are derived strictly from empirical ball-by-ball execution data, independent of player reputation or subjective narratives.

---

## 2. Official Ranked Lists

### Top 10 Batters Under Pressure (2022–2026)

| Rank | Player Name | BPRI Score | Pressure Balls | Matches | Effective Balls ($N_{\text{eff}}$) | Pressure SR | Expected SR | Delta SR | Pressure Avg | Baseline SR | Dot % | Boundary % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Shashank Singh** | **113.34** | 274 | 30 | 230.3 | **193.1** | 162.7 | **+30.3** | 58.8 | 133.5 | 24.8% | 26.6% |
| **2** | **Nicholas Pooran** | **94.82** | 422 | 48 | 357.2 | **183.2** | 156.1 | **+27.1** | 42.9 | 148.3 | 32.7% | 27.5% |
| **3** | **Rinku Singh** | **87.32** | 593 | 66 | 500.0 | **176.2** | 155.3 | **+21.0** | 38.7 | 130.0 | 33.6% | 26.6% |
| **4** | **Tim David** | **87.30** | 526 | 68 | 454.7 | **196.6** | 162.1 | **+34.5** | 32.3 | 160.8 | 31.6% | 29.1% |
| **5** | **Tristan Stubbs** | **86.10** | 451 | 49 | 378.6 | **168.3** | 153.7 | **+14.6** | 42.2 | 132.5 | 25.9% | 22.6% |
| **6** | **Ashutosh Sharma** | **84.95** | 219 | 24 | 184.1 | **190.4** | 157.6 | **+32.8** | 29.8 | 125.0 | 28.3% | 28.3% |
| **7** | **Donovan Ferreira** | **84.60** | 190 | 21 | 160.0 | **187.4** | 147.4 | **+40.0** | 29.7 | 127.5 | 32.6% | 27.4% |
| **8** | **Romario Shepherd** | **83.22** | 330 | 51 | 292.7 | **183.3** | 156.8 | **+26.5** | 30.2 | 127.6 | 32.1% | 26.4% |
| **9** | **Shubman Gill** | **80.81** | 128 | 15 | 108.9 | **169.5** | 161.3 | **+8.3** | 54.2 | 148.9 | 19.5% | 19.5% |
| **10** | **Naman Dhir** | **80.75** | 155 | 23 | 136.7 | **201.9** | 167.0 | **+34.9** | 28.5 | 153.8 | 27.1% | 32.9% |

### Top 10 Bowlers Under Pressure (2022–2026)

| Rank | Player Name | BPRI Score | Pressure Balls | Matches | Effective Balls ($N_{\text{eff}}$) | Pressure Econ | Expected Econ | Runs Saved | Wicket % | Baseline Econ | Dot % | Boundary % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Taskin Ahmed** | **92.46** | 236 | 31 | 204.4 | **7.37** | 8.21 | **+0.84** | 10.6% | 7.79 | 44.5% | 14.0% |
| **2** | **Matheesha Pathirana** | **91.62** | 397 | 41 | 330.0 | **7.75** | 9.51 | **+1.75** | 10.3% | 7.67 | 37.8% | 16.1% |
| **3** | **Mahedi Hasan** | **90.51** | 164 | 22 | 142.5 | **5.71** | 6.47 | **+0.77** | 6.7% | 7.15 | 49.4% | 9.8% |
| **4** | **Varun Chakravarthy** | **90.14** | 353 | 52 | 310.9 | **6.44** | 7.72 | **+1.28** | 5.9% | 8.08 | 47.0% | 11.3% |
| **5** | **Sunil Narine** | **89.47** | 231 | 30 | 199.7 | **6.00** | 8.02 | **+2.02** | 5.6% | 6.89 | 38.1% | 8.2% |
| **6** | **Mitchell Santner** | **89.21** | 227 | 35 | 201.2 | **6.11** | 7.45 | **+1.35** | 6.2% | 7.36 | 41.0% | 10.1% |
| **7** | **Rashid Khan** | **87.78** | 502 | 73 | 441.3 | **7.34** | 8.17 | **+0.83** | 8.4% | 7.29 | 38.2% | 12.2% |
| **8** | **Rasikh Salam** | **87.37** | 195 | 18 | 158.5 | **8.15** | 9.03 | **+0.88** | 10.8% | 9.57 | 36.9% | 17.4% |
| **9** | **Jasprit Bumrah** | **85.89** | 460 | 54 | 391.2 | **7.28** | 9.34 | **+2.07** | 7.0% | 6.62 | 38.3% | 14.1% |
| **10** | **Haris Rauf** | **83.19** | 343 | 35 | 284.4 | **8.12** | 9.13 | **+1.01** | 8.5% | 8.53 | 42.6% | 18.7% |

---

## 3. Methodological Pillars

1. **Zero-Leakage Pre-Delivery Context**:
   - Every delivery is evaluated using only the match state known **strictly before the ball was bowled** (innings, balls remaining, wickets in hand, cumulative runs, required run rate).
   - **34 rain-revised matches are strictly excluded** to eliminate target leakage caused by unrecorded rain delays.
2. **Context-Expected Benchmark (`HistGradientBoosting`)**:
   - Machine learning models trained on 2022–2024 matches establish situational expectations ($E[Y \mid X]$).
   - Explains why ball-level $R^2 \approx 0.027$: **97.3% of single-ball variance is irreducible physical noise**. The model’s value is establishing zero-biased conditional expectations.
3. **Clustering & Design Effects ($DEFF$)**:
   - Intra-Cluster Correlation ($\rho = 0.0233$ for batting, $\rho = 0.0234$ for bowling) accounts for correlated conditions within the same match. Effective sample size is shrunken via:
     $$N_{\text{eff}} = \frac{N}{1 + (\bar{m} - 1)\rho}$$
4. **Empirical Bayes Shrinkage**:
   - Dual-baseline differentials are stabilized using credibility weighting ($k_{\text{balls}} = 60$, $k_{\text{outs}} = 100$, $k_{\text{econ}} = 75$, $k_{\text{wkt}} = 120$).
5. **Out-of-Sample Temporal Validation (2022–2024 $\to$ 2025–2026 Holdout)**:
   - Batting Strike Rate RMSE: **$-27.1\%$ error reduction** vs simple career baseline ($r = +0.640$).
   - Bowling Economy RMSE: **$-12.1\%$ error reduction** vs simple career baseline ($r = +0.318$).

---

## 4. Repository Structure

```
├── Data Pack/                           # Source CSV data pack (883 matches, 200,999 balls)
│   ├── ball_by_ball.csv
│   ├── competitions.csv
│   ├── competition_team_player_mapping.csv
│   ├── matches.csv
│   ├── playing_eleven.csv
│   └── teams.csv
├── outputs/                             # Pipeline generated deliverables & trained model
│   ├── figures/                         # High-res publication charts
│   ├── full_batter_rankings.csv         # Complete qualified batter rankings
│   ├── full_bowler_rankings.csv         # Complete qualified bowler rankings
│   ├── model_validation_comparison.csv  # Out-of-sample holdout validation results
│   ├── expected_outcome_model_comparison.csv # Single-ball model architecture benchmark
│   ├── sensitivity_analysis.csv         # Definition stability (Standard/Strict/Broad)
│   ├── top10_batters.csv                # Submission Top 10 Batters
│   ├── top10_bowlers.csv                # Submission Top 10 Bowlers
│   ├── pressure_model.py                # Standalone reproducible retraining script
│   └── saved_model.pkl                  # Serialized ML models & parameters
├── src/                                 # Reusable modular pipeline library
│   ├── config.py                        # Hyperparameters, thresholds & auto-detected paths
│   ├── data_loader.py                   # Ingestion, validation & hygiene filters
│   ├── evaluator.py                     # Out-of-sample holdout validation engine
│   ├── metrics.py                       # ML models, ICC, shrinkage & dual baselines
│   ├── pressure_engine.py               # Pre-delivery state reconstruction (zero leakage)
│   ├── ranker.py                        # BPRI scoring formula & qualification
│   └── visualizer.py                    # Publication figure generator
├── assets/                              # Author profile assets
│   └── Muthu_Kumar_M_Resume_ATS_SE.pdf  # Candidate resume
├── .streamlit/
│   └── config.toml                      # Production server & dark theme configuration
├── dashboard.py                         # Interactive Streamlit dashboard
├── run_analysis.py                      # Main entrypoint script
├── METHODOLOGY_REPORT.md                # Comprehensive technical methodology report
├── Procfile                             # Process file for cloud container deployment
├── render.yaml                          # Render infrastructure-as-code blueprint
├── requirements.txt                     # Python package dependencies
└── README.md
```

---

## 5. Quickstart & Local Setup

### Prerequisites
- Python 3.10+ recommended

### Installation
```bash
# Clone the repository
git clone https://github.com/MuthuKumar257/SupeRR-Selector-Season-2-.git
cd SupeRR-Selector-Season-2-

# Install required dependencies
pip install -r requirements.txt
```

### Running the Interactive Dashboard
```bash
streamlit run dashboard.py
```
Open your browser at `http://localhost:8501`.

### Retraining the Pressure Model from Source Data
To retrain the expected-outcome ML models, re-evaluate holdout temporal accuracy, and re-export all rankings from the raw CSVs:
```bash
python run_analysis.py
# or:
python outputs/pressure_model.py
```

---

## 6. Cloud Deployment (Render)

This repository is pre-configured with `render.yaml` and `Procfile` for 1-click cloud deployment:
1. Connect your GitHub repository to **[Render](https://render.com/)**.
2. Create a new **Web Service**.
3. Set **Build Command** to: `pip install -r requirements.txt`
4. Set **Start Command** to: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
5. Select the **Free** plan and deploy.

---

## 7. Author & Contact

**Muthu Kumar M**  
- **Portfolio**: [https://muthukumar725.vercel.app/](https://muthukumar725.vercel.app/)  
- **Resume**: Downloadable directly in the dashboard sidebar or via [assets/Muthu_Kumar_M_Resume_ATS_SE.pdf](assets/Muthu_Kumar_M_Resume_ATS_SE.pdf)
