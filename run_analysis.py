"""
Main Execution Script for SupeRR Selector Season 2 Round 1 Challenge.
Orchestrates end-to-end data preparation, pre-delivery pressure classification,
performance measurement against baseline, out-of-sample temporal validation, ranking,
sensitivity checks, and visual outputs.

Usage:
    python run_analysis.py [--data-dir <path>] [--output-dir <path>]
"""

import os
import argparse
from outputs.pressure_model import train_and_evaluate
from src.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="SupeRR Selector Season 2: Player Reliability Under Pressure")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to Data Pack folder containing CSV files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    args = parser.parse_args()
    
    train_and_evaluate(data_dir=args.data_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
