#!/usr/bin/env python3
"""
Downloads and prepares the real NCBI GEO Colorectal Cancer RNA-seq cohorts
and MSigDB Hallmark gene sets according to Sections 6 & 7.
No synthetic or dummy data are used.
"""
import os
import pickle
import yaml
from src.data_loader import load_real_geo_cohorts, run_quality_control

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("[01_prepare_cohorts] Downloading real NCBI GEO RNA-seq cohorts and MSigDB Hallmark sets...")
    raw_cohorts, gene_sets = load_real_geo_cohorts(config, raw_dir="data/raw")

    print("[01_prepare_cohorts] Running 7-step Quality Control Checklist (Section 7.2)...")
    filtered_cohorts, table5_df, pca_records = run_quality_control(
        raw_cohorts, fdr_alpha=config["thresholds"]["fdr_alpha"]
    )

    table5_df.to_csv("results/tables/Table5_Cohort_Characteristics.csv", index=False)
    table5_df.to_latex("results/tables/Table5_Cohort_Characteristics.tex", index=False)

    with open("data/processed/curated_cohorts.pkl", "wb") as f:
        pickle.dump({
            "cohorts": filtered_cohorts,
            "gene_sets": gene_sets,
            "pca_records": pca_records,
            "table5": table5_df
        }, f)

    print("\n=== Cohort Summary (Table 5) ===")
    print(table5_df.to_string(index=False))
    print("\n[01_prepare_cohorts] Saved verified real cohorts to data/processed/curated_cohorts.pkl")

if __name__ == "__main__":
    main()
