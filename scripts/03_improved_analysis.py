#!/usr/bin/env python3
# 03_improved_analysis.py — critique A-E applied
# A) SNR scores (StableOmics_Rg_SNR, SNR_g_only)
# B) Report SumRank/Meta_Effect alongside R_g
# C) TopK_Overlap at K=20 (sharp) in addition to K=50
# D) 3-cohort sensitivity excluding GSE44076 (quality-confounded)
# E) Fixed BootstrapSeq (already patched in src/resampling.py)
import pickle, yaml
import numpy as np
import pandas as pd
from src.resampling import run_stratified_resampling
from src.de_engine import fit_differential_expression
from src.scoring import compute_all_rankings_outer_fold
from src.evaluation import evaluate_method_on_test_cohort


def run_loco(cohorts, cohort_de, cohort_resamp, methods, K, fdr_alpha):
    names = list(cohorts.keys())
    rows = []
    for held_out in names:
        dev = [c for c in names if c != held_out]
        rank_df = compute_all_rankings_outer_fold(
            {c: cohort_de[c] for c in dev},
            {c: cohort_resamp[c] for c in dev}
        )
        test_de = cohort_de[held_out]
        for m in methods:
            if m not in rank_df.columns:
                continue
            metrics = evaluate_method_on_test_cohort(
                rank_df, m, test_de, K=K, val_q_alpha=fdr_alpha
            )
            rows.append({"Held_Out_Cohort": held_out, "K": K, "Method": m, **metrics})
    return pd.DataFrame(rows)


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    d = pickle.load(open("data/processed/curated_cohorts.pkl", "rb"))
    cohorts = d["cohorts"]

    B = cfg["resampling"]["B_quick"]
    seed = cfg["project"]["random_seed"]
    fdr = cfg["thresholds"]["fdr_alpha"]
    lfc = cfg["thresholds"]["min_abs_lfc"]
    rule = cfg["thresholds"]["selection_rule"]
    k_top = cfg["thresholds"]["primary_top_k"]

    methods = [
        "Single_DE", "S_g_only", "BootstrapSeq", "SumRank_Bulk", "Meta_Effect",
        "SNR_g_only", "Ablation_No_Cg (S_g*D_g)",
        "StableOmics_WeightedSum", "StableOmics_Rg", "StableOmics_Rg_SNR",
    ]

    print("STEP 1/5: resampling with SNR + fixed BootstrapSeq (B=%d)" % B)
    cohort_de, cohort_resamp = {}, {}
    for idx, (name, c) in enumerate(cohorts.items()):
        print("  -> %s" % name)
        cohort_de[name] = fit_differential_expression(c["expr"], c["meta"], c["assay"])
        cohort_resamp[name] = run_stratified_resampling(
            c["expr"], c["meta"], c["assay"], B=B,
            fdr_alpha=fdr, min_lfc=lfc, selection_rule=rule,
            top_k=k_top, seed=seed + idx * 100,
        )

    print("STEP 2/5: LOCO — all 4 cohorts, K=50 and K=20")
    df_k50 = run_loco(cohorts, cohort_de, cohort_resamp, methods, 50, fdr)
    df_k20 = run_loco(cohorts, cohort_de, cohort_resamp, methods, 20, fdr)
    df_k50.to_csv("results/tables/Improvement_PerCohort_K50.csv", index=False)
    df_k20.to_csv("results/tables/Improvement_PerCohort_K20.csv", index=False)

    print("STEP 3/5: 3-cohort sensitivity (no GSE44076, quality p=0.000)")
    cohorts_3 = {k: v for k, v in cohorts.items() if k != "GSE44076"}
    df_3c = run_loco(cohorts_3, cohort_de, cohort_resamp, methods, 50, fdr)
    df_3c.to_csv("results/tables/Improvement_3cohort.csv", index=False)

    print("\nSTEP 4/5: Summary TopK_Overlap ranking")
    for tag, df in [("K=50 (primary)", df_k50),
                    ("K=20 (sharp)", df_k20),
                    ("3-cohort (no GSE44076, K=50)", df_3c)]:
        print("\n--- %s ---" % tag)
        agg = (df.groupby("Method")["TopK_Overlap"]
                 .agg(["mean", "min", "max"]).sort_values("mean", ascending=False))
        print(agg.round(3).to_string())

    print("\nSTEP 5/5: Paired deltas vs Single_DE (TopK_Overlap)")
    for tag, df in [("K=50", df_k50), ("K=20", df_k20), ("3-cohort", df_3c)]:
        ref = df[df["Method"] == "Single_DE"].set_index("Held_Out_Cohort")["TopK_Overlap"]
        rows = []
        for m in df["Method"].unique():
            if m == "Single_DE":
                continue
            oth = df[df["Method"] == m].set_index("Held_Out_Cohort")["TopK_Overlap"]
            dd = oth - ref
            rows.append({"Method": m, "delta": dd.mean(),
                         "min": dd.min(), "max": dd.max(),
                         "wins": int((dd > 0).sum()),
                         "losses": int((dd < 0).sum())})
        print("\n--- %s ---" % tag)
        pv = pd.DataFrame(rows).sort_values("delta", ascending=False)
        print(pv.round(3).to_string(index=False))
        safe = tag.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        pv.to_csv("results/tables/Improvement_PairedDelta_%s.csv" % safe, index=False)

    print("\nDONE — all outputs in results/tables/Improvement_*.csv")


if __name__ == "__main__":
    main()
