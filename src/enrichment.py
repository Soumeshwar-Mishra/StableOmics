import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

def run_pathway_enrichment(
    selected_genes: list,
    background_genes: list,
    gene_sets: dict,
    fdr_alpha: float = 0.05
) -> pd.DataFrame:
    """
    Runs Over-Representation Analysis (ORA) against a documented background universe (Sec 8.8).
    """
    bg_set = set(background_genes)
    sel_set = set(selected_genes).intersection(bg_set)
    M = len(bg_set)
    n = len(sel_set)

    rows = []
    for pathway, members in gene_sets.items():
        pw_set = set(members).intersection(bg_set)
        N = len(pw_set)
        if N < 3:
            continue
        k = len(sel_set.intersection(pw_set))
        # Hypergeometric survival function P(X >= k)
        pval = hypergeom.sf(k - 1, M, N, n) if k > 0 else 1.0
        rows.append({
            "Pathway": pathway,
            "Pathway_Size": N,
            "Overlap_Count": k,
            "Overlap_Ratio": k / float(N),
            "p_val": float(pval)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        _, qvals, _, _ = multipletests(df["p_val"].values, alpha=fdr_alpha, method="fdr_bh")
        df["q_val"] = qvals
        df = df.sort_values("p_val", ascending=True)
    return df

def evaluate_pathway_reproducibility(
    outer_fold_rankings: dict,
    cohort_de_dict: dict,
    gene_sets: dict,
    top_k: int = 100
) -> pd.DataFrame:
    """Aggregates pathway enrichment support across outer-fold cohorts for Figure 7."""
    bg_genes = list(next(iter(cohort_de_dict.values())).index)
    pathway_records = []

    for held_out, rank_df in outer_fold_rankings.items():
        top_so = list(rank_df["StableOmics_Rg"].nlargest(top_k).index)
        ora_df = run_pathway_enrichment(top_so, bg_genes, gene_sets)
        ora_df["Held_Out_Cohort"] = held_out
        pathway_records.append(ora_df)

    all_ora = pd.concat(pathway_records, ignore_index=True)
    summary = all_ora.groupby("Pathway", as_index=False).agg(
        Pathway_Size=("Pathway_Size", "first"),
        Mean_Overlap_Ratio=("Overlap_Ratio", "mean"),
        Median_q_val=("q_val", "median"),
        Cohort_Support=("q_val", lambda q: int((q < 0.10).sum()))
    ).sort_values(["Cohort_Support", "Median_q_val"], ascending=[False, True])

    return summary
