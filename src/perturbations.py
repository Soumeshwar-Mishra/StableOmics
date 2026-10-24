import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.de_engine import fit_differential_expression
from src.resampling import run_stratified_resampling
from src.evaluation import jaccard_similarity

def run_sample_size_ladder(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    assay: str,
    sample_sizes: list,
    B: int = 50,
    top_k: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Evaluates within-cohort stability across feasible sample sizes (Section 8.7 & Fig 3)."""
    n_cases = int((meta_df["condition"] == 1).sum())
    n_ctrls = int((meta_df["condition"] == 0).sum())
    max_feasible = min(n_cases, n_ctrls) - 1

    feasible_sizes = sorted(list(set([min(n, max_feasible) for n in sample_sizes if min(n, max_feasible) >= 3])))
    if not feasible_sizes:
        feasible_sizes = [max(3, max_feasible)]

    rows = []
    for n_per_grp in feasible_sizes:
        res = run_stratified_resampling(
            expr_df, meta_df, assay=assay, B=B, n_per_group=n_per_grp,
            top_k=top_k, seed=seed + n_per_grp, track_pairwise_sets=True
        )
        sets = res["sampled_sets"]
        jacc_vals = [
            jaccard_similarity(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))
        ] or [0.0]

        # Real bootstrap distribution for the StableOmics top-K stability:
        # 5 independent split-halves; Jaccard between top-K sets from each half.
        so_jaccs = []
        n_half = max(10, B // 2)
        for split in range(5):
            r1 = run_stratified_resampling(expr_df, meta_df, assay, B=n_half, n_per_group=n_per_grp, seed=seed + 1000 + split*2)
            r2 = run_stratified_resampling(expr_df, meta_df, assay, B=n_half, n_per_group=n_per_grp, seed=seed + 1000 + split*2 + 1)
            t1 = set(r1["summary_df"]["S_g_x_D_g"].nlargest(top_k).index)
            t2 = set(r2["summary_df"]["S_g_x_D_g"].nlargest(top_k).index)
            so_jaccs.append(jaccard_similarity(t1, t2))

        rows.append({
            "Samples_per_group": n_per_grp,
            "Method": "Single-analysis DE",
            "Median_Stability": float(np.median(jacc_vals)),
            "IQR_Low": float(np.percentile(jacc_vals, 25)),
            "IQR_High": float(np.percentile(jacc_vals, 75))
        })
        rows.append({
            "Samples_per_group": n_per_grp,
            "Method": "StableOmics (S_g * D_g)",
            "Median_Stability": float(np.median(so_jaccs)),
            "IQR_Low": float(np.percentile(so_jaccs, 25)),
            "IQR_High": float(np.percentile(so_jaccs, 75))
        })
    return pd.DataFrame(rows)

def run_perturbation_experiments(
    dev_cohort: dict,
    val_cohort: dict,
    intensities: list,
    B: int = 40,
    top_k: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Tests sensitivity to controlled quality-imbalance perturbations on real expression matrices (Sec 8.7 & Fig 4)."""
    rng = np.random.default_rng(seed)
    val_de = fit_differential_expression(val_cohort["expr"], val_cohort["meta"], val_cohort["assay"])
    val_scores = -np.log10(np.maximum(val_de["p_val"].values, 1e-300)) * np.sign(val_de["beta"].values)

    rows = []
    for intensity in intensities:
        pert_expr = dev_cohort["expr"].copy().astype(float)
        meta = dev_cohort["meta"].copy()

        if intensity > 0:
            case_mask = (meta["condition"].values == 1)
            n_cases = int(case_mask.sum())
            n_genes = pert_expr.shape[1]
            noise = rng.normal(loc=intensity * 1.5, scale=intensity * 1.2, size=(n_cases, n_genes))
            if dev_cohort["assay"] == "rnaseq_counts":
                pert_expr.iloc[case_mask] = np.maximum(0, np.round(pert_expr.iloc[case_mask] * (2.0 ** (noise * 0.3)))).astype(int)
            else:
                pert_expr.iloc[case_mask] += noise

        de_pert = fit_differential_expression(pert_expr, meta, dev_cohort["assay"])
        de_score = -np.log10(np.maximum(de_pert["p_val"].values, 1e-300)) * np.sign(de_pert["beta"].values)
        rho_de, _ = spearmanr(de_score, val_scores)

        res_so = run_stratified_resampling(pert_expr, meta, dev_cohort["assay"], B=B, top_k=top_k, seed=seed)
        so_score = res_so["summary_df"]["S_g_x_D_g"].values * np.sign(res_so["summary_df"]["mean_beta"].values)
        rho_so, _ = spearmanr(so_score, val_scores)

        rows.append({
            "Perturbation_Intensity": intensity,
            "Method": "Single-cohort DE",
            "Cross_Cohort_Rank_Agreement": float(abs(rho_de)),
            "Type": "Perturbed Real Cohort"
        })
        rows.append({
            "Perturbation_Intensity": intensity,
            "Method": "StableOmics (S_g * D_g)",
            "Cross_Cohort_Rank_Agreement": float(abs(rho_so)),
            "Type": "Perturbed Real Cohort"
        })

    return pd.DataFrame(rows)
