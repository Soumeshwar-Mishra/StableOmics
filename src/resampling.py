import numpy as np
import pandas as pd
from src.de_engine import fit_differential_expression, select_genes

def run_stratified_resampling(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    assay: str,
    B: int,
    n_per_group: int = None,
    case_fraction: float = 0.5,
    fdr_alpha: float = 0.05,
    min_lfc: float = 0.585,
    selection_rule: str = "fdr_lfc",
    top_k: int = 50,
    seed: int = 42,
    track_pairwise_sets: bool = False
) -> dict:
    """
    Executes B class-stratified subsamples using streaming counters (Section 12.1).
    Computes Selection Stability S_g (Eq 1), Direction Consistency D_g (Eq 2),
    and BootstrapSeq replicability score.
    """
    rng = np.random.default_rng(seed)
    genes = expr_df.columns
    n_genes = len(genes)

    case_indices = np.where(meta_df["condition"].values == 1)[0]
    ctrl_indices = np.where(meta_df["condition"].values == 0)[0]

    if n_per_group is None:
        # Default 80% subsampling without replacement for stability estimation
        n_case_sub = max(4, int(0.8 * len(case_indices)))
        n_ctrl_sub = max(4, int(0.8 * len(ctrl_indices)))
    else:
        total_sub = 2 * n_per_group
        n_case_sub = max(3, min(len(case_indices), int(round(total_sub * case_fraction))))
        n_ctrl_sub = max(3, min(len(ctrl_indices), total_sub - n_case_sub))

    # Streaming O(G) counters
    sel_counter = np.zeros(n_genes, dtype=np.float64)
    sign_counter = np.zeros(n_genes, dtype=np.float64)
    bootstrap_seq_counter = np.zeros(n_genes, dtype=np.float64)
    beta_sum = np.zeros(n_genes, dtype=np.float64)
    # NEW (critique A): t-statistic accumulators for signal-to-noise ratio
    t_sum    = np.zeros(n_genes, dtype=np.float64)
    t_sq_sum = np.zeros(n_genes, dtype=np.float64)

    sampled_sets = []
    sampled_ranks = []

    for b in range(B):
        sub_cases = rng.choice(case_indices, size=n_case_sub, replace=False)
        sub_ctrls = rng.choice(ctrl_indices, size=n_ctrl_sub, replace=False)
        sub_idx = np.concatenate([sub_cases, sub_ctrls])

        sub_expr = expr_df.iloc[sub_idx]
        sub_meta = meta_df.iloc[sub_idx]

        de_res = fit_differential_expression(sub_expr, sub_meta, assay=assay)
        selected_mask = select_genes(de_res, rule=selection_rule, fdr_alpha=fdr_alpha, min_lfc=min_lfc, top_k=top_k)

        betas = de_res["beta"].values
        signs = np.sign(betas)

        sel_counter += selected_mask.astype(float)
        sign_counter += signs
        beta_sum += betas

        # NEW (critique A): accumulate t-statistics for SNR
        t_vals = de_res["t_stat"].values
        t_sum    += t_vals
        t_sq_sum += t_vals ** 2

        # FIXED BootstrapSeq (critique E): per-resample 95% bootstrap CI excludes zero.
        # This is genuinely distinct from S_g (which uses FDR+effect threshold).
        bseq_mask = np.abs(betas) > 1.96 * de_res["se"].values
        bootstrap_seq_counter += bseq_mask.astype(float)

        if track_pairwise_sets and len(sampled_sets) < 40:
            top_k_genes = set(genes[np.argsort(de_res["p_val"].values)[:top_k]])
            sampled_sets.append(top_k_genes)
            sampled_ranks.append(de_res["t_stat"].values)

    S_g = sel_counter / B
    D_g = np.abs(sign_counter / B)
    mean_beta = beta_sum / B
    bseq_score = bootstrap_seq_counter / B

    # NEW (critique A): SNR = |mean(t)| / (std(t)+eps) — disease signal, not housekeeping noise
    t_mean = t_sum / B
    t_var  = np.maximum(t_sq_sum / B - t_mean ** 2, 1e-10)
    t_std  = np.sqrt(t_var)
    SNR_g  = np.abs(t_mean) / (t_std + 1e-8)

    # Descriptive binomial MCSE (Section 8.6 & 11.4)
    mcse_S_g = np.sqrt(S_g * (1.0 - S_g) / B)

    summary_df = pd.DataFrame({
        "S_g": S_g,
        "D_g": D_g,
        "S_g_x_D_g": S_g * D_g,
        "BootstrapSeq": bseq_score,
        "mean_beta": mean_beta,
        "MCSE_S_g": mcse_S_g,
        "SNR_g": SNR_g,
        "t_mean": t_mean,
        "t_std": t_std
    }, index=genes)

    return {
        "summary_df": summary_df,
        "sampled_sets": sampled_sets,
        "sampled_ranks": sampled_ranks
    }
