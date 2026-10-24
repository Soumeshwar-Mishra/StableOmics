import numpy as np
import pandas as pd

def compute_cross_cohort_consistency(dev_de_dict: dict) -> pd.Series:
    """
    Computes C_g^{(-h)} (Eq 3) using ONLY development cohorts in outer fold (-h).
    """
    cohort_names = list(dev_de_dict.keys())
    M = len(cohort_names)
    genes = dev_de_dict[cohort_names[0]].index

    if M < 2:
        # Reduced score case (Section 6.4 & 8.5): C_g is not estimable without leakage
        return pd.Series(np.nan, index=genes)

    concordance_sum = np.zeros(len(genes), dtype=np.float64)
    valid_pairs    = np.zeros(len(genes), dtype=np.float64)

    for i in range(M):
        for j in range(i + 1, M):
            beta_i = dev_de_dict[cohort_names[i]].loc[genes, "beta"].values
            beta_j = dev_de_dict[cohort_names[j]].loc[genes, "beta"].values
            sign_i = np.sign(beta_i)
            sign_j = np.sign(beta_j)
            both = (sign_i != 0) & (sign_j != 0)
            concordance_sum += ((sign_i == sign_j) & both).astype(float)
            valid_pairs     += both.astype(float)

    # Prespecified handling: pairs with undefined (zero) direction are excluded,
    # not counted as disagreements. C_g = NaN when no valid pairs exist.
    C_g = np.where(valid_pairs > 0, concordance_sum / np.maximum(valid_pairs, 1), np.nan)
    return pd.Series(C_g, index=genes)

def compute_meta_heterogeneity(dev_de_dict: dict) -> pd.DataFrame:
    """
    Computes random-effects inverse-variance meta-analysis, Cochran's Q_g, and I_g^2 (Eq 9, Sec 11.6).
    """
    cohort_names = list(dev_de_dict.keys())
    C = len(cohort_names)
    genes = dev_de_dict[cohort_names[0]].index

    betas = np.column_stack([dev_de_dict[c].loc[genes, "beta"].values for c in cohort_names])
    ses = np.column_stack([dev_de_dict[c].loc[genes, "se"].values for c in cohort_names])
    weights = 1.0 / np.maximum(ses ** 2, 1e-8)

    fixed_mean = np.sum(weights * betas, axis=1) / np.sum(weights, axis=1)
    Q_g = np.sum(weights * ((betas - fixed_mean[:, None]) ** 2), axis=1)

    I2_g = 100.0 * np.maximum(0.0, (Q_g - (C - 1.0)) / np.maximum(Q_g, 1e-8))
    meta_se = 1.0 / np.sqrt(np.sum(weights, axis=1))
    meta_z = np.abs(fixed_mean / np.maximum(meta_se, 1e-8))

    return pd.DataFrame({
        "meta_beta": fixed_mean,
        "meta_abs_z": meta_z,
        "Cochran_Q": Q_g,
        "I2_percent": I2_g
    }, index=genes)

def compute_all_rankings_outer_fold(dev_de_dict: dict, dev_resamp_dict: dict) -> pd.DataFrame:
    """
    Computes StableOmics R_g^{(-h)} (Eq 4-6) and all required comparators & ablations (Sec 11.7)
    strictly within development cohorts.
    """
    cohort_names = list(dev_de_dict.keys())
    M = len(cohort_names)
    genes = dev_de_dict[cohort_names[0]].index

    # Equal cohort weighting (Eq 4 & Eq 5)
    S_mat = np.column_stack([dev_resamp_dict[c]["summary_df"].loc[genes, "S_g"].values for c in cohort_names])
    D_mat = np.column_stack([dev_resamp_dict[c]["summary_df"].loc[genes, "D_g"].values for c in cohort_names])
    bseq_mat = np.column_stack([dev_resamp_dict[c]["summary_df"].loc[genes, "BootstrapSeq"].values for c in cohort_names])
    mean_beta_mat = np.column_stack([dev_de_dict[c].loc[genes, "beta"].values for c in cohort_names])
    snr_mat = np.column_stack([dev_resamp_dict[c]["summary_df"].loc[genes, "SNR_g"].values for c in cohort_names])

    S_bar = S_mat.mean(axis=1)
    D_bar = D_mat.mean(axis=1)
    bseq_bar = bseq_mat.mean(axis=1)
    snr_bar = snr_mat.mean(axis=1)
    dev_consensus_beta = mean_beta_mat.mean(axis=1)

    # Single-cohort DE baseline (uses the largest development cohort as single-study reference)
    first_cohort_de = dev_de_dict[cohort_names[0]].loc[genes]
    single_de_score = - np.log10(np.maximum(first_cohort_de["p_val"].values, 1e-300)) * np.abs(first_cohort_de["beta"].values)

    # Bulk ADAPTATION of SumRank (Nakatsuka et al., 2025) — NOT a direct reproduction.
    # Upstream SumRank targets sc/snRNA-seq; here we average 1 - normalized p-value ranks across dev cohorts.
    rank_mat = np.column_stack([
        dev_de_dict[c].loc[genes, "p_val"].rank(ascending=True).values / len(genes)
        for c in cohort_names
    ])
    sumrank_score = 1.0 - rank_mat.mean(axis=1)

    meta_df = compute_meta_heterogeneity(dev_de_dict)

    if M >= 2:
        C_g = compute_cross_cohort_consistency(dev_de_dict).values
        R_g = S_bar * D_bar * C_g
        R_g_weighted_sum = 0.4 * S_bar + 0.3 * D_bar + 0.3 * C_g
    else:
        C_g = np.ones(len(genes))
        R_g = S_bar * D_bar  # Reduced score R_g^(2)
        R_g_weighted_sum = 0.5 * S_bar + 0.5 * D_bar

    # NEW (critique A): SNR replaces S_g in the product. Normalize to [0,1] via rank.
    snr_norm = pd.Series(snr_bar, index=genes).rank(pct=True).values
    C_or_1 = C_g if M >= 2 else np.ones(len(genes))
    R_g_snr = snr_norm * D_bar * C_or_1
    R_g_snr_raw = snr_bar * D_bar * C_or_1

    return pd.DataFrame({
        "dev_beta": dev_consensus_beta,
        "single_cohort_beta": first_cohort_de["beta"].values,
        "Single_DE": single_de_score,
        "S_g_only": S_bar,
        "D_g_only": D_bar,
        "C_g_only": C_g,
        "Ablation_No_Cg (S_g*D_g)": S_bar * D_bar,
        "Ablation_No_Dg (S_g*C_g)": S_bar * C_g,
        "BootstrapSeq": bseq_bar,
        "SumRank_Bulk": sumrank_score,
        "Meta_Effect": meta_df["meta_abs_z"].values,
        "StableOmics_Rg": R_g,
        "StableOmics_WeightedSum": R_g_weighted_sum,
        # NEW methods (critique A)
        "SNR_g_only": snr_bar,
        "StableOmics_Rg_SNR": R_g_snr,
        "StableOmics_Rg_SNR_raw": R_g_snr_raw,
        "Cochran_Q": meta_df["Cochran_Q"].values,
        "I2_percent": meta_df["I2_percent"].values
    }, index=genes)
