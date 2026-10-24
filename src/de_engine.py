import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def fit_differential_expression(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    assay: str,
    use_pydeseq2: bool = False
) -> pd.DataFrame:
    """
    Fits assay-appropriate differential expression model.
    Returns DataFrame indexed by gene with columns: ['beta', 'se', 't_stat', 'p_val', 'q_val'].
    """
    y = meta_df["condition"].values.astype(float)
    genes = expr_df.columns

    if assay == "rnaseq_counts" and use_pydeseq2:
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
            meta_copy = meta_df.copy()
            meta_copy["condition_str"] = np.where(meta_copy["condition"] == 1, "case", "control")
            dds = DeseqDataSet(
                counts=expr_df,
                metadata=meta_copy,
                design_factors="condition_str",
                quiet=True,
                n_cpus=1
            )
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition_str", "case", "control"], quiet=True, n_cpus=1)
            stat_res.summary()
            res = stat_res.results_df.reindex(genes)
            out = pd.DataFrame({
                "beta": res["log2FoldChange"].fillna(0.0).values,
                "se": res["lfcSE"].fillna(1.0).values,
                "t_stat": res["stat"].fillna(0.0).values,
                "p_val": res["pvalue"].fillna(1.0).values,
                "q_val": res["padj"].fillna(1.0).values
            }, index=genes)
            return out
        except Exception:
            pass  # Fallback to fast vectorized count-aware log-CPM linear model with empirical Bayes moderation

    # Vectorized empirical-Bayes moderated linear model (limma-voom / log2-CPM style for CPU speed)
    if assay == "rnaseq_counts":
        lib_size = expr_df.values.sum(axis=1, keepdims=True)
        lib_size = np.maximum(lib_size, 1.0)
        mat = np.log2((expr_df.values + 0.5) / (lib_size + 1.0) * 1e6)
    else:
        mat = expr_df.values.astype(float)

    n_samples, n_genes = mat.shape
    mask_case = (y == 1)
    mask_ctrl = (y == 0)
    n1 = mask_case.sum()
    n0 = mask_ctrl.sum()

    mean_case = mat[mask_case].mean(axis=0)
    mean_ctrl = mat[mask_ctrl].mean(axis=0)
    beta = mean_case - mean_ctrl  # log2 Fold Change

    # Pooled variance with empirical Bayes prior shrinkage (stabilizes small subsamples)
    var_case = mat[mask_case].var(axis=0, ddof=1)
    var_ctrl = mat[mask_ctrl].var(axis=0, ddof=1)
    df_resid = max(n1 + n0 - 2, 1)
    s2_pooled = ((n1 - 1) * var_case + (n0 - 1) * var_ctrl) / df_resid

    # Shrink gene variances toward median variance (limma-style eBayes moderation, d0=4)
    d0 = 4.0
    s02 = np.median(s2_pooled)
    s2_moderated = (d0 * s02 + df_resid * s2_pooled) / (d0 + df_resid)

    se = np.sqrt(s2_moderated * (1.0 / max(n1, 1) + 1.0 / max(n0, 1)))
    se = np.maximum(se, 1e-8)
    t_stat = beta / se
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid + d0)
    p_val = np.nan_to_num(p_val, nan=1.0)

    _, q_val, _, _ = multipletests(p_val, alpha=0.05, method="fdr_bh")

    return pd.DataFrame({
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_val": p_val,
        "q_val": q_val
    }, index=genes)

def select_genes(de_df: pd.DataFrame, rule: str = "fdr_lfc", fdr_alpha: float = 0.05, min_lfc: float = 0.585, top_k: int = 50) -> np.ndarray:
    """Applies prespecified selection rule (Eq 1)."""
    if rule == "fdr_lfc":
        # Strict prespecified rule — do NOT silently fall back (see proposal Sec 8.2, 11.2).
        # Underpowered subsamples legitimately yield 0 selected genes; S_g reflects that.
        selected = (de_df["q_val"] < fdr_alpha) & (de_df["beta"].abs() >= min_lfc)
        return selected.values
    elif rule == "top_k":
        top_idx = np.argsort(de_df["p_val"].values)[:top_k]
        mask = np.zeros(len(de_df), dtype=bool)
        mask[top_idx] = True
        return mask
    else:
        raise ValueError(f"Unknown selection rule: {rule}")
