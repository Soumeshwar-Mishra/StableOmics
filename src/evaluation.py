import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from src.de_engine import fit_differential_expression

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes J(A, B) = |A cap B| / |A cup B| (Eq 8)."""
    union_len = len(set_a.union(set_b))
    return len(set_a.intersection(set_b)) / union_len if union_len > 0 else 0.0

def evaluate_method_on_test_cohort(
    dev_rankings_df: pd.DataFrame,
    score_col: str,
    test_de_df: pd.DataFrame,
    K: int = 50,
    val_q_alpha: float = 0.05
) -> dict:
    """
    Evaluates a frozen development ranking once on the locked test cohort D_h (Eq 7 & 8).
    """
    # Identify top-K genes from development ranking
    ordered_dev_genes = dev_rankings_df[score_col].sort_values(ascending=False).index
    top_k_dev = list(ordered_dev_genes[:K])

    # Identify top-K genes in validation cohort by test statistical significance
    ordered_val_genes = test_de_df["p_val"].sort_values(ascending=True).index
    top_k_val = list(ordered_val_genes[:K])

    # Directional replication DR_{K,c} (Eq 7)
    beta_col = "single_cohort_beta" if score_col == "Single_DE" else "dev_beta"
    dev_signs = np.sign(dev_rankings_df.loc[top_k_dev, beta_col].values)
    val_signs = np.sign(test_de_df.loc[top_k_dev, "beta"].values)
    val_qvals = test_de_df.loc[top_k_dev, "q_val"].values

    dir_rep = float(np.mean(dev_signs == val_signs))
    sig_dir_rep = float(np.mean((dev_signs == val_signs) & (val_qvals < val_q_alpha)))

    # Gene-set overlap (Eq 8)
    jacc = jaccard_similarity(set(top_k_dev), set(top_k_val))
    overlap_k = len(set(top_k_dev).intersection(set(top_k_val))) / float(K)

    # Whole-transcriptome rank correlation (Spearman rho)
    common_genes = dev_rankings_df.index
    val_rank_score = -np.log10(np.maximum(test_de_df.loc[common_genes, "p_val"].values, 1e-300))
    rho, _ = spearmanr(dev_rankings_df.loc[common_genes, score_col].values, val_rank_score)

    return {
        "DR_K": dir_rep,
        "Sig_DR_K": sig_dir_rep,
        "TopK_Overlap": overlap_k,
        "Jaccard": jacc,
        "Spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "Error_Rate": 1.0 - dir_rep
    }

def patient_bootstrap_ci_test_cohort(
    dev_rankings_df: pd.DataFrame,
    methods: list,
    test_expr: pd.DataFrame,
    test_meta: pd.DataFrame,
    test_assay: str,
    K: int = 50,
    n_boot: int = 100,
    seed: int = 42
) -> dict:
    """
    Computes 95% confidence intervals on the held-out cohort using patient-level stratified
    bootstrap (Section 11.5 Rule 2), avoiding pseudoreplication.
    """
    rng = np.random.default_rng(seed)
    case_idx = np.where(test_meta["condition"].values == 1)[0]
    ctrl_idx = np.where(test_meta["condition"].values == 0)[0]

    boot_records = {m: [] for m in methods}

    for _ in range(n_boot):
        b_cases = rng.choice(case_idx, size=len(case_idx), replace=True)
        b_ctrls = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
        b_idx = np.concatenate([b_cases, b_ctrls])

        b_de = fit_differential_expression(test_expr.iloc[b_idx], test_meta.iloc[b_idx], assay=test_assay)
        for m in methods:
            res = evaluate_method_on_test_cohort(dev_rankings_df, m, b_de, K=K)
            boot_records[m].append(res["DR_K"])

    ci_dict = {}
    for m in methods:
        arr = np.array(boot_records[m])
        ci_dict[m] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci_dict

def evaluate_secondary_prediction(
    dev_cohorts: dict,
    test_cohort: dict,
    selected_genes: list
) -> dict:
    """
    Evaluates secondary predictive generalization (ROC-AUC, PR-AUC, Brier Score)
    using standardized expression features and L2-regularized logistic regression (Sec 8.6 & 9).
    """
    X_train_list, y_train_list = [], []
    for c_name, d in dev_cohorts.items():
        mat = d["expr"][selected_genes].values.astype(float)
        if d["assay"] == "rnaseq_counts":
            mat = np.log2(mat + 1.0)
        mat = StandardScaler().fit_transform(mat)
        X_train_list.append(mat)
        y_train_list.append(d["meta"]["condition"].values)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    X_test = test_cohort["expr"][selected_genes].values.astype(float)
    if test_cohort["assay"] == "rnaseq_counts":
        X_test = np.log2(X_test + 1.0)
    X_test = StandardScaler().fit_transform(X_test)
    y_test = test_cohort["meta"]["condition"].values

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    return {
        "ROC_AUC": float(roc_auc_score(y_test, probs)),
        "PR_AUC": float(average_precision_score(y_test, probs)),
        "Brier_Score": float(brier_score_loss(y_test, probs))
    }
