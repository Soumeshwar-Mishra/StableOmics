import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300
})

def _save_fig(fig, out_dir: str, basename: str):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{basename}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{basename}.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_fig2_qc_panel(pca_records: dict, out_dir: str):
    """Fig 2: Per-cohort PCA and quality score distributions."""
    cohorts = list(pca_records.keys())
    n_c = len(cohorts)
    fig, axes = plt.subplots(2, n_c, figsize=(3.6 * n_c, 6.5))
    if n_c == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, c_name in enumerate(cohorts):
        df = pca_records[c_name]
        ax_pca = axes[0, idx]
        for cond, label, marker, color in [(1, "Case", "o", "#D55E00"), (0, "Control", "s", "#0072B2")]:
            sub = df[df["condition"] == cond]
            ax_pca.scatter(sub["PC1"], sub["PC2"], label=label, marker=marker, c=color, alpha=0.8, edgecolors="k", linewidth=0.4)
        v1 = df["var_exp_1"].iloc[0] * 100
        v2 = df["var_exp_2"].iloc[0] * 100
        ax_pca.set_title(f"{c_name} PCA")
        ax_pca.set_xlabel(f"PC1 ({v1:.1f}% var)")
        ax_pca.set_ylabel(f"PC2 ({v2:.1f}% var)")
        if idx == 0:
            ax_pca.legend(loc="best")

        ax_qual = axes[1, idx]
        sns.boxplot(
            data=df.assign(Group=np.where(df["condition"] == 1, "Case", "Control")),
            x="Group", y="quality_score", ax=ax_qual, palette={"Case": "#D55E00", "Control": "#0072B2"}, width=0.45
        )
        ax_qual.set_title(f"{c_name} Quality Audit")
        ax_qual.set_xlabel("Phenotype Group")
        ax_qual.set_ylabel("Sample Quality Score (0-1 scale)")
        ax_qual.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig2_Cohort_QC_Panel")

def plot_fig3_stability_vs_sample_size(ladder_df: pd.DataFrame, out_dir: str, top_k: int):
    """Fig 3: Median top-K Jaccard stability vs sample size with IQR uncertainty bands."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = {
        "Single-analysis DE": {"color": "#0072B2", "ls": "-", "marker": "o"},
        "StableOmics (S_g * D_g)": {"color": "#D55E00", "ls": "--", "marker": "s"}
    }
    for method, grp in ladder_df.groupby("Method"):
        st = styles.get(method, {"color": "k", "ls": "-", "marker": "^"})
        ax.plot(grp["Samples_per_group"], grp["Median_Stability"], label=method,
                color=st["color"], linestyle=st["ls"], marker=st["marker"], lw=2)
        ax.fill_between(grp["Samples_per_group"], grp["IQR_Low"], grp["IQR_High"], color=st["color"], alpha=0.18)

    ax.set_xlabel("Discovery Samples per Group (n cases = n controls)")
    ax.set_ylabel(f"Median Top-{top_k} Jaccard Stability (IQR band)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig3_Stability_vs_SampleSize")

def plot_fig4_perturbation_response(pert_df: pd.DataFrame, out_dir: str):
    """Fig 4: Sensitivity to controlled quality imbalance perturbations."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = {
        "Single-cohort DE": {"color": "#0072B2", "ls": "-", "marker": "o"},
        "StableOmics (S_g * D_g)": {"color": "#009E73", "ls": "--", "marker": "D"}
    }
    for method, grp in pert_df.groupby("Method"):
        st = styles.get(method, {"color": "k", "ls": "-", "marker": "o"})
        ax.plot(grp["Perturbation_Intensity"], grp["Cross_Cohort_Rank_Agreement"],
                label=f"{method} (Simulated Perturbation)", color=st["color"], linestyle=st["ls"], marker=st["marker"], lw=2)

    ax.set_xlabel("Simulated Quality-Imbalance Perturbation Intensity (0 = Unperturbed)")
    ax.set_ylabel("External Cohort Rank Agreement (Spearman |rho|)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig4_Perturbation_Sensitivity")

def plot_fig5_validation_forest(per_cohort_eval_df: pd.DataFrame, out_dir: str, K: int):
    """Fig 5: Per-cohort external directional replication forest plot (no hidden cohorts)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cohorts = per_cohort_eval_df["Held_Out_Cohort"].unique()
    methods = ["Single_DE", "S_g_only", "BootstrapSeq", "SumRank_Bulk", "StableOmics_Rg"]
    palette = {"Single_DE": "#56B4E9", "S_g_only": "#E69F00", "BootstrapSeq": "#CC79A7",
               "SumRank_Bulk": "#0072B2", "StableOmics_Rg": "#D55E00"}
    markers = {"Single_DE": "o", "S_g_only": "^", "BootstrapSeq": "v", "SumRank_Bulk": "D", "StableOmics_Rg": "s"}

    y_positions = []
    y_labels = []
    pos = 0
    for c_name in cohorts:
        sub_c = per_cohort_eval_df[per_cohort_eval_df["Held_Out_Cohort"] == c_name]
        for m in methods:
            row = sub_c[sub_c["Method"] == m]
            if row.empty:
                continue
            est = row["DR_K"].iloc[0]
            ci_l = row["CI_Low"].iloc[0]
            ci_h = row["CI_High"].iloc[0]
            ax.errorbar(est, pos, xerr=[[max(0, est - ci_l)], [max(0, ci_h - est)]],
                        fmt=markers[m], color=palette[m], label=m if c_name == cohorts[0] else "", capsize=3)
            y_positions.append(pos)
            y_labels.append(f"{c_name} | {m}")
            pos += 1
        pos += 0.8

    ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="Random Chance (0.5)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(f"Top-{K} Directional Replication Rate (DR_{K} with 95% Patient-Bootstrap CI)")
    ax.set_xlim(0.2, 1.05)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig5_External_Validation_ForestPlot")

def plot_fig6_method_heatmap(per_cohort_eval_df: pd.DataFrame, out_dir: str, K: int):
    """Fig 6 (manuscript Figure 3): Method-by-cohort heatmaps — fixed x-axis label overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    piv_dr = per_cohort_eval_df.pivot(index="Method", columns="Held_Out_Cohort", values="DR_K")
    piv_ov = per_cohort_eval_df.pivot(index="Method", columns="Held_Out_Cohort", values="TopK_Overlap")

    sns.heatmap(
        piv_dr, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.4, vmax=1.0,
        ax=axes[0], cbar_kws={"label": f"DR_{K}"},
        annot_kws={"fontsize": 9},
    )
    axes[0].set_title(f"Directional Replication (DR_{K} ↑)")
    axes[0].set_xlabel("Locked External Cohort")
    axes[0].set_ylabel("Ranking Method")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right", fontsize=10)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=9)

    sns.heatmap(
        piv_ov, annot=True, fmt=".2f", cmap="BuPu", vmin=0.0, vmax=0.8,
        ax=axes[1], cbar_kws={"label": f"O_{K}"},
        annot_kws={"fontsize": 9},
    )
    axes[1].set_title(f"Top-{K} Gene Set Overlap (O_{K} ↑)")
    axes[1].set_xlabel("Locked External Cohort")
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right", fontsize=10)
    axes[1].tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig6_Ranking_Comparison_Heatmap")


def plot_fig7_pathway_reproducibility(pw_summary_df: pd.DataFrame, out_dir: str):
    """Fig 7: Dot plot of pathway-level reproducibility across outer folds."""
    top_pw = pw_summary_df.head(12).copy()
    top_pw["neg_log10_q"] = -np.log10(np.maximum(top_pw["Median_q_val"], 1e-10))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    scatter = ax.scatter(
        top_pw["Mean_Overlap_Ratio"],
        top_pw["Pathway"],
        s=top_pw["Cohort_Support"] * 55 + 30,
        c=top_pw["neg_log10_q"],
        cmap="viridis",
        edgecolors="k",
        alpha=0.9
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Median -log10(FDR q-value)")
    ax.set_xlabel("Mean Gene-Set Overlap Ratio (Selected / Pathway Size)")
    ax.set_ylabel("MSigDB Gene Set")
    ax.set_title("Pathway-Level Reproducibility (Dot size = Outer-Fold Cohort Support)")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig7_Pathway_Reproducibility")

def plot_fig8_resource_profile(prof_records: list, out_dir: str):
    """Fig 8: Computational runtime and peak memory profile."""
    df = pd.DataFrame(prof_records)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    sns.barplot(data=df, y="Task", x="Time_s", ax=axes[0], color="#0072B2", capsize=0.15)
    axes[0].set_xlabel("Wall-Clock Runtime (seconds)")
    axes[0].set_ylabel("Pipeline Stage")
    axes[0].set_title("CPU Runtime by Stage")

    sns.barplot(data=df, y="Task", x="Peak_RAM_GB", ax=axes[1], color="#009E73", capsize=0.15)
    axes[1].axvline(12.0, color="#D55E00", linestyle="--", label="Hardware Limit (12 GB)")
    axes[1].set_xlabel("Peak Resident RAM (GB)")
    axes[1].set_ylabel("")
    axes[1].set_title("Memory Footprint vs 12 GB Target")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    _save_fig(fig, out_dir, "Fig8_Resource_Profile")
