import os
import io
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

def parse_gpl_annotation(annot_filename: str, raw_dir: str = "data/raw") -> dict:
    """
    Parses official NCBI GEO GPL*.annot.gz or GPL*_family.soft.gz files
    to map Probe ID -> Official HGNC Gene Symbol with disk caching for speed.
    """
    cache_path = os.path.join("data/processed", f"{os.path.splitext(annot_filename)[0]}_map.pkl")
    if os.path.exists(cache_path):
        print(f"  [Annotation] Loading cached probe map from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    annot_path = os.path.join(raw_dir, annot_filename)
    if not os.path.exists(annot_path):
        # Try soft-family fallback with same stem
        stem = os.path.splitext(annot_filename)[0]
        soft_alt = os.path.join(raw_dir, f"{stem}_family.soft.gz")
        if os.path.exists(soft_alt):
            annot_path = soft_alt
        else:
            raise FileNotFoundError(
                f"Annotation file not found: {annot_path} or {soft_alt}. "
                f"Run scripts/download_geo_data.py first."
            )

    print(f"  [Annotation] Parsing large platform file {os.path.basename(annot_path)} (this happens once)...")
    probe_to_gene = {}
    in_table = False
    id_idx, sym_idx = 0, 2

    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                header_line = next(f, "").strip().split("\t")
                headers = [h.lower().strip() for h in header_line]
                if "id" in headers:
                    id_idx = headers.index("id")
                for idx, h in enumerate(headers):
                    if h in ("gene symbol", "symbol", "gene_symbol", "gb_acc"):
                        sym_idx = idx
                        break
                continue
            elif line.startswith("!platform_table_end"):
                in_table = False
                break
            elif in_table:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > max(id_idx, sym_idx):
                    probe_id = parts[id_idx].strip()
                    gene_sym = parts[sym_idx].strip()
                    if gene_sym and gene_sym != "---" and not gene_sym.startswith("//"):
                        primary_sym = gene_sym.split("///")[0].strip().upper()
                        if primary_sym and primary_sym != "---":
                            probe_to_gene[probe_id] = primary_sym

    os.makedirs("data/processed", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(probe_to_gene, f)

    print(f"  [Annotation] Cached {len(probe_to_gene)} probe-to-HGNC mappings to {cache_path}")
    return probe_to_gene

def parse_series_matrix(gse_id: str, raw_dir: str = "data/raw") -> tuple:
    """Parses clinical metadata and expression matrix from a GEO *_series_matrix.txt.gz file."""
    matrix_path = os.path.join(raw_dir, f"{gse_id}_series_matrix.txt.gz")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Missing {matrix_path}.")

    sample_ids, titles, sources, instruments = [], [], [], []
    char_lines = []
    expr_lines = []
    in_matrix = False

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_matrix:
                if line.startswith("!Sample_geo_accession"):
                    sample_ids = [x.strip('"') for x in line.strip().split("\t")[1:]]
                elif line.startswith("!Sample_title"):
                    titles = [x.strip('"') for x in line.strip().split("\t")[1:]]
                elif line.startswith("!Sample_source_name_ch1"):
                    sources = [x.strip('"') for x in line.strip().split("\t")[1:]]
                elif line.startswith("!Sample_characteristics_ch1"):
                    char_lines.append([x.strip('"') for x in line.strip().split("\t")[1:]])
                elif line.startswith("!Sample_instrument_model"):
                    instruments = [x.strip('"') for x in line.strip().split("\t")[1:]]
                elif line.startswith("!series_matrix_table_begin"):
                    in_matrix = True
            else:
                if line.startswith("!series_matrix_table_end"):
                    break
                expr_lines.append(line)

    n_samples = len(sample_ids)
    combined_chars = [
        " | ".join([row[i] for row in char_lines if i < len(row)])
        for i in range(n_samples)
    ] if char_lines else [""] * n_samples

    meta_df = pd.DataFrame({
        "sample_id": sample_ids,
        "title": titles if len(titles) == n_samples else [""] * n_samples,
        "source_name": sources if len(sources) == n_samples else [""] * n_samples,
        "characteristics": combined_chars,
        "instrument": instruments if len(instruments) == n_samples else ["Affymetrix_Array"] * n_samples
    }, index=sample_ids)

    expr_raw = pd.read_csv(
        io.StringIO("".join(expr_lines)),
        sep="\t",
        index_col=0,
        quotechar='"',
        low_memory=False
    )
    expr_raw.columns = [str(c).strip('"') for c in expr_raw.columns]
    expr_raw.index = expr_raw.index.astype(str).str.strip('"')
    return expr_raw, meta_df

def assign_phenotype_labels(meta_df: pd.DataFrame, spec: dict) -> pd.Series:
    """Assigns Case (1), Control (0), or Excluded (-1) from sample metadata."""
    labels = []
    case_kws = [k.lower() for k in spec["case_keywords"]]
    ctrl_kws = [k.lower() for k in spec["control_keywords"]]
    excl_kws = [k.lower() for k in spec.get("exclude_keywords", [])]

    for _, row in meta_df.iterrows():
        title = str(row["title"]).strip()
        text = f"{title} {row['source_name']} {row['characteristics']}".lower()

        if any(ek in text for ek in excl_kws):
            labels.append(-1)
            continue

        # GSE44861-style title suffix: "<id> N" = normal, "<id> T" = tumor.
        # Check only the last token so patient IDs like "10167" don't misfire.
        tokens = title.rsplit(None, 1)
        if len(tokens) == 2 and tokens[0].rstrip("0123456789 ") == "":
            suffix = tokens[1].upper()
            if suffix == "N":
                labels.append(0)
                continue
            elif suffix == "T":
                labels.append(1)
                continue

        if any(w in text for w in ["non-cancerous", "non-tumor", "non-tumour", "normal mucosa", "healthy", "mucosa_normal", "normal"]):
            if not any(w in row["title"].lower() for w in ["tumor", "tumour", "adenocarcinoma", "carcinoma"]):
                labels.append(0)
                continue

        is_ctrl = any(ck in text for ck in ctrl_kws)
        is_case = any(tk in text for tk in case_kws)

        if is_ctrl and not is_case:
            labels.append(0)
        elif is_case and not is_ctrl:
            labels.append(1)
        elif is_ctrl and is_case:
            if any(w in row["title"].lower() for w in ["normal", "mucosa", "non-cancerous", "adjacent", "healthy"]):
                labels.append(0)
            else:
                labels.append(1)
        else:
            labels.append(-1)

    return pd.Series(labels, index=meta_df.index)

def load_real_geo_cohorts(config: dict, raw_dir: str = "data/raw") -> tuple:
    """Loads and harmonizes the 4 real NCBI GEO Colorectal Cancer cohorts."""
    annot_cache = {}
    cohorts = {}
    min_per_grp = config["cohorts"].get("min_samples_per_group", 8)

    for gse_id, spec in config["cohorts"]["geo_accessions"].items():
        print(f"[GEO Loader] Parsing {gse_id}...")
        annot_file = spec["platform_annot"]
        if annot_file not in annot_cache:
            annot_cache[annot_file] = parse_gpl_annotation(annot_file, raw_dir=raw_dir)
        probe_map = annot_cache[annot_file]

        expr_probes, meta_df = parse_series_matrix(gse_id, raw_dir=raw_dir)
        expr_probes = expr_probes.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        mapped_symbols = expr_probes.index.map(lambda p: probe_map.get(str(p), None))
        valid_probes = mapped_symbols.notna()
        expr_mapped = expr_probes.loc[valid_probes].copy()
        expr_mapped["GeneSymbol"] = mapped_symbols[valid_probes]
        expr_genes = expr_mapped.groupby("GeneSymbol").mean()

        expr_df = expr_genes.T

        if np.nanpercentile(expr_df.values, 99) > 100.0:
            expr_df = np.log2(np.maximum(expr_df, 0.0) + 1.0)

        common_samples = [s for s in expr_df.index if s in meta_df.index]
        expr_df = expr_df.loc[common_samples]
        meta_df = meta_df.loc[common_samples]

        meta_df["condition"] = assign_phenotype_labels(meta_df, spec)
        meta_df["assay"] = spec["assay"]
        meta_df["batch"] = meta_df["instrument"].replace("", "Batch_1")

        cohort_median_profile = expr_df.median(axis=0).fillna(0).values
        sample_corrs = [
            np.corrcoef(expr_df.iloc[i].fillna(0).values, cohort_median_profile)[0, 1]
            for i in range(len(expr_df))
        ]
        meta_df["quality_score"] = pd.Series(sample_corrs, index=expr_df.index).rank(pct=True).values

        n_cases = int((meta_df["condition"] == 1).sum())
        n_ctrls = int((meta_df["condition"] == 0).sum())
        print(f"  -> {gse_id}: {n_cases} CRC Cases, {n_ctrls} Normal Controls, {expr_df.shape[1]} Mapped HGNC Genes.")

        if n_cases >= min_per_grp and n_ctrls >= min_per_grp:
            cohorts[gse_id] = {"expr": expr_df, "meta": meta_df, "assay": spec["assay"]}
        else:
            raise RuntimeError(f"{gse_id} failed group count check (Cases={n_cases}, Controls={n_ctrls}).")

    gmt_cache = os.path.join("data/gene_sets", "MSigDB_Hallmark_2020.pkl")
    os.makedirs("data/gene_sets", exist_ok=True)
    if os.path.exists(gmt_cache):
        gene_sets = pd.read_pickle(gmt_cache)
    else:
        import gseapy as gp
        print("[MSigDB Loader] Fetching authentic MSigDB_Hallmark_2020 gene sets...")
        gene_sets = gp.get_library(name=config["cohorts"]["msigdb_library"], organism="Human")
        pd.to_pickle(gene_sets, gmt_cache)

    return cohorts, gene_sets

def run_quality_control(cohorts: dict, fdr_alpha: float = 0.05) -> tuple:
    """Executes Section 7.2 Quality-Control Checklist & Section 7.3 Quality Imbalance Audit."""
    qc_summary_rows = []
    pca_records = {}
    filtered_cohorts = {}

    for c_name, data in cohorts.items():
        expr = data["expr"].copy()
        meta = data["meta"].copy()
        assay = data["assay"]

        n_initial = len(expr)
        valid_mask = meta["condition"].isin([0, 1]) & (expr.isna().mean(axis=1) < 0.05)
        n_excluded = int((~valid_mask).sum())
        expr = expr.loc[valid_mask]
        meta = meta.loc[valid_mask]

        if expr.isna().any().any():
            expr = expr.fillna(expr.median(axis=0))

        # Coerce to float to avoid NaN var from object dtype
        expr = expr.astype(float)

        gene_vars = expr.var(axis=0).fillna(0.0)
        # Safe filter: keep genes with variance > 0, plus top 90% by variance.
        # The old percentile-based `>` could return an empty set on homogeneous cohorts.
        nonzero_var = gene_vars[gene_vars > 0]
        if len(nonzero_var) >= 10:
            thresh = float(np.percentile(nonzero_var.values, 10))
            keep_genes = (gene_vars > thresh) & (gene_vars > 0)
        else:
            keep_genes = gene_vars > 0
        if keep_genes.sum() == 0:
            # Last-resort fallback: keep everything (a QC warning, not a crash)
            print(f"  [QC WARN] {c_name}: no genes passed variance filter; keeping all {expr.shape[1]} genes")
            keep_genes = pd.Series(True, index=expr.columns)
        expr_filtered = expr.loc[:, keep_genes]

        # Guard: PCA needs at least 2 samples and 2 genes
        if expr_filtered.shape[1] < 2 or expr_filtered.shape[0] < 2:
            raise RuntimeError(
                f"{c_name}: not enough data for PCA after QC "
                f"(samples={expr_filtered.shape[0]}, genes={expr_filtered.shape[1]}). "
                f"Check raw expression matrix."
            )

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(expr_filtered.values)
        pca_df = pd.DataFrame(coords, index=expr_filtered.index, columns=["PC1", "PC2"])
        pca_df["condition"] = meta["condition"].values
        pca_df["batch"] = meta["batch"].values
        pca_df["quality_score"] = meta["quality_score"].values
        pca_df["var_exp_1"] = pca.explained_variance_ratio_[0]
        pca_df["var_exp_2"] = pca.explained_variance_ratio_[1]
        pca_records[c_name] = pca_df

        q_cases = meta.loc[meta["condition"] == 1, "quality_score"]
        q_ctrl = meta.loc[meta["condition"] == 0, "quality_score"]
        _, qual_p = mannwhitneyu(q_cases, q_ctrl) if (len(q_cases) > 0 and len(q_ctrl) > 0) else (0, 1.0)

        notes = (
            f"Initial N={n_initial}, Excluded={n_excluded}; "
            f"Retained HGNC genes={expr_filtered.shape[1]}; Quality-imbalance p={qual_p:.3f}"
        )
        qc_summary_rows.append({
            "Cohort / accession": c_name,
            "Assay": "Log2 Normalized Expression",
            "Cases": int((meta["condition"] == 1).sum()),
            "Controls": int((meta["condition"] == 0).sum()),
            "Expression scale": "log2 intensity",
            "Covariates, sample exclusions, and inclusion notes": notes
        })

        filtered_cohorts[c_name] = {
            "expr": expr_filtered,
            "meta": meta,
            "assay": assay,
            "qual_imbalance_p": qual_p
        }

    common_genes = sorted(list(set.intersection(*[set(d["expr"].columns) for d in filtered_cohorts.values()])))
    print(f"[QC Harmonization] Common HGNC gene universe across all {len(filtered_cohorts)} cohorts: {len(common_genes)} genes.")
    for c_name in filtered_cohorts:
        filtered_cohorts[c_name]["expr"] = filtered_cohorts[c_name]["expr"][common_genes]

    return filtered_cohorts, pd.DataFrame(qc_summary_rows), pca_records
