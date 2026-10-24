#!/usr/bin/env python3
"""
Executes Section 5.3 Novelty Verification Procedure.
Queries PubMed E-utilities for the 6 prespecified boolean strings and logs exact record counts and dates.
"""
import os
import time
import urllib.parse
import urllib.request
import json
import pandas as pd

QUERIES = [
    '"RNA-seq" AND "gene selection stability" AND cross-cohort',
    '"biomarker reproducibility" AND uncertainty AND transcriptomics',
    '"differential expression" AND subsampling AND sample size',
    '"stability selection" AND RNA-seq AND external validation',
    '"RNA-seq" AND quality imbalance AND reproducibility',
    'transcriptomics feature selection reproducibility independent cohort bootstrap'
]

def query_pubmed_count(term: str) -> int:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json"})
    try:
        with urllib.request.urlopen(f"{base_url}?{params}", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return int(data["esearchresult"]["count"])
    except Exception:
        return -1  # Offline fallback indicator

def main():
    os.makedirs("results/logs", exist_ok=True)
    records = []
    date_str = time.strftime("%Y-%m-%d")
    for idx, q in enumerate(QUERIES, 1):
        count = query_pubmed_count(q)
        records.append({
            "Query_ID": f"Q{idx}",
            "Search_Date": date_str,
            "Database": "PubMed (NCBI Entrez)",
            "Exact_Query": q,
            "Records_Found": count if count >= 0 else "Offline_Verify_Manually",
            "Screening_Criteria": "Bulk/scRNA-seq DE stability, cross-cohort validation, quality perturbation",
            "Closest_Prior_Benchmarks_Included": "Degen & Medo (2025) BootstrapSeq; Nakatsuka et al. (2025) SumRank"
        })
        time.sleep(0.4)

    df = pd.DataFrame(records)
    out_path = "results/logs/literature_search_log.csv"
    df.to_csv(out_path, index=False)
    print(f"[Novelty Verification] Saved reproducible search log to {out_path}")

if __name__ == "__main__":
    main()
