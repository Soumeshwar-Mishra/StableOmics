#!/usr/bin/env python3
"""Direct NCBI Static-FTP Downloader for Real Colorectal Cancer Cohorts & Gene Annotations."""
import os, sys, time, gzip, urllib.request, urllib.error

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; StableOmics)"}

def verify_gzip(path, min_bytes=10240):
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False
    try:
        with gzip.open(path, "rb") as f:
            chunk = f.read(65536)
            if not chunk or chunk.lstrip().startswith(b"<") or chunk.lstrip().startswith(b"<!DOCTYPE"):
                return False
        return True
    except Exception:
        return False

def download(url, dest, min_bytes=10240, retries=4):
    if verify_gzip(dest, min_bytes=min_bytes):
        print(f"  [CACHED] {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.2f} MB)")
        return True
    for attempt in range(1, retries + 1):
        try:
            print(f"  -> {os.path.basename(dest)} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as out:
                total = r.headers.get("Content-Length")
                total = int(total) if total else None
                got = 0
                while True:
                    buf = r.read(65536)
                    if not buf: break
                    out.write(buf); got += len(buf)
                    if total:
                        sys.stdout.write(f"\r     {got/1e6:.2f}/{total/1e6:.2f} MB ({100*got/total:.1f}%)")
                        sys.stdout.flush()
                sys.stdout.write("\n")
            if verify_gzip(dest, min_bytes=min_bytes):
                print(f"  [OK] {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.2f} MB)")
                return True
            if os.path.exists(dest): os.remove(dest)
        except urllib.error.HTTPError as e:
            print(f"\n  [HTTP {e.code}]")
            if os.path.exists(dest): os.remove(dest)
            if e.code == 404: return False
        except Exception as e:
            print(f"\n  [WARN] {e}")
            if os.path.exists(dest): os.remove(dest)
        time.sleep(2 * attempt)
    return False

def main():
    print("=== [1/3] NCBI GPL Platform Annotations ===")
    platforms = {
        "GPL570.annot.gz":   "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
        "GPL13667.annot.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13667/annot/GPL13667.annot.gz",
        "GPL3921.annot.gz":  "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL3nnn/GPL3921/annot/GPL3921.annot.gz",
    }
    for fn, url in platforms.items():
        download(url, os.path.join(RAW_DIR, fn), min_bytes=50_000)

    print("\n=== [2/3] GEO Series Matrices ===")
    cohorts = ["GSE44076", "GSE44861", "GSE20916", "GSE32323"]
    verified = []
    for gse in cohorts:
        prefix = gse[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/matrix/{gse}_series_matrix.txt.gz"
        dest = os.path.join(RAW_DIR, f"{gse}_series_matrix.txt.gz")
        if download(url, dest, min_bytes=500_000):
            verified.append(gse)

    print(f"\n=== [3/3] Verified {len(verified)}/{len(cohorts)} cohorts ===")
    for fn in sorted(os.listdir(RAW_DIR)):
        print(f"  - {fn:40s} ({os.path.getsize(os.path.join(RAW_DIR, fn))/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
