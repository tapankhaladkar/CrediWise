"""Download the UCI "Default of Credit Card Clients" dataset.

    python scripts/download_data.py

Fetches the archive from the UCI Machine Learning Repository, checks it
against a known SHA-256 so the data cannot silently change underneath the
results, and writes the spreadsheet out as data/credit_default.csv with the
original column names untouched. Renaming and cleaning happen in code
(src/crediwise/data.py), so the CSV stays a faithful copy of the source.

Source:  Yeh, I. (2009). Default of Credit Card Clients [Dataset].
         UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = ("https://archive.ics.uci.edu/static/public/350/"
       "default+of+credit+card+clients.zip")
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
SHEET_NAME = "default of credit card clients.xls"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "credit_default.csv"


def main() -> None:
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()

    digest = hashlib.sha256(archive).hexdigest()
    if digest != ARCHIVE_SHA256:
        sys.exit(f"Checksum mismatch: expected {ARCHIVE_SHA256}, got {digest}. "
                 "The published file has changed; do not train on it unverified.")
    print(f"Checksum OK ({digest[:12]})")

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        sheet = zf.read(SHEET_NAME)

    # Row 0 holds the generic labels X1..X23; the real names are on row 1.
    df = pd.read_excel(io.BytesIO(sheet), header=1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.relative_to(OUTPUT.parents[1])}: "
          f"{len(df):,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
