"""Remove direct identifiers from the raw dataset.

`Name` and `SSN` are never used by the model, but shipping them in a public
repository makes the data indistinguishable from a real PII leak: 88.6% of the
SSN strings in the source file are structurally valid (plausible area, group and
serial codes). `Customer_ID` is kept -- it is a pseudonymous key and the grouped
train/test split depends on it.

Usage:
    python scripts/strip_pii.py <source.csv> <destination.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DIRECT_IDENTIFIERS = ["Name", "SSN"]


def strip_pii(src: Path, dst: Path) -> None:
    # Read every field as text. Parsing to float and writing back changes the
    # last bit of ~18,000 values through the repr round-trip; harmless for
    # modelling, but there is no reason for a column-removal step to alter
    # numbers at all.
    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_filter=False)
    present = [c for c in DIRECT_IDENTIFIERS if c in df.columns]
    if not present:
        print(f"No direct identifiers found in {src}; copying unchanged.")
    else:
        print(f"Dropping {present} ({len(df):,} rows)")
    df.drop(columns=present).to_csv(dst, index=False)
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    strip_pii(Path(sys.argv[1]), Path(sys.argv[2]))
