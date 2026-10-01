"""Generate a synthetic, deliberately messy infirmary log for scripts 02 and 03.

All names and values are invented. The file mimics common data quality problems
(missing values, impossible dates, mixed units, typos, outliers, inconsistent IDs)
so that the quality checks and cleaning steps have something realistic to work on.

Usage:
    python data/generate_nurse_log.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "nurse_log_sample.csv"

COLUMNS = [
    "medical_record_number", "first_name", "last_name", "visit_id", "date",
    "time_spent", "height(cm)", "weight(kg)", "charge", "supplies_used",
]

PATIENTS = [
    ("Alder", "Finch", 10231), ("Briar", "Hollis", 20418), ("Cedric", "Marsh", 30552),
    ("Dara", "Quill", 40876), ("Elio", "Thorne", 51009), ("Fenna", "Vale", 61347),
    ("Gale", "Wren", 71562), ("Hollis", "Ash", 81790), ("Isla", "Brook", 92045),
    ("Jory", "Crane", 12984),
]


def build_log(n_rows: int = 70, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic log with injected data quality problems."""
    rng = np.random.default_rng(seed)
    rows = []
    for visit_id in range(1, n_rows + 1):
        first, last, mrn = PATIENTS[rng.integers(len(PATIENTS))]
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        rows.append({
            "medical_record_number": mrn,
            "first_name": first,
            "last_name": last,
            "visit_id": visit_id,
            "date": f"{month}/{day}/94",
            "time_spent": f"{int(rng.choice([5, 10, 15, 20, 25, 30]))} minutes",
            "height(cm)": int(rng.integers(150, 192)),
            "weight(kg)": int(rng.integers(40, 110)),
            "charge": round(float(rng.uniform(10, 120)), 2),
            "supplies_used": int(rng.integers(0, 15)),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)

    # Inject problems at fixed, reproducible positions.
    df.loc[[3, 17, 41], "time_spent"] = np.nan                      # missing durations
    df.loc[[5, 22], "time_spent"] = ["1 hour", "2 hours"]           # hours instead of minutes
    df.loc[[9, 30], "time_spent"] = ["15 minute", "20 minuets"]     # unit typos
    df.loc[[12], "date"] = "2/30/94"                                # impossible date
    df.loc[[25], "date"] = "not a date"                             # not a date at all
    df.loc[[33], "date"] = np.nan                                   # missing date
    df.loc[[38], "date"] = "1994-07-04"                             # wrong format
    df.loc[[14], "height(cm)"] = 2500                               # typo outlier
    df.loc[[47], "height(cm)"] = 610                                # unit confusion outlier
    df.loc[[6, 28], "supplies_used"] = [-4, 250]                    # impossible supply counts
    df.loc[[19, 52], "supplies_used"] = np.nan                      # missing supply counts
    df.loc[[8], "medical_record_number"] = 99999                    # inconsistent ID
    df.loc[[44], "medical_record_number"] = 15689
    df.loc[[11], "first_name"] = df.loc[11, "first_name"].upper()   # case variation
    df.loc[[36], "last_name"] = df.loc[36, "last_name"].lower()
    df.loc[[55], "weight(kg)"] = np.nan                             # missing weight
    return df


if __name__ == "__main__":
    log = build_log()
    log.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(log)} rows to {OUT_PATH}")
