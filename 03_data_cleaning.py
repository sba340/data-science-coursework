"""03. Data cleaning pipeline for the clinic visit log.

Each step is a small function that returns a new DataFrame, applied in an order
where earlier steps do not break later ones:

  1. drop rows with a missing time_spent
  2. drop rows with an impossible height (> 250 cm)
  3. replace bad supplies_used values with the mean of the valid ones
  4. split time_spent into a numeric value and a unit, converting hours to minutes
  5. replace bad dates with January 1st, 1994
  6. make medical record numbers consistent within each patient name
  7. report aggregates (mean visit length, busiest month)

Usage:
    python 03_data_cleaning.py [path/to/log.csv]
"""
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/nurse_log_sample.csv")
MAX_HEIGHT_CM = 250
MAX_SUPPLIES = 100
FALLBACK_DATE = pd.Timestamp("1994-01-01")
DATE_FORMATS = ("%m/%d/%y", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d")


def drop_missing_time(df: pd.DataFrame) -> pd.DataFrame:
    """Remove visits with no recorded duration."""
    return df.dropna(subset=["time_spent"]).copy()


def drop_impossible_heights(df: pd.DataFrame, limit: int = MAX_HEIGHT_CM) -> pd.DataFrame:
    """Remove rows whose height exceeds the limit. Missing heights are kept."""
    return df[~(df["height(cm)"] > limit)].copy()


def fix_supplies_used(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing, negative, and over-limit supply counts with the valid mean."""
    df = df.copy()
    valid = df["supplies_used"].between(0, MAX_SUPPLIES)  # NaN compares False
    valid_mean = df.loc[valid, "supplies_used"].mean()
    df.loc[~valid, "supplies_used"] = valid_mean
    return df


def normalize_time_spent(df: pd.DataFrame) -> pd.DataFrame:
    """Split "15 minute" style text into time_spent (minutes, float) and time_spent_unit.

    Any unit starting with "h" is treated as hours and converted to minutes; any unit
    starting with "m" (minute, minutes, and misspellings such as minuets) becomes "minutes".
    Entries that do not match "<number> <unit>" become NaN.
    """
    df = df.copy()
    parts = df["time_spent"].astype(str).str.strip().str.extract(r"^([\d.]+)\s*([A-Za-z]+)$")
    value = pd.to_numeric(parts[0], errors="coerce")
    unit = parts[1].str.lower()
    is_hours = unit.str.startswith("h", na=False)
    is_minutes = unit.str.startswith("m", na=False)
    df["time_spent"] = value.where(~is_hours, value * 60)
    df.loc[~(is_hours | is_minutes), "time_spent"] = float("nan")
    df["time_spent_unit"] = "minutes"
    return df


def parse_date(text) -> pd.Timestamp:
    """Parse a date string using the accepted formats; return NaT if none fit."""
    if not isinstance(text, str):
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(text.strip(), format=fmt)
        except ValueError:
            continue
    return pd.NaT


def replace_bad_dates(df: pd.DataFrame):
    """Parse the date column and replace unparseable entries with the fallback date.

    Returns the cleaned frame and the number of dates replaced.
    """
    df = df.copy()
    parsed = df["date"].map(parse_date)
    n_bad = int(parsed.isna().sum())
    df["date"] = parsed.fillna(FALLBACK_DATE)
    return df, n_bad


def make_ids_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """Within each first/last name pair (ignoring case), use the most common record number.

    Names are standardized to title case so "FINCH" and "Finch" are one patient.
    Ties are broken toward the smaller number, which is deterministic.
    """
    df = df.copy()
    df["first_name"] = df["first_name"].str.strip().str.title()
    df["last_name"] = df["last_name"].str.strip().str.title()
    df["medical_record_number"] = (
        df.groupby(["first_name", "last_name"])["medical_record_number"]
        .transform(lambda ids: ids.mode().min())
    )
    return df


def clean(df: pd.DataFrame):
    """Run every step in order. Returns the cleaned frame and a dict of audit counts."""
    audit = {"rows_start": len(df)}
    df = drop_missing_time(df)
    audit["after_drop_missing_time"] = len(df)
    df = drop_impossible_heights(df)
    audit["after_drop_heights"] = len(df)
    df = fix_supplies_used(df)
    df = normalize_time_spent(df)
    df, audit["dates_replaced"] = replace_bad_dates(df)
    df = make_ids_consistent(df)
    return df, audit


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"{path} not found. Run: python data/generate_nurse_log.py")
    raw = pd.read_csv(path)
    cleaned, audit = clean(raw)

    print("Audit trail:", audit)
    print(f"\nCleaned data ({len(cleaned)} rows), first 8:")
    print(cleaned.head(8).to_string(index=False))

    print(f"\nMean visit length: {cleaned['time_spent'].mean():.2f} minutes")

    in_1994 = cleaned[cleaned["date"].dt.year == 1994]
    monthly = in_1994.groupby(in_1994["date"].dt.month_name())["time_spent"].sum()
    print(f"Busiest month in 1994: {monthly.idxmax()} ({monthly.max():.0f} minutes logged)")

    # Remaining concerns when reporting monthly totals to administrators:
    #  * Replaced dates are all set to January 1st, so January is inflated by bad data.
    #  * Dropped rows (missing time, impossible height) silently reduce monthly totals.
    #  * Time values with unrecognized units become NaN and are skipped by sum().
    #  * A patient who changed name spelling could still be split across record numbers.
    print(f"\nNote: {audit['dates_replaced']} dates were replaced with 1/1/1994, "
          "which inflates January.")

    out = Path("data/nurse_log_clean.csv")
    cleaned.to_csv(out, index=False)
    print(f"Saved cleaned data to {out}")


if __name__ == "__main__":
    main()
