"""02. Data quality checks on a clinic visit log.

Reusable pandas functions that flag common data quality problems:
  * missing values
  * bad dates (impossible, wrongly formatted, or not a date at all)
  * numeric outliers (z-score rule)
  * inconsistent units in a duration column

The functions are tested on small hand-built records first, then applied to the
synthetic log in data/nurse_log_sample.csv (see data/generate_nurse_log.py).

Usage:
    python 02_data_quality_checks.py [path/to/log.csv]
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/nurse_log_sample.csv")
DATE_FORMATS = ("%m/%d/%y", "%m/%d/%Y", "%m-%d-%Y")
# X = 3 standard deviations is the usual rule of thumb: for roughly normal data it
# flags about 0.3% of points, so it highlights genuine anomalies rather than normal spread.
OUTLIER_STD_DEVS = 3.0


def count_missing(data: pd.DataFrame, column: str) -> int:
    """Number of missing (NaN/None) values in a column."""
    return int(data[column].isna().sum())


def is_valid_date(value) -> bool:
    """True when the value parses as a real calendar date in one of DATE_FORMATS.

    strptime already rejects impossible dates such as 2/30 or 9/31 and handles
    leap years, so no manual month-length logic is needed.
    """
    if not isinstance(value, str):
        return False
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def count_bad_dates(data: pd.DataFrame, column: str) -> int:
    """Number of values in a column that are not valid dates (missing counts as bad)."""
    return int((~data[column].map(is_valid_date)).sum())


def count_outliers(data: pd.DataFrame, column: str, n_std: float = OUTLIER_STD_DEVS) -> int:
    """Number of values more than n_std standard deviations from the column mean.

    Returns 0 for non-numeric columns, where the standard deviation is undefined.
    Missing values are ignored.
    """
    series = data[column]
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    values = series.dropna()
    std = values.std()
    if len(values) < 2 or std == 0 or np.isnan(std):
        return 0
    return int((np.abs(values - values.mean()) > n_std * std).sum())


def find_inconsistent_units(data: pd.DataFrame, column: str, unit: str = "minutes") -> pd.Series:
    """Entries whose text does not end in the expected unit (case-insensitive).

    Missing values are excluded because they are reported by count_missing.
    Parameters are the DataFrame, the column name, and the standard unit, so the same
    function works for any column that stores text such as "15 minutes".
    """
    text = data[column].dropna().astype(str).str.strip().str.lower()
    return data.loc[text.index[~text.str.endswith(unit)], column]


def generate_report(data: pd.DataFrame) -> str:
    """Summarize all checks per column."""
    lines = ["MISSING VALUES"]
    lines += [f"  {col}: {count_missing(data, col)}" for col in data.columns]

    lines += ["", "OUTLIERS (beyond %.0f standard deviations)" % OUTLIER_STD_DEVS]
    numeric_cols = data.select_dtypes(include="number").columns
    lines += [f"  {col}: {count_outliers(data, col)}" for col in numeric_cols]

    lines += ["", "BAD DATES", f"  date: {count_bad_dates(data, 'date')}"]

    bad_units = find_inconsistent_units(data, "time_spent")
    lines += ["", "INCONSISTENT UNITS", f"  time_spent: {len(bad_units)} values not in minutes"]
    lines += [f"    {v!r}" for v in bad_units.unique()]
    return "\n".join(lines)


def run_tests() -> None:
    """Hand-built records where the correct answer is known in advance.

    Good test cases cover one problem each plus the boundaries: a leap day that
    exists (2/29/96) next to one that does not (2/29/94), a day 31 in a 30-day
    month, a different valid format, and a value that is not a date at all.
    """
    dates = pd.DataFrame({"date": ["2/29/96", "2/29/94", "9/31/94", "4-15-1994",
                                   "7/4/94", "Potter", None]})
    assert count_bad_dates(dates, "date") == 4
    assert count_missing(dates, "date") == 1

    nums = pd.DataFrame({"x": [10] * 20 + [1000], "label": ["a"] * 21})
    assert count_outliers(nums, "x") == 1
    assert count_outliers(nums, "label") == 0

    times = pd.DataFrame({"t": ["10 minutes", "1 hour", "15 MINUTES", "20 minute", None]})
    assert len(find_inconsistent_units(times, "t")) == 2


def main() -> None:
    run_tests()
    print("Self-tests passed.\n")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"{path} not found. Run: python data/generate_nurse_log.py")
    print(generate_report(pd.read_csv(path)))


if __name__ == "__main__":
    main()
