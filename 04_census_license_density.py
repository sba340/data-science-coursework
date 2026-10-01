"""04. Alcohol license density by census tract (APIs, geocoding, and merging).

Pipeline:
  1. read a CSV of active licenses (one row per license, with a street address)
  2. download tract-level population from the Census Bureau ACS 5-year API
  3. geocode each address to a census tract with the Census Geocoder API
  4. count licenses per tract and divide by population (licenses per 1,000 residents)
  5. rank tracts by rate and by raw count

The Census APIs are free and need no key. Geocoding is slow (about one request per
second is polite), so results are cached in data/census_tract_cache.json and a rerun
only geocodes new addresses.

Usage:
    python 04_census_license_density.py --licenses ActiveLicenses.csv
    python 04_census_license_density.py --licenses ActiveLicenses.csv --limit 50
    python 04_census_license_density.py --demo        # offline, synthetic data
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ACS_URL = "https://api.census.gov/data/2020/acs/acs5/profile"
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/address"
CACHE_PATH = Path("data/census_tract_cache.json")
TIMEOUT_S = 30


def fetch_tract_population(state_fips: str = "21", county_fips: str = "067") -> pd.DataFrame:
    """Total population per census tract. Defaults: Kentucky (21), Fayette County (067)."""
    params = {
        "get": "NAME,DP05_0001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    response = requests.get(ACS_URL, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    rows = response.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["population"] = pd.to_numeric(df["DP05_0001E"], errors="coerce")
    # The ACS uses large negative numbers (such as -666666666) for "not available".
    df = df[df["population"] > 0]
    return df[["tract", "NAME", "population"]]


def geocode_tract(street: str, city: str, state: str, session: requests.Session):
    """Return the 6-digit census tract code for an address, or None if no match."""
    params = {
        "street": street, "city": city, "state": state,
        "benchmark": "Public_AR_Census2020", "vintage": "Census2020_Census2020",
        "layers": "6", "format": "json",
    }
    try:
        response = session.get(GEOCODER_URL, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        match = response.json()["result"]["addressMatches"][0]
        return match["geographies"]["Census Tracts"][0]["TRACT"]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


def geocode_all(addresses, city: str, state: str, delay_s: float = 1.0) -> dict:
    """Geocode unique addresses, reusing and updating the on-disk cache."""
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    todo = [a for a in dict.fromkeys(addresses) if a not in cache]
    print(f"{len(cache)} addresses cached, {len(todo)} to geocode")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for i, address in enumerate(todo, start=1):
            cache[address] = geocode_tract(address, city, state, session)
            if i % 25 == 0:
                CACHE_PATH.write_text(json.dumps(cache))  # checkpoint
                print(f"  geocoded {i}/{len(todo)}")
            time.sleep(delay_s)
    CACHE_PATH.write_text(json.dumps(cache))
    return cache


def tract_license_table(licenses: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """One row per tract: license count, population, and licenses per 1,000 residents.

    `licenses` needs a `tract` column. Licenses without a tract are excluded, and tracts
    with no licenses do not appear. Aggregating before ranking matters: ranking the
    license-level rows would list the same tract once per license.
    """
    counts = licenses.dropna(subset=["tract"]).groupby("tract").size().rename("n_licenses")
    table = counts.reset_index().merge(population, on="tract", how="inner")
    table["licenses_per_1000"] = table["n_licenses"] / table["population"] * 1000
    return table


def run_tests() -> None:
    """Offline check of the aggregation logic with a known answer."""
    lic = pd.DataFrame({"tract": ["000100", "000100", "000100", "000200", None]})
    pop = pd.DataFrame({"tract": ["000100", "000200"], "NAME": ["A", "B"],
                        "population": [3000, 500]})
    table = tract_license_table(lic, pop).set_index("tract")
    assert table.loc["000100", "n_licenses"] == 3
    assert abs(table.loc["000100", "licenses_per_1000"] - 1.0) < 1e-9
    assert abs(table.loc["000200", "licenses_per_1000"] - 2.0) < 1e-9


def demo_inputs(seed: int = 0):
    """Synthetic licenses and tracts so the ranking code can run without any network."""
    rng = np.random.default_rng(seed)
    tracts = [f"{n:06d}" for n in range(100, 140)]
    population = pd.DataFrame({
        "tract": tracts,
        "NAME": [f"Census Tract {t}" for t in tracts],
        "population": rng.integers(1500, 9000, len(tracts)),
    })
    weights = rng.dirichlet(np.ones(len(tracts)) * 0.5)
    licenses = pd.DataFrame({"tract": rng.choice(tracts, size=400, p=weights)})
    return licenses, population


def show_rankings(table: pd.DataFrame, top_n: int = 20) -> None:
    cols = ["tract", "NAME", "n_licenses", "population", "licenses_per_1000"]
    print(f"\nTop {top_n} tracts by licenses per 1,000 residents:")
    print(table.nlargest(top_n, "licenses_per_1000")[cols].round(2).to_string(index=False))
    print(f"\nTop {top_n} tracts by total number of licenses:")
    print(table.nlargest(top_n, "n_licenses")[cols].round(2).to_string(index=False))
    # Interpretation: the rate ranking favors small-population tracts (such as downtown or
    # commercial districts) where a handful of licenses serve few residents. The count
    # ranking favors large or busy areas. A tract in both lists has high absolute supply and
    # high supply per resident. Caveat: license holders also serve commuters and visitors,
    # so per-resident rates overstate availability in areas with few residents.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--licenses", help="CSV of active licenses")
    parser.add_argument("--street-column", default="PremisesStreet")
    parser.add_argument("--skiprows", type=int, default=3, help="header rows to skip in the CSV")
    parser.add_argument("--city", default="Lexington")
    parser.add_argument("--state", default="KY")
    parser.add_argument("--limit", type=int, help="only geocode the first N licenses (for testing)")
    parser.add_argument("--demo", action="store_true", help="run offline on synthetic data")
    args = parser.parse_args()

    run_tests()
    if args.demo:
        licenses, population = demo_inputs()
    else:
        if not args.licenses:
            parser.error("--licenses is required unless --demo is used")
        licenses = pd.read_csv(args.licenses, skiprows=range(args.skiprows))
        if args.limit:
            licenses = licenses.head(args.limit)
        population = fetch_tract_population()
        lookup = geocode_all(licenses[args.street_column].dropna().astype(str),
                             args.city, args.state)
        licenses["tract"] = licenses[args.street_column].astype(str).map(lookup)
        matched = licenses["tract"].notna().mean()
        print(f"Geocoded {matched:.1%} of licenses to a census tract")

    table = tract_license_table(licenses, population)
    show_rankings(table)
    out = Path("data/license_density_by_tract.csv")
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
