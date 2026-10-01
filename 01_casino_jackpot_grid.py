"""01. Casino jackpot grid: working with 3D NumPy arrays.

A casino has a 6x6 grid of gambling machines. Each machine pays out between 0 and
10 jackpots per hour. A value of 0 means the machine was out of order that hour.

This script covers:
  * simulating 12 hours of data as a (hours, rows, cols) array
  * saving and loading the array (JSON text and NumPy binary formats)
  * per-machine averages, including an average that ignores zero hours
  * disabling "too hot" machines and re-running the simulation

Usage:
    python 01_casino_jackpot_grid.py            # default seed 42
    python 01_casino_jackpot_grid.py --seed 7
"""
import argparse
import json
from pathlib import Path

import numpy as np

GRID_SHAPE = (6, 6)
HOURS = 12
MAX_JACKPOTS = 10
TOO_HOT_THRESHOLD = 7
OUT_DIR = Path("data")


def simulate_hours(rng: np.random.Generator, hours: int = HOURS) -> np.ndarray:
    """Return an array of shape (hours, 6, 6) of random integers in [0, 10]."""
    return rng.integers(0, MAX_JACKPOTS + 1, size=(hours, *GRID_SHAPE))


def save_grid(path: Path, data: np.ndarray) -> None:
    """Save the array. The extension selects the format: .json (text) or .npy (binary)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data.tolist()))
    elif path.suffix == ".npy":
        np.save(path, data)
    else:
        raise ValueError(f"Unsupported extension '{path.suffix}'. Use .json or .npy")


def load_grid(path: Path) -> np.ndarray:
    """Load an array written by save_grid."""
    path = Path(path)
    if path.suffix == ".json":
        return np.array(json.loads(path.read_text()))
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported extension '{path.suffix}'. Use .json or .npy")


def average_per_machine(data: np.ndarray) -> np.ndarray:
    """Mean jackpots per hour for each machine, averaged over the hour axis."""
    return data.mean(axis=0)


def average_nonzero_per_machine(data: np.ndarray) -> np.ndarray:
    """Mean over hours in which the machine was working (value > 0).

    A machine that was out of order for every hour gets -1.
    """
    working_hours = (data > 0).sum(axis=0)
    totals = data.sum(axis=0)
    # np.divide with `where` avoids a divide-by-zero warning for dead machines.
    means = np.divide(totals, working_hours, out=np.full(totals.shape, -1.0),
                      where=working_hours > 0)
    return means


def disable_hot_machines(data: np.ndarray, averages: np.ndarray,
                         threshold: float = TOO_HOT_THRESHOLD):
    """Zero out every hour for machines whose average is at or above the threshold.

    Returns the modified copy and the (row, col) indices of the disabled machines.
    """
    hot_mask = averages >= threshold
    disabled = data.copy()
    disabled[:, hot_mask] = 0  # boolean mask selects individual machines, not whole rows
    return disabled, np.argwhere(hot_mask)


def self_check() -> None:
    """Small deterministic tests for the helper functions."""
    sample = np.zeros((4, 6, 6), dtype=int)
    sample[:, 0, 0] = [2, 0, 4, 0]     # working twice, mean of working hours = 3
    sample[:, 1, 1] = [10, 10, 10, 10]
    assert average_per_machine(sample)[0, 0] == 1.5
    nonzero = average_nonzero_per_machine(sample)
    assert nonzero[0, 0] == 3.0
    assert nonzero[5, 5] == -1.0       # never working
    disabled, idx = disable_hot_machines(sample, average_per_machine(sample))
    assert idx.tolist() == [[1, 1]]
    assert disabled[:, 1, 1].sum() == 0 and disabled[:, 0, 0].sum() == 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    self_check()

    # Part 1: simulate
    first_run = simulate_hours(rng)
    print(f"Simulated array shape (hours, rows, cols): {first_run.shape}")
    print("Hour 1 grid:")
    print(first_run[0], "\n")

    # Part 2 and 3: round trip through both file formats
    for name in ("jackpots.json", "jackpots.npy"):
        save_grid(OUT_DIR / name, first_run)
        reloaded = load_grid(OUT_DIR / name)
        print(f"{name}: round trip identical -> {np.array_equal(first_run, reloaded)}")

    # Part 4: averages
    averages = average_per_machine(first_run)
    print("\nAverage jackpots per hour, per machine:")
    print(np.round(averages, 2))

    # Part 5: averages that ignore out-of-order hours
    print("\nAverage over working hours only (-1 means never worked):")
    print(np.round(average_nonzero_per_machine(first_run), 2))

    # Part 6: disable hot machines and simulate again
    second_run, hot_indices = disable_hot_machines(simulate_hours(rng), averages)
    print(f"\nMachines averaging >= {TOO_HOT_THRESHOLD} in the first run (row, col):")
    print(hot_indices.tolist() if len(hot_indices) else "none")
    print("\nSecond run, hot machines forced to 0 in every hour. Total jackpots per hour:")
    print(second_run.sum(axis=(1, 2)))


if __name__ == "__main__":
    main()
