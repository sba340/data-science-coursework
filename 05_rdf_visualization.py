"""05. Visualizing a radial distribution function (RDF) with Matplotlib.

The RDF, g(r), describes how atomic density varies with distance r from a reference
atom, relative to a uniform distribution. It is dimensionless: g(r) = 0 where atoms
cannot be (too close), peaks at preferred bond distances, and tends toward 1 at long range.
Such curves come out of molecular dynamics simulations (for example ab initio MD of an
amorphous material) and are a standard way to read off bond lengths and local order.

The script makes five figures from a two-column text file (r in angstroms, g(r)):
  1. line plot      2. scatter plot      3. histogram of g(r) values
  4. annotated peak plot (area plot with detected peaks)      5. two-panel figure

If no file is given, a synthetic RDF-like curve is generated so the script runs anywhere.

Usage:
    python 05_rdf_visualization.py                       # synthetic demo data
    python 05_rdf_visualization.py my_rdf.dat            # your own two-column file
    python 05_rdf_visualization.py my_rdf.dat --show     # also open figure windows
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

FIG_DIR = Path("figures")


def synthetic_rdf(seed: int = 1):
    """A plausible amorphous-material RDF: excluded core, a few peaks, decay to 1, noise."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, 8.0, 400)
    g = np.ones_like(r)
    for center, height, width in [(2.4, 4.0, 0.12), (3.1, 1.8, 0.2), (4.6, 0.9, 0.35), (6.2, 0.4, 0.5)]:
        g += height * np.exp(-0.5 * ((r - center) / width) ** 2) * np.exp(-(r - 2.4) / 6)
    g[r < 1.9] = 0.0
    g += rng.normal(0, 0.03, r.size) * (r >= 1.9)
    return r, np.clip(g, 0, None)


def load_rdf(path: str):
    """Read a whitespace-separated file whose first two columns are r and g(r)."""
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("Expected at least two columns: r and g(r)")
    return data[:, 0], data[:, 1]


def finish(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datafile", nargs="?", help="two-column file: r (angstrom), g(r)")
    parser.add_argument("--show", action="store_true", help="display the figures")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")  # no display needed when only saving files
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    r, g = load_rdf(args.datafile) if args.datafile else synthetic_rdf()
    source = args.datafile or "synthetic demo data"
    xlabel, ylabel = "r (Å)", "g(r)"

    # 1. Line plot: the standard view for an RDF because r is a continuous axis.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, g, label=f"g(r), {source}")
    ax.set(xlabel=xlabel, ylabel=ylabel, title="Radial distribution function")
    ax.legend(); ax.grid(alpha=0.3)
    finish(fig, "1_line")

    # 2. Scatter plot: shows the individual sampled points and the noise level.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r, g, s=10, color="tab:blue", label="sampled points")
    ax.set(xlabel=xlabel, ylabel=ylabel, title="RDF, individual samples")
    ax.legend(); ax.grid(alpha=0.3)
    finish(fig, "2_scatter")

    # 3. Histogram of g(r) values. Bin edges use the Freedman-Diaconis rule,
    #    width = 2 * IQR * n^(-1/3), which adapts to both spread and sample size.
    edges = np.histogram_bin_edges(g, bins="fd")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g, bins=edges, edgecolor="black", alpha=0.75)
    ax.set(xlabel=ylabel, ylabel="Number of r samples",
           title=f"Distribution of g(r) values ({len(edges) - 1} bins, Freedman-Diaconis)")
    ax.grid(axis="y", alpha=0.3)
    finish(fig, "3_histogram")

    # 4. Peaks: an area plot with detected maxima marked and labeled by bond distance.
    #    Prominence filters out noise; 10% of the curve's range is a reasonable start.
    peaks, _ = find_peaks(g, prominence=0.1 * (g.max() - g.min()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(r, g, alpha=0.3, label="g(r)")
    ax.plot(r, g)
    ax.plot(r[peaks], g[peaks], "o", color="tab:red", label="detected peaks")
    for p in peaks:
        ax.annotate(f"{r[p]:.2f} Å", (r[p], g[p]), textcoords="offset points",
                    xytext=(5, 8), fontsize=9)
    ax.set(xlabel=xlabel, ylabel=ylabel, title="RDF with detected peak positions")
    ax.legend(); ax.grid(alpha=0.3)
    finish(fig, "4_peaks")
    print("Peak positions (Å):", ", ".join(f"{r[p]:.2f}" for p in peaks) or "none found")

    # 5. Two subplots: curve and value distribution side by side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(r, g); ax1.set(xlabel=xlabel, ylabel=ylabel, title="g(r) versus r"); ax1.grid(alpha=0.3)
    ax2.hist(g, bins=edges, edgecolor="black", color="skyblue")
    ax2.set(xlabel=ylabel, ylabel="Number of r samples", title="Histogram of g(r)")
    ax2.grid(axis="y", alpha=0.3)
    finish(fig, "5_subplots")

    print(f"Saved 5 figures to {FIG_DIR}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
