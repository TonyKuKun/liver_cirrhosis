"""Plot publication-style pointwise vessel profiles from unified features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


DEFAULT_PATIENT_DIR = Path(
    r"F:\PCG data\dataset\test4all_sample\0013996314YangTingFu"
)
VESSEL_ORDER = ("mpv", "sv", "smv", "lpv", "rpv")
VESSEL_LABELS = {
    "mpv": "Main portal vein (MPV)",
    "sv": "Splenic vein (SV)",
    "smv": "Superior mesenteric vein (SMV)",
    "lpv": "Left portal vein (LPV)",
    "rpv": "Right portal vein (RPV)",
}
FEATURE_SPECS = (
    ("area", "Area", r"Area (mm$^2$)", "#1F4E79"),
    ("eq_diameter", "Equivalent diameter", "Diameter (mm)", "#C55A4A"),
    ("circularity", "Circularity", "Circularity", "#2F7D6D"),
    ("curvature", "Curvature", r"Curvature (mm$^{-1}$)", "#76528B"),
    ("perimeter", "Perimeter", "Perimeter (mm)", "#B07D2B"),
    ("inscribed_radius", "Inscribed radius", "Radius (mm)", "#3973A8"),
)


def _configure_style() -> None:
    """Apply a restrained, high-resolution scientific-figure style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 30,
            "axes.titlesize": 33,
            "axes.titleweight": "semibold",
            "axes.labelsize": 28.5,
            "axes.labelcolor": "#27313D",
            "axes.edgecolor": "#AEB7C2",
            "axes.linewidth": 1.4,
            "xtick.labelsize": 25.5,
            "ytick.labelsize": 25.5,
            "xtick.color": "#4D5966",
            "ytick.color": "#4D5966",
            "text.color": "#17202A",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def _load_pointwise(unified_path: Path) -> dict[str, dict]:
    if not unified_path.is_file():
        raise FileNotFoundError(f"Unified feature file not found: {unified_path}")

    with unified_path.open("r", encoding="utf-8") as handle:
        unified = json.load(handle)

    pointwise = unified.get("pointwise")
    if not isinstance(pointwise, dict):
        raise ValueError(f"Missing pointwise block in {unified_path}")
    return pointwise


def _sample_indices(n_values: int, n_plot_points: int) -> np.ndarray:
    if n_values < 2:
        raise ValueError("At least two profile points are required for plotting")
    count = min(n_values, max(2, n_plot_points))
    return np.unique(np.rint(np.linspace(0, n_values - 1, count)).astype(int))


def _profile_arrays(profile: dict, n_plot_points: int) -> tuple[np.ndarray, dict]:
    position = np.asarray(profile.get("position", []), dtype=float)
    if position.ndim != 1 or len(position) < 2:
        raise ValueError("Profile has no valid one-dimensional position channel")

    indices = _sample_indices(len(position), n_plot_points)
    x = position[indices] * 100.0
    channels = {}
    for key, *_ in FEATURE_SPECS:
        values = np.asarray(profile.get(key, []), dtype=float)
        if values.shape != position.shape:
            raise ValueError(
                f"Channel {key!r} has {len(values)} values; expected {len(position)}"
            )
        channels[key] = values[indices]
    return x, channels


def _format_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8DEE6", linewidth=1.2, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=6.0, width=1.2, pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(x=0.015)


def _format_mean(key: str, value: float) -> str:
    precision = 4 if key == "curvature" else 3 if key == "circularity" else 2
    return f"Mean = {value:.{precision}f}"


def plot_vessel(
    patient_id: str,
    vessel: str,
    profile: dict,
    output_path: Path,
    n_plot_points: int,
    dpi: int,
) -> None:
    x, channels = _profile_arrays(profile, n_plot_points)
    source_count = len(profile["position"])

    # The enlarged canvas preserves the 1.5 width/height ratio while giving the
    # three-times-larger typography enough room around every panel.
    fig, axes = plt.subplots(3, 2, figsize=(24.0, 16.0), sharex=True)
    panel_letters = "abcdef"

    for index, (ax, spec) in enumerate(zip(axes.flat, FEATURE_SPECS)):
        key, title, ylabel, color = spec
        y = channels[key]
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            raise ValueError(f"{vessel}: channel {key!r} has fewer than two finite values")

        x_valid = x[finite]
        y_valid = y[finite]
        full_values = np.asarray(profile[key], dtype=float)
        full_finite = full_values[np.isfinite(full_values)]
        mean_value = float(np.mean(full_finite))

        ax.axhline(
            mean_value,
            color="#5B6570",
            linewidth=1.8,
            linestyle=(0, (5, 3)),
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            x_valid,
            y_valid,
            color=color,
            linewidth=3.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        ax.set_title(f"({panel_letters[index]})  {title}", loc="left", pad=18)
        ax.text(
            1.0,
            1.045,
            _format_mean(key, mean_value),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=26.1,
            color="#5B6570",
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Normalized vessel position (%)")
        _format_axis(ax)
        ax.tick_params(axis="x", labelbottom=True)
        y_max = max(float(np.max(y_valid)), mean_value)
        if key == "circularity":
            y_upper = max(1.05, y_max * 1.04)
        else:
            y_upper = y_max * 1.10 if y_max > 0 else 1.0
        ax.set_ylim(0.0, y_upper)

    vessel_label = VESSEL_LABELS.get(vessel, vessel.upper())
    fig.suptitle(
        f"{vessel_label} | Pointwise profile features",
        x=0.08,
        y=0.965,
        ha="left",
        fontsize=48,
        fontweight="semibold",
        color="#111820",
    )
    fig.text(
        0.08,
        0.905,
        f"Patient {patient_id}  |  {len(x)} samples shown  |  Mean from all {source_count} positions",
        ha="left",
        va="top",
        fontsize=27.6,
        color="#5B6672",
    )
    fig.text(
        0.985,
        0.018,
        "Source: features/unified_features.json",
        ha="right",
        va="bottom",
        fontsize=24.6,
        color="#697481",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.98,
        bottom=0.12,
        top=0.79,
        wspace=0.30,
        hspace=0.90,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_patient(
    patient_dir: Path,
    output_dir: Path | None = None,
    n_plot_points: int = 50,
    dpi: int = 320,
) -> list[Path]:
    patient_dir = patient_dir.resolve()
    unified_path = patient_dir / "features" / "unified_features.json"
    output_dir = (output_dir or patient_dir / "picture").resolve()
    pointwise = _load_pointwise(unified_path)

    missing = [vessel for vessel in VESSEL_ORDER if vessel not in pointwise]
    if missing:
        raise ValueError(
            "Missing expected vessel profiles: " + ", ".join(name.upper() for name in missing)
        )

    _configure_style()
    outputs = []
    for vessel in VESSEL_ORDER:
        output_path = output_dir / f"{vessel}_pointwise_profiles.png"
        plot_vessel(
            patient_id=patient_dir.name,
            vessel=vessel,
            profile=pointwise[vessel],
            output_path=output_path,
            n_plot_points=n_plot_points,
            dpi=dpi,
        )
        outputs.append(output_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot six pointwise profile channels for five portal vessels."
    )
    parser.add_argument(
        "patient_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PATIENT_DIR,
        help="Patient directory containing features/unified_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <patient_dir>/picture)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=50,
        help="Number of evenly spaced points per curve (default: 50)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG resolution in dots per inch (default: 320)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = plot_patient(
        patient_dir=args.patient_dir,
        output_dir=args.output_dir,
        n_plot_points=args.points,
        dpi=args.dpi,
    )
    print(f"Generated {len(outputs)} figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
