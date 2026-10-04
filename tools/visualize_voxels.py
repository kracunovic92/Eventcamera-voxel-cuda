
#!/usr/bin/env python3
"""Visualize temporal slices of an exported event voxel grid.

Examples:
    python3 tools/visualize_voxels.py \
        results/voxels/sparklers.npy

    python3 tools/visualize_voxels.py \
        results/voxels/sparklers.npy \
        --show

    python3 tools/visualize_voxels.py \
        results/voxels/sparklers.npy \
        --clip-percentile 99
"""

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np


def load_voxels(path: Path) -> np.ndarray:
    """Load and validate a [bins, height, width] tensor."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Voxel file does not exist: {path}"
        )

    voxels = np.load(
        path,
        allow_pickle=False,
    )

    if voxels.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional voxel tensor "
            f"[bins, height, width], got {voxels.shape}"
        )

    if voxels.dtype != np.dtype("float32"):
        raise ValueError(
            "Expected float32 voxel values, "
            f"got {voxels.dtype}"
        )

    if any(dimension == 0 for dimension in voxels.shape):
        raise ValueError(
            "Voxel tensor dimensions must be positive"
        )

    if not np.isfinite(voxels).all():
        raise ValueError(
            "Voxel tensor contains NaN or infinity"
        )

    return voxels


def calculate_color_limit(
    voxels: np.ndarray,
    clip_percentile: float,
) -> float:
    """Calculate one symmetric color scale for all bins."""

    absolute_values = np.abs(voxels)

    if clip_percentile == 100.0:
        limit = float(absolute_values.max())
    else:
        # Exclude zero-valued pixels so sparse event data
        # doesn't force the percentile toward zero.
        active_values = absolute_values[
            absolute_values > 0
        ]

        if active_values.size == 0:
            return 1.0

        limit = float(
            np.percentile(
                active_values,
                clip_percentile,
            )
        )

    # Matplotlib requires a nonzero color interval.
    return limit if limit > 0 else 1.0


def print_statistics(voxels: np.ndarray) -> None:
    """Summarize the numerical content of each bin."""

    bins, height, width = voxels.shape

    print("Voxel grid:")
    print(f"  Shape: {bins} x {height} x {width}")
    print(f"  Data type: {voxels.dtype}")
    print(f"  Array bytes: {voxels.nbytes}")
    print()

    for index, frame in enumerate(voxels):
        print(
            f"  Bin {index}: "
            f"min={frame.min():.6g}, "
            f"max={frame.max():.6g}, "
            f"sum={frame.sum(dtype=np.float64):.6g}, "
            f"nonzero={np.count_nonzero(frame)}"
        )


def visualize(
    voxels: np.ndarray,
    output_path: Path,
    title: str,
    clip_percentile: float,
    show: bool,
    dpi: int,
) -> None:
    """Create a figure containing all temporal slices."""

    # Select a noninteractive backend when running on
    # servers or machines without a graphical desktop.
    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    bins, _, _ = voxels.shape

    columns = min(bins, 3)
    rows = math.ceil(bins / columns)

    color_limit = calculate_color_limit(
        voxels,
        clip_percentile,
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    axes_flat = axes.flat
    images = []
    active_axes = []

    for index, frame in enumerate(voxels):
        ax = next(axes_flat)

        image = ax.imshow(
            frame,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            origin="upper",
            interpolation="nearest",
        )

        ax.set_title(f"Temporal bin {index}")
        ax.set_xlabel("X coordinate (pixels)")
        ax.set_ylabel("Y coordinate (pixels)")

        images.append(image)
        active_axes.append(ax)

    # Hide unused positions in the subplot grid.
    for ax in axes_flat:
        ax.set_visible(False)

    # All panels use identical limits, so they can
    # share one color bar.
    fig.colorbar(
        images[0],
        ax=active_axes,
        orientation="vertical",
        shrink=0.85,
        label="Signed event contribution",
    )

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print()
    print(f"Visualization saved: {output_path}")
    print(f"Shared color scale: ±{color_limit:.6g}")

    if clip_percentile < 100.0:
        print(
            "Note: Color values exceeding this limit "
            "are visually clipped. The underlying "
            "voxel data remains unchanged."
        )

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a [bins, height, width] "
            "event voxel grid stored as NumPy .npy"
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="NumPy voxel grid file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path",
    )

    parser.add_argument(
        "--title",
        type=str,
        help="Optional figure title",
    )

    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=100.0,
        help=(
            "Percentile used for symmetric color limits "
            "(default: 100, no clipping)"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Saved figure resolution",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )

    args = parser.parse_args()

    if not 0 < args.clip_percentile <= 100:
        parser.error(
            "--clip-percentile must be in (0, 100]"
        )

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    output_path = args.output

    if output_path is None:
        output_path = (
            Path("results/visualizations")
            / f"{args.input.stem}.png"
        )

    title = args.title or (
        f"Event voxel grid: {args.input.stem}"
    )

    voxels = load_voxels(args.input)

    print_statistics(voxels)

    visualize(
        voxels=voxels,
        output_path=output_path,
        title=title,
        clip_percentile=args.clip_percentile,
        show=args.show,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
