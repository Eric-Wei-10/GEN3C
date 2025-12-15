import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _make_mask_2d(mask, target_shape):
    """Ensure mask is 2D and matches (H,W). Accepts [H,W], [W,H], or 1D [H]/[W]."""
    if mask.ndim == 2:
        if mask.shape == target_shape:
            return mask
        if mask.T.shape == target_shape:
            return mask.T
        raise ValueError(
            f"mask shape {mask.shape} does not match target {target_shape}, "
            f"and transposing does not fix it."
        )

    if mask.ndim == 1:
        H, W = target_shape
        if mask.shape[0] == W:
            return np.tile(mask[None, :], (H, 1))
        if mask.shape[0] == H:
            return np.tile(mask[:, None], (1, W))
        raise ValueError(
            f"1D mask length {mask.shape[0]} incompatible with target {target_shape}."
        )

    raise ValueError(f"mask has unsupported ndim={mask.ndim}, shape={mask.shape}.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-pixel variance saved by compute_per_pixel_variance.py and debug coverage (valid_counts).\n"
            "Shows a color grid where different colors indicate different variance ranges. Works for all modes."
        )
    )
    parser.add_argument("--result_npz", type=str, required=True,
                        help="Path to .npz file produced by compute_per_pixel_variance.py")
    parser.add_argument("--save_path", type=str, default=None,
                        help="If provided, save visualization to this path (e.g. 'variance.png')")

    parser.add_argument("--mask_source", type=str, default="intersection",
                        choices=["intersection", "counts", "none"],
                        help="Which mask to use for blanking + boundary contour.")
    parser.add_argument("--min_count", type=int, default=2,
                        help="Used when mask_source=counts: show only where valid_counts>=min_count.")

    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)

    parser.add_argument("--save_coverage", action="store_true",
                        help="If set and valid_counts exists, save *_coverage.png (debug).")
    parser.add_argument("--no_boundary", action="store_true",
                        help="Disable drawing the red boundary contour.")

    args = parser.parse_args()

    # Load results.
    data = np.load(args.result_npz)
    var_map = data["var_map"].astype(np.float32)            # [C, H, W]
    frame_index = int(data["frame_index"]) if "frame_index" in data else -1
    mode = str(data["mode"]) if "mode" in data else "unknown"
    var_scalar = var_map.mean(axis=0).astype(np.float32)    # [H, W]
    H, W = var_scalar.shape

    # Choose mask.
    if args.mask_source == "none":
        mask_bool = np.ones((H, W), dtype=bool)
        mask_label = "Full image"
    elif args.mask_source == "counts":
        if "valid_counts" not in data:
            raise KeyError("mask_source=counts but valid_counts not found in npz.")
        vc = _make_mask_2d(data["valid_counts"].astype(np.float32), (H, W))
        mask_bool = vc >= float(args.min_count)
        mask_label = f"valid_counts >= {args.min_count}"
    else:  # intersection
        if "intersection_mask" not in data:
            raise KeyError("mask_source=intersection but intersection_mask not found in npz.")
        inter = _make_mask_2d(data["intersection_mask"], (H, W))
        mask_bool = inter > 0.5
        mask_label = "Intersection boundary"

    # NaN outside mask.
    var_display = np.full((H, W), np.nan, dtype=np.float32)
    var_display[mask_bool] = var_scalar[mask_bool]

    # Plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        var_display,
        cmap="viridis",
        interpolation="nearest",
        origin="upper",
        vmin=args.vmin,
        vmax=args.vmax,
    )

    if mode == "frame_t":
        plt.title(f"Per-pixel variance on frame t (frame_index = {frame_index})")
    else:
        plt.title(f"Per-pixel variance on frame 0 (mode={mode}, frame_index = {frame_index})")

    plt.xlabel("u (pixel)")
    plt.ylabel("v (pixel)")
    cbar = plt.colorbar(im)
    cbar.set_label("Variance")

    # Boundary contour + legend.
    if (not args.no_boundary) and (mask_label is not None):
        try:
            plt.contour(
                mask_bool.astype(float),
                levels=[0.5],
                colors="red",
                linewidths=1.0,
                linestyles="-",
            )
            proxy = Line2D([0], [0], color="red", lw=1.0, linestyle="-", label=mask_label)
            plt.legend(handles=[proxy], loc="upper right")
        except Exception as e:
            print(f"[WARN] Failed to draw contour: {e}")

    plt.tight_layout()

    if args.save_path is not None:
        plt.savefig(args.save_path, dpi=200)
        print(f"Saved variance visualization to {args.save_path}")
    else:
        plt.show()

    # Optional: save coverage debug.
    if args.save_coverage and ("valid_counts" in data) and (args.save_path is not None):
        vc = _make_mask_2d(data["valid_counts"].astype(np.float32), (H, W))

        root, ext = os.path.splitext(args.save_path)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        coverage_path = f"{root}_coverage{ext}"

        plt.figure(figsize=(8, 6))
        im2 = plt.imshow(vc, interpolation="nearest", origin="upper")
        plt.title(f"Coverage valid_counts (mode={mode}, frame_index = {frame_index})")
        plt.xlabel("u (pixel)")
        plt.ylabel("v (pixel)")
        cbar2 = plt.colorbar(im2)
        cbar2.set_label("valid_counts")

        if (not args.no_boundary) and (mask_label is not None):
            try:
                plt.contour(
                    mask_bool.astype(float),
                    levels=[0.5],
                    colors="red",
                    linewidths=1.0,
                    linestyles="-",
                )
                proxy = Line2D([0], [0], color="red", lw=1.0, linestyle="-", label=mask_label)
                plt.legend(handles=[proxy], loc="upper right")
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(coverage_path, dpi=200)
        print(f"Saved coverage visualization to {coverage_path}")


if __name__ == "__main__":
    main()
