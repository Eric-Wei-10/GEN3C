import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-pixel variance saved by compute_per_pixel_variance.py.\n"
            "Shows a color grid where different colors indicate different variance ranges."
        )
    )
    parser.add_argument(
        "--result_npz", type=str, required=True,
        help="Path to .npz file produced by compute_per_pixel_variance.py"
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="If provided, save visualization to this path (e.g. 'variance.png')"
    )

    args = parser.parse_args()

    # Load results.
    data = np.load(args.result_npz)
    var_map = data["var_map"]                   # [C, H, W]
    inter_mask = data["intersection_mask"]      # Could be [H, W], [W, H] or 1D
    frame_index = int(data["frame_index"])

    # Reduce per-pixel per-channel 3D variance map to per-pixel 2D variance map.
    # Average over channels -> scalar variance per pixel.
    var_scalar = var_map.mean(axis=0)  # [H, W]

    mask = inter_mask
    # Ensure mask shape match (H, W).
    if mask.ndim == 2:
        # Case 1: [H, W] or [W, H].
        if mask.shape == var_scalar.shape:          # [H, W]
            pass 
        elif mask.T.shape == var_scalar.shape:      # [W, H]
            mask = mask.T
        else:
            raise ValueError(
                f"intersection_mask shape {mask.shape} does not match var_map spatial shape "
                f"{var_scalar.shape}, and transposing does not fix it."
            )
    elif mask.ndim == 1:
        # Case 2: [H] or [W] – broadcast to 2D.
        if mask.shape[0] == var_scalar.shape[1]:
            # length == W -> broadcast along H
            mask = np.tile(mask[None, :], (var_scalar.shape[0], 1))  # [H, W]
        elif mask.shape[0] == var_scalar.shape[0]:
            # length == H -> broadcast along W
            mask = np.tile(mask[:, None], (1, var_scalar.shape[1]))  # [H, W]
        else:
            raise ValueError(
                f"1D intersection_mask length {mask.shape[0]} is incompatible with "
                f"var_map spatial shape {var_scalar.shape}."
            )
    else:
        raise ValueError(
            f"intersection_mask has unsupported ndim={mask.ndim}, shape={mask.shape}."
        )

    # Convert the mask to a boolean mask where True indicates valid pixels.
    mask_bool = mask > 0.5

    # Only show variance within the intersection mask.
    # Outside the mask, set to NaN so that it appears blank in the visualization.
    var_display = np.full(var_scalar.shape, np.nan, dtype=np.float32)
    var_display[mask_bool] = var_scalar[mask_bool]

    # Plot.
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        var_display,
        cmap="viridis",
        interpolation="nearest",
        origin="upper",
    )
    plt.title(f"Per-pixel variance on frame 0 (frame_index = {frame_index})")
    plt.xlabel("u (pixel)")
    plt.ylabel("v (pixel)")
    cbar = plt.colorbar(im)
    cbar.set_label("Variance")

    # Draw intersection boundary
    try:
        cs = plt.contour(
            mask_bool.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=1.0,
            linestyles="-",
        )
        cs.collections[0].set_label("Intersection boundary")
        plt.legend(loc="upper right")
    except Exception:
        pass

    plt.tight_layout()

    # Save/show.
    if args.save_path is not None:
        plt.savefig(args.save_path, dpi=200)
        print(f"Saved variance visualization to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
