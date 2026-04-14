import argparse
import numpy as np
import matplotlib.pyplot as plt


def _load_npz(path):
    """
    Load a .npz file.
    
    :param path: Path to the .npz file.
    :return: Loaded numpy data.
    """
    return np.load(path, allow_pickle=True)


def _is_combined(data):
    """
    Check if the loaded data corresponds to combined (multi-trajectory) results.
    
    :param data: Loaded numpy data.
    :return: True if combined results, False otherwise.
    """
    return ("rgb_variance_map" in data) or ("variance_map" in data)


def _convert_to_HW(arr):
    """
    Convert array to 2D (H, W) by squeezing singleton dimensions.
    
    :param arr: Input array.
    :return: 2D array with shape (H, W).
    """
    a = np.asarray(arr)
    while a.ndim >= 3 and 1 in a.shape:
        a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D (H,W), got {a.shape}")
    return a


def _save_im(arr, path, title, cbar_label, boundary_mask=None, cmap_name=None, vmin=None, vmax=None, overlay_text=None):
    """
    Save a 2D array as an image with colorbar and optional boundary overlay and text overlay.
    
    :param arr: 2D array to visualize.
    :param path: Path to save the image.
    :param title: Title of the image.
    :param cbar_label: Label for the colorbar.
    :param boundary_mask: Optional mask to overlay boundaries.
    :param cmap_name: Colormap name.
    :param vmin: Minimum value for colormap scaling.
    :param vmax: Maximum value for colormap scaling.
    :param overlay_text: Optional text to overlay on the image.
    """
    arr = np.asarray(arr)

    cmap = plt.get_cmap(cmap_name).copy() if cmap_name else plt.get_cmap().copy()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, origin="upper", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("u (pixel)")
    plt.ylabel("v (pixel)")
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    if boundary_mask is not None:
        try:
            plt.contour(boundary_mask.astype(float), levels=[0.5], colors="red", linewidths=1.0)
        except Exception:
            pass
    
    if overlay_text is not None and len(str(overlay_text).strip()) > 0:
        plt.gcf().text(
            0.01, 0.01, overlay_text,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description="Save variance + coverage + (combined) source in one pass.")
    p.add_argument("--result_npz", type=str, required=True)
    p.add_argument("--out_prefix", type=str, required=True,
                   help="Prefix for outputs. Writes *_variance.png, *_coverage.png, *_source.png")
    p.add_argument("--traj_mask", type=str, required=True, choices=["fill", "intersection"],
                   help="Single: chooses fill_mask vs intersection_mask. Combined: used for labeling/sanity-check.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    data = _load_npz(args.result_npz)

    mode = str(data["mode"].item()) if ("mode" in data and np.asarray(data["mode"]).shape == ()) else str(data.get("mode", "unknown"))
    channel_type = str(data["channel_type"].item()) if ("channel_type" in data and np.asarray(data["channel_type"]).shape == ()) else str(data.get("channel_type", "unknown"))
    occlude = bool(data["occlude"]) if "occlude" in data else False

    t = int(data["frame_index"]) if "frame_index" in data else -1

    # Combined case.
    if _is_combined(data):
        var = data["rgb_variance_map"].astype(np.float32) if "rgb_variance_map" in data else data["variance_map"].mean(axis=0).astype(np.float32)
        cm = _convert_to_HW(data["combined_mask"]).astype(np.float32)
        cm_bool = cm > 0.5

        # variance: mask outside combined coverage -> NaN.
        var_disp = np.full_like(var, np.nan, dtype=np.float32)
        var_disp[cm_bool] = var[cm_bool]

        _save_im(
            var_disp,
            f"{args.out_prefix}_variance.png",
            title=f"Combined variance",
            cbar_label="Variance",
            boundary_mask=cm_bool,
            cmap_name="viridis",
            vmin=args.vmin,
            vmax=args.vmax,
        )

        _save_im(
            cm.astype(np.float32),
            f"{args.out_prefix}_coverage.png",
            title=f"Combined coverage",
            cbar_label="mask",
            boundary_mask=cm_bool,
            cmap_name=None,
        )

        if "source_trajectory_index" in data:
            src = _convert_to_HW(data["source_trajectory_index"]).astype(np.int32)

            # Build "index -> result_{theta}_{phi}" text overlay.
            names = None
            if "trajectory_names" in data:
                names = [str(x) for x in list(data["trajectory_names"])]

            theta_phi = None
            if "trajectory_theta_phi" in data:
                theta_phi = np.asarray(data["trajectory_theta_phi"]).astype(np.float32)  # [M, 2]

            uniq = np.unique(src[src >= 0])
            uniq = np.sort(uniq)

            max_labels = 25  # keep image readable
            shown = uniq[:max_labels]

            lines = ["traj index → folder (theta, phi)"]
            if names is None:
                lines.append("(trajectory_names missing in npz)")
            else:
                for i in shown:
                    i = int(i)
                    name = names[i] if 0 <= i < len(names) else f"<idx {i} out of range>"
                    if theta_phi is not None and 0 <= i < theta_phi.shape[0]:
                        th, ph = float(theta_phi[i, 0]), float(theta_phi[i, 1])
                        lines.append(f"{i}: {name} (theta={th}, phi={ph})")
                    else:
                        lines.append(f"{i}: {name}")

                if uniq.size > max_labels:
                    lines.append(f"... ({uniq.size - max_labels} more indices not shown)")

            overlay = "\n".join(lines)

            # Visualize source map (invalid as NaN).
            src_disp = src.astype(np.float32)
            src_disp[src < 0] = np.nan

            _save_im(
                src_disp,
                f"{args.out_prefix}_source.png",
                title="Source trajectory index map",
                cbar_label="traj index",
                boundary_mask=cm_bool,
                cmap_name=None,
                overlay_text=overlay,
            )
        else:
            print("[INFO] No source_trajectory_index in NPZ; skipping source output.")

    else:
        var = data["var_scalar"].astype(np.float32) if "var_scalar" in data else data["var_map"].mean(axis=0).astype(np.float32)

        if args.traj_mask == "fill":
            if "fill_mask" not in data:
                raise KeyError("Single NPZ missing fill_mask.")
            cov = _convert_to_HW(data["fill_mask"]).astype(np.float32)
        else:
            if "intersection_mask" not in data:
                raise KeyError("Single NPZ missing intersection_mask.")
            cov = _convert_to_HW(data["intersection_mask"]).astype(np.float32)

        cov_bool = cov > 0.5
        var_disp = np.full_like(var, np.nan, dtype=np.float32)
        var_disp[cov_bool] = var[cov_bool]

        _save_im(
            var_disp,
            f"{args.out_prefix}_variance.png",
            title=f"Single variance (mode={mode}, t={t}, traj_mask={args.traj_mask}, channel={channel_type}, occlude={occlude})",
            cbar_label="Variance",
            boundary_mask=cov_bool,
            cmap_name="viridis",
            vmin=args.vmin,
            vmax=args.vmax,
        )

        _save_im(
            cov.astype(np.float32),
            f"{args.out_prefix}_coverage.png",
            title=f"Single coverage (traj_mask={args.traj_mask}, occlude={occlude})",
            cbar_label="mask",
            boundary_mask=cov_bool,
            cmap_name=None,
        )

        # no source for single.
        print("[INFO] Single NPZ: source output not applicable; skipping.")


if __name__ == "__main__":
    main()
