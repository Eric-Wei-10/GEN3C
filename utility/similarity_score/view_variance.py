#!/usr/bin/env python3
# Visualize the saved per-frame variance vector.
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    # python view_variance.py --help.
    parser = argparse.ArgumentParser(
        description=(
            "Load the .npy file that contains the per-frame variance vector, "
            "print the variance list, and visualize it as a line plot.\n"
            "X-axis: frame index (0..T-1), Y-axis: variance."
        )
    )
    # python view_variance.py --input-npy path/to/frame_variance.npy
    parser.add_argument(
        "--input-npy",
        default="frame_variance.npy",
        help="Path to the per-frame variance .npy file (default: %(default)s).",
    )
    # python view_variance.py --output-img frame_variance.png
    parser.add_argument(
        "--output-img",
        default="frame_variance.png",
        help="Path to save the visualization image (default: %(default)s).",
    )
    # python view_variance.py --show
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving the image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Safety Check 1: check if input .npy file exists.
    if not os.path.isfile(args.input_npy):
        print(f"Error: input file '{args.input_npy}' not found.")
        return

    # Load the per-frame variance vector.
    var = np.load(args.input_npy)

    # Safety Check 2: ensure var is a 1D array.
    if var.ndim != 1:
        print(f"Warning: expected a 1D array, got shape {var.shape}. Flattening.")
        var = var.flatten()

    T = var.shape[0]
    frame_indices = np.arange(T)

    # Convert var to a Python list and print it.
    frame_var_list = var.tolist()
    print(f"Number of frames: {len(frame_var_list)}")
    print("frame_variances =", frame_var_list)

    # Plot variance vs. frame index.
    plt.figure()
    plt.plot(frame_indices, var, marker="o", linewidth=1)
    plt.xlabel("Frame index")
    plt.ylabel("Variance")
    plt.title("Per-frame variance across videos")
    plt.grid(True)

    # Save the figure.
    plt.tight_layout()
    plt.savefig(args.output_img, dpi=150)
    print(f"Saved visualization to {args.output_img}.")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
