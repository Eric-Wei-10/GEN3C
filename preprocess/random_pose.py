"""
random_pose.py

Randomly sample N camera poses, render each from a .glb file, and optionally
filter out invalid views (collision / not explorable) using data_filtering.py.

Sampling distribution:
    x      ~ Uniform[-0.3, 7.5]
    y      ~ Uniform[-3.6, 4.4]
    z      ~ choice{1.0, -2.0}
    angle  ~ choice{0, 90, 180, 270}

Usage:
    # Render 10 poses
    python random_pose.py --n 10 --glb eval_data/mesh/000000-kfPV7w3FaU5.glb

    # Render + filter invalid views (deletes bad images automatically)
    python random_pose.py --n 10 --glb eval_data/mesh/000000-kfPV7w3FaU5.glb --filter
"""

import argparse
import os
import numpy as np
from pathlib import Path

# Required for Open3D's OffscreenRenderer on headless cluster nodes
os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/runtime-{os.getuid()}")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], mode=0o700, exist_ok=True)

from get_w2c import generate_matrix, format_matrix_lines
from data_filtering import process_one


def render_glb(glb_path, w2c_matrix, width, height, fx, fy, cx, cy, output_path):
    """
    Render an RGB image of a .glb file from the given world-to-camera pose
    using Open3D's OffscreenRenderer.
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    import cv2

    renderer = rendering.OffscreenRenderer(width, height)

    model = o3d.io.read_triangle_model(glb_path)
    if model is None or len(model.meshes) == 0:
        print(f"错误: 无法加载模型 {glb_path}")
        return

    for i, mesh_info in enumerate(model.meshes):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        if mesh_info.material_idx >= 0 and mesh_info.material_idx < len(model.materials):
            src = model.materials[mesh_info.material_idx]
            if src.albedo_img is not None:
                mat.albedo_img = src.albedo_img
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        renderer.scene.add_geometry(f"mesh_{i}", mesh_info.mesh, mat)

    renderer.scene.scene.enable_sun_light(False)
    renderer.scene.scene.enable_indirect_light(False)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ])
    renderer.setup_camera(K, w2c_matrix, width, height)

    img = np.asarray(renderer.render_to_image())
    renderer.scene.clear_geometry()

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"渲染图像已保存: {output_path}")


def sample_pose(rng: np.random.Generator):
    x     = rng.uniform(-0.3, 7.5)
    y     = rng.uniform(-3.6, 4.4)
    z     = rng.choice([1.0, -2.0])
    angle = rng.choice([0.0, 90.0, 180.0, 270.0])
    return x, y, z, angle


def main():
    parser = argparse.ArgumentParser(description="Random camera pose → render + optional filter")
    parser.add_argument("--n",               type=int,   default=1,              help="Number of poses to sample (default: 1)")
    parser.add_argument("--seed",            type=int,   default=42,             help="Random seed (default: 42)")
    parser.add_argument("--glb",             type=str,   default=None,           help="Path to .glb file to render")
    parser.add_argument("--output",          type=str,   default="dummy_render", help="Output directory for rendered images (default: dummy_render)")
    parser.add_argument("--width",           type=int,   default=720,            help="Render width (default: 720)")
    parser.add_argument("--height",          type=int,   default=544,            help="Render height (default: 544)")
    parser.add_argument("--fx",              type=float, default=300.0,          help="Focal length fx (default: 300)")
    parser.add_argument("--fy",              type=float, default=300.0,          help="Focal length fy (default: 300)")
    parser.add_argument("--cx",              type=float, default=None,           help="Principal point cx (default: width/2)")
    parser.add_argument("--cy",              type=float, default=None,           help="Principal point cy (default: height/2)")
    # Filtering options
    parser.add_argument("--filter",          action="store_true",                help="Run data_filtering on each render and delete invalid images")
    parser.add_argument("--forward",         type=float, default=1.0,            help="Forward-move distance for explorability check (default: 1.0 m)")
    parser.add_argument("--depth_threshold", type=float, default=0.5,            help="Depth threshold for explorability check (default: 0.5 m)")
    parser.add_argument("--device",          type=str,   default="cuda",         help="Device for MoGE model (default: cuda)")
    parser.add_argument("--filter_out_dir",  type=str,   default=None,           help="Directory for filter debug outputs (default: <output>/filter_debug)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load MoGE once if filtering is enabled
    moge_model = None
    if args.filter:
        from moge.model.v1 import MoGeModel
        import torch
        device = args.device if torch.cuda.is_available() else "cpu"
        print("Loading MoGE model for filtering …")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        moge_model.eval()
        filter_out_dir = Path(args.filter_out_dir or os.path.join(args.output, "filter_debug"))
        filter_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        device = args.device

    rng = np.random.default_rng(args.seed)
    cx  = args.cx if args.cx is not None else args.width  / 2.0
    cy  = args.cy if args.cy is not None else args.height / 2.0

    n_kept = n_collision = n_not_explorable = 0

    for i in range(args.n):
        x, y, z, angle = sample_pose(rng)
        matrix    = generate_matrix(x, y, z, angle)
        new_lines = format_matrix_lines(matrix)

        print("-" * 40)
        print(f"[{i+1}/{args.n}]  x={x:.4f}  y={y:.4f}  z={z:.1f}  angle={angle:.1f}°")
        print("W2C matrix:")
        for line in new_lines:
            print(" ", line.strip())

        if args.glb is None:
            continue

        out_path = Path(args.output) / f"{i:03d}.png"
        render_glb(
            glb_path=args.glb,
            w2c_matrix=matrix,
            width=args.width,
            height=args.height,
            fx=args.fx,
            fy=args.fy,
            cx=cx,
            cy=cy,
            output_path=str(out_path),
        )

        if args.filter:
            status = process_one(
                image_path=out_path,
                out_dir=filter_out_dir,
                forward=args.forward,
                depth_threshold=args.depth_threshold,
                device=device,
                moge_model=moge_model,
                delete_invalid=True,
            )
            print(f"  → filter status: {status}")
            if status == "collision":
                n_collision += 1
            elif status == "not_explorable":
                n_not_explorable += 1
            else:
                n_kept += 1
        else:
            n_kept += 1

    print("-" * 40)
    print(f"Done.  kept={n_kept}  collision={n_collision}  not_explorable={n_not_explorable}")


if __name__ == "__main__":
    main()
