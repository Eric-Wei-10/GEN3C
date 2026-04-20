#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a 3D camera trajectory (SE3) with forward motion and left/right yaw exploration.

Pose representation: 4x4 homogeneous transform matrix T in SE(3)
    T = [ R  t ]
        [ 0  1 ]
where
    R: 3x3 rotation matrix
    t: 3x1 translation (x, y, z)

Convention:
    - Robot / camera local frame:
        * x-axis: forward
        * y-axis: left
        * z-axis: up
    - We mainly perturb yaw (rotation around z-axis) to create a fan-shaped exploration.

CLI input:
    - Initial pose via (x0, y0, z0, yaw0_deg, pitch0_deg, roll0_deg)
    - Forward speed v_forward
    - Time step dt
    - Number of steps num_steps
"""

import argparse
import numpy as np


def euler_zyx_to_rot(yaw, pitch, roll):
    """
    Convert ZYX euler angles to rotation matrix.
    yaw   (psi): rotation around z-axis
    pitch (theta): rotation around y-axis
    roll  (phi): rotation around x-axis

    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Rotation around Z
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])

    # Rotation around Y
    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])

    # Rotation around X
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    R = Rz @ Ry @ Rx
    return R


def make_SE3_from_xyz_euler(x, y, z, yaw, pitch, roll):
    """Create 4x4 SE3 from translation and ZYX euler angles (in radians)."""
    R = euler_zyx_to_rot(yaw, pitch, roll)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def generate_trajectory_3d(start_pose_SE3,
                           dt,
                           num_steps,
                           v_forward,
                           yaw_noise_std_deg=5.0,
                           max_fan_deg=60.0,
                           seed=None):
    """
    Generate a 3D trajectory as a sequence of SE3 matrices.

    Args:
        start_pose_SE3: (4,4) homogeneous transform, initial camera pose.
        dt: time step size.
        num_steps: how many steps to simulate.
        v_forward: forward speed along local x-axis (units per second).
        yaw_noise_std_deg: std of yaw noise per step, in degrees.
        max_fan_deg: max yaw deviation (around global z) w.r.t initial yaw, in degrees.
        seed: random seed for reproducibility.

    Returns:
        traj: (num_steps + 1, 4, 4) array of SE3 matrices.
    """
    traj = np.zeros((num_steps + 1, 4, 4), dtype=float)
    traj[0] = start_pose_SE3.copy()

    rng = np.random.default_rng(seed)
    yaw_noise_std_rad = np.deg2rad(yaw_noise_std_deg)
    max_fan_rad = np.deg2rad(max_fan_deg)

    # Extract initial yaw from start_pose_SE3 by looking at R
    R0 = start_pose_SE3[:3, :3]
    # yaw0 from R0 (assuming ZYX convention); robust way from atan2
    yaw0 = np.arctan2(R0[1, 0], R0[0, 0])

    # We'll keep pitch and roll fixed equal to their initial values.
    # Recover pitch and roll from R0.
    # For ZYX:
    #   yaw   = atan2(R10, R00)
    #   pitch = asin(-R20)
    #   roll  = atan2(R21, R22)
    pitch0 = np.arcsin(-R0[2, 0])
    roll0 = np.arctan2(R0[2, 1], R0[2, 2])

    # Current state
    T = start_pose_SE3.copy()
    yaw = yaw0

    for t in range(1, num_steps + 1):
        # 1) Add yaw noise (around global z or local z, here we approximate via global z)
        dyaw = rng.normal(0.0, yaw_noise_std_rad)
        yaw = yaw + dyaw

        # 2) Clamp yaw into fan [yaw0 - max_fan, yaw0 + max_fan]
        yaw = np.clip(yaw, yaw0 - max_fan_rad, yaw0 + max_fan_rad)

        # 3) Build rotation with updated yaw but fixed pitch and roll
        R = euler_zyx_to_rot(yaw, pitch0, roll0)

        # 4) Move forward in local frame x-axis
        # Local forward direction in world = R * [1, 0, 0]^T
        forward_world = R @ np.array([1.0, 0.0, 0.0])
        delta_pos = v_forward * dt * forward_world

        # 5) Update translation
        pos = T[:3, 3] + delta_pos

        # 6) Compose new SE3
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = pos

        traj[t] = T

    return traj


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D camera trajectory (SE3) with forward motion and left/right yaw exploration."
    )

    # initial pose parameters
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y")
    parser.add_argument("--z0", type=float, default=0.0, help="Initial z")

    parser.add_argument(
        "--yaw0_deg",
        type=float,
        default=0.0,
        help="Initial yaw in degrees (rotation around z-axis)"
    )
    parser.add_argument(
        "--pitch0_deg",
        type=float,
        default=0.0,
        help="Initial pitch in degrees (rotation around y-axis)"
    )
    parser.add_argument(
        "--roll0_deg",
        type=float,
        default=0.0,
        help="Initial roll in degrees (rotation around x-axis)"
    )

    # motion parameters
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="How many time steps to simulate"
    )
    parser.add_argument(
        "--v_forward",
        type=float,
        default=1.0,
        help="Forward speed along local x-axis (units/second)"
    )

    # exploration parameters
    parser.add_argument(
        "--yaw_noise_std_deg",
        type=float,
        default=5.0,
        help="Std of yaw noise per step (degrees)"
    )
    parser.add_argument(
        "--max_fan_deg",
        type=float,
        default=60.0,
        help="Max yaw deviation from initial heading (degrees)"
    )

    # misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducibility)"
    )

    args = parser.parse_args()

    # Convert angles to radians
    yaw0 = np.deg2rad(args.yaw0_deg)
    pitch0 = np.deg2rad(args.pitch0_deg)
    roll0 = np.deg2rad(args.roll0_deg)

    # build initial SE3
    T0 = make_SE3_from_xyz_euler(
        x=args.x0,
        y=args.y0,
        z=args.z0,
        yaw=yaw0,
        pitch=pitch0,
        roll=roll0
    )

    traj = generate_trajectory_3d(
        start_pose_SE3=T0,
        dt=args.dt,
        num_steps=args.num_steps,
        v_forward=args.v_forward,
        yaw_noise_std_deg=args.yaw_noise_std_deg,
        max_fan_deg=args.max_fan_deg,
        seed=args.seed,
    )

    # Output: one 4x4 matrix per step, separated by blank lines
    # You can redirect to a file and parse it yourself in Python / C++ / Unity, etc.
    for i, T in enumerate(traj):
        print(f"# step {i}")
        for r in range(4):
            print(" ".join(f"{v:.8f}" for v in T[r]))
        print()  # blank line between steps


if __name__ == "__main__":
    main()
