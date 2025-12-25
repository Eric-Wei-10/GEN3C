# variance/orientation.py
# Parse theta, phi from trajectory folder name and compute orientation distance in radians.
import os
import re
import math

# Match folder names like "result_<theta>_<phi>".
_dir_re = re.compile(r"^result_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")


def _wrap_deg(a: float) -> float:
    """
    Wrap angle in degrees to [-180, 180).
    
    :param a: Angle in degrees.
    :return: Wrapped angle in degrees.
    """
    return (a + 180.0) % 360.0 - 180.0


def parse_theta_phi_from_dir(traj_dir: str):
    """
    Parse theta and phi from trajectory directory name.
    
    :param traj_dir: The trajectory directory path.
    :return: A tuple (theta, phi) in degrees.
    """
    base = os.path.basename(os.path.normpath(traj_dir))
    m = _dir_re.match(base)
    if not m:
        raise ValueError(f"Trajectory folder name must be 'result_{{theta}}_{{phi}}', got: {base}")
    return float(m.group(1)), float(m.group(2))


def orientation_distance_rad(theta_deg: float, phi_deg: float) -> float:
    """
    Compute orientation distance in radians between the given (theta, phi) and the reference (0, 0).
    
    :param theta_deg: Angle between projected motion direction onto x-z plane and +z axis (yaw-like) in degrees.
    :param phi_deg: Angle between motion direction and x-z plane (elevation-like) in degrees.
    :return: Orientation distance from (0,0) in radians.
    """
    theta_deg = _wrap_deg(theta_deg)
    phi_deg = _wrap_deg(phi_deg)
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)

    # radian distance = arccos(cos(phi) * cos(theta)).
    c = math.cos(ph) * math.cos(th)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)
