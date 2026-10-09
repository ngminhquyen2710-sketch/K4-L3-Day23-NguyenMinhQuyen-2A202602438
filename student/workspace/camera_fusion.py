"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    p_v = np.asarray(x, dtype=float).ravel()[:3]
    transform = np.asarray(sensor.veh_to_sens)
    p_s = transform[:3, :3] @ p_v + transform[:3, 3]
    if not np.isfinite(p_s).all():
        return False
    x_s, y_s, _ = p_s
    is_camera = (
        getattr(sensor, "name", None) == "camera"
        or hasattr(sensor, "f_i")
        or hasattr(sensor, "depth_epsilon")
    )
    if is_camera and x_s <= 1e-6:
        return False
    angle = np.arctan2(y_s, x_s)
    min_fov = min(sensor.fov)
    max_fov = max(sensor.fov)
    return bool(min_fov <= angle <= max_fov)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    p_v = np.asarray(x, dtype=float).ravel()[:3]
    transform = np.asarray(sensor.veh_to_sens)
    p_s = transform[:3, :3] @ p_v + transform[:3, 3]
    if not np.isfinite(p_s).all() or p_s[0] <= 1e-6:
        raise ValueError(
            f"Camera projection requires finite coordinates and positive depth > 1e-6; got p_s={p_s.tolist()}"
        )
    x_s, y_s, z_s = p_s
    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)
    return np.asmatrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.asmatrix([[float(z[0])], [float(z[1])]])
    R = np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {
        "z": z_mat,
        "R": R,
        "sensor": sensor,
    }
