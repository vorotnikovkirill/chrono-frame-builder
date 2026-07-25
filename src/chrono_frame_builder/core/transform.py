import numpy as np

from chrono_frame_builder.core.frame import Frame


def rotation_matrix_from_three_points(
    origin: np.ndarray,
    x_direction_point: np.ndarray,
    xy_plane_point: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Build a right-handed rotation matrix from origin, +X point, and XY-plane point."""
    origin = np.asarray(origin, dtype=float)
    x_direction_point = np.asarray(x_direction_point, dtype=float)
    xy_plane_point = np.asarray(xy_plane_point, dtype=float)

    if origin.shape != (3,) or x_direction_point.shape != (3,) or xy_plane_point.shape != (3,):
        raise ValueError("Three-point frame construction requires three 3D points.")

    x_vector = x_direction_point - origin
    x_length = np.linalg.norm(x_vector)
    if x_length <= tolerance:
        raise ValueError("P0 and P1 must be distinct to define the positive X direction.")

    x_axis = x_vector / x_length
    plane_vector = xy_plane_point - origin
    z_vector = np.cross(x_axis, plane_vector)
    z_length = np.linalg.norm(z_vector)
    if z_length <= tolerance:
        raise ValueError("P0, P1, and P2 must not be collinear.")

    z_axis = z_vector / z_length
    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack((x_axis, y_axis, z_axis))


def frame_from_three_points(
    body: str,
    name: str,
    origin: np.ndarray,
    x_direction_point: np.ndarray,
    xy_plane_point: np.ndarray,
) -> Frame:
    """Create a Frame from origin, +X point, and XY-plane point."""
    origin = np.asarray(origin, dtype=float)
    rotation_matrix = rotation_matrix_from_three_points(
        origin,
        x_direction_point,
        xy_plane_point,
    )
    return Frame(body=body, name=name, origin=origin, rotation_matrix=rotation_matrix)
