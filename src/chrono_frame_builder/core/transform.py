import numpy as np

from chrono_frame_builder.core.frame import Frame

AXIS_SELECTORS = frozenset({"+X", "-X", "+Y", "-Y", "+Z", "-Z"})
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _normalize_vector(vector: np.ndarray, message: str, tolerance: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("Frame construction vectors must have shape (3,).")

    length = np.linalg.norm(vector)
    if length <= tolerance:
        raise ValueError(message)

    return vector / length


def _parse_axis_selector(axis_selector: str) -> tuple[int, int]:
    axis_selector = axis_selector.upper()
    if axis_selector not in AXIS_SELECTORS:
        valid = ", ".join(sorted(AXIS_SELECTORS))
        raise ValueError(f"Axis selector must be one of: {valid}.")

    sign = 1 if axis_selector[0] == "+" else -1
    axis_index = AXIS_INDEX[axis_selector[1]]
    return axis_index, sign


def rotation_matrix_from_origin_primary_secondary(
    primary_axis: str,
    primary_direction: np.ndarray,
    secondary_axis: str,
    secondary_direction: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Build a right-handed frame from primary and projected secondary directions."""
    primary_index, primary_sign = _parse_axis_selector(primary_axis)
    secondary_index, secondary_sign = _parse_axis_selector(secondary_axis)
    if primary_index == secondary_index:
        raise ValueError("Primary and secondary selectors must use different frame axes.")

    primary_unit = _normalize_vector(
        primary_direction,
        "Primary direction vector must be non-zero.",
        tolerance,
    )

    secondary_unit = _normalize_vector(
        secondary_direction,
        "Secondary direction vector must be non-zero.",
        tolerance,
    )
    projected_secondary = secondary_unit - np.dot(secondary_unit, primary_unit) * primary_unit
    projected_secondary = _normalize_vector(
        projected_secondary,
        "Secondary direction must not be parallel to the primary direction.",
        tolerance,
    )

    rotation_matrix = np.zeros((3, 3), dtype=float)
    rotation_matrix[:, primary_index] = primary_sign * primary_unit
    rotation_matrix[:, secondary_index] = secondary_sign * projected_secondary

    remaining_index = ({0, 1, 2} - {primary_index, secondary_index}).pop()
    rotation_matrix[:, remaining_index] = np.cross(
        rotation_matrix[:, (remaining_index + 1) % 3],
        rotation_matrix[:, (remaining_index + 2) % 3],
    )

    identity = np.eye(3)
    if not np.allclose(rotation_matrix.T @ rotation_matrix, identity, atol=tolerance):
        raise ValueError("Constructed rotation matrix is not orthonormal.")

    determinant = np.linalg.det(rotation_matrix)
    if not np.isclose(determinant, 1.0, atol=tolerance):
        raise ValueError("Constructed rotation matrix is not right-handed.")

    return rotation_matrix


def frame_from_origin_primary_secondary(
    name: str,
    origin: np.ndarray,
    primary_axis: str,
    primary_direction: np.ndarray,
    secondary_axis: str,
    secondary_direction: np.ndarray,
    parent_frame_name: str,
    *,
    frame_type: str | None = None,
    notes: str | None = None,
) -> Frame:
    """Create a Frame from origin plus primary and secondary feature directions."""
    del frame_type, notes

    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,):
        raise ValueError("Frame origin must have shape (3,).")

    rotation_matrix = rotation_matrix_from_origin_primary_secondary(
        primary_axis,
        primary_direction,
        secondary_axis,
        secondary_direction,
    )

    return Frame(
        body=parent_frame_name,
        name=name,
        origin=origin,
        rotation_matrix=rotation_matrix,
    )


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

    try:
        return rotation_matrix_from_origin_primary_secondary(
            "+X",
            x_direction_point - origin,
            "+Y",
            xy_plane_point - origin,
            tolerance=tolerance,
        )
    except ValueError as error:
        if "Primary direction" in str(error):
            raise ValueError(
                "P0 and P1 must be distinct to define the positive X direction."
            ) from error
        if "Secondary direction must not be parallel" in str(error):
            raise ValueError("P0, P1, and P2 must not be collinear.") from error
        if "Secondary direction vector" in str(error):
            raise ValueError("P0 and P2 must be distinct to define the XY plane.") from error
        raise ValueError("P0, P1, and P2 must not be collinear.") from error


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
