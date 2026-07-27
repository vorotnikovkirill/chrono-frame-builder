import numpy as np
import pytest

from chrono_frame_builder.core.transform import (
    frame_from_origin_primary_secondary,
    rotation_matrix_from_origin_primary_secondary,
)


def assert_orthonormal_right_handed(rotation_matrix):
    np.testing.assert_allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation_matrix) == pytest.approx(1.0)


def test_primary_positive_z_from_face_normal_secondary_positive_x_from_edge_direction():
    rotation_matrix = rotation_matrix_from_origin_primary_secondary(
        "+Z",
        np.array([0.0, 0.0, 10.0]),
        "+X",
        np.array([4.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(rotation_matrix, np.eye(3))
    assert_orthonormal_right_handed(rotation_matrix)


def test_secondary_vector_is_projected_onto_plane_normal_to_primary():
    rotation_matrix = rotation_matrix_from_origin_primary_secondary(
        "+Z",
        np.array([0.0, 0.0, 1.0]),
        "+X",
        np.array([2.0, 2.0, 5.0]),
    )

    expected_x = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    expected_y = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(rotation_matrix[:, 0], expected_x)
    np.testing.assert_allclose(rotation_matrix[:, 1], expected_y)
    np.testing.assert_allclose(rotation_matrix[:, 2], [0.0, 0.0, 1.0])
    assert_orthonormal_right_handed(rotation_matrix)


def test_signed_axis_selectors_are_respected():
    rotation_matrix = rotation_matrix_from_origin_primary_secondary(
        "-Z",
        np.array([0.0, 0.0, 1.0]),
        "+X",
        np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(rotation_matrix[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation_matrix[:, 1], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(rotation_matrix[:, 2], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(-rotation_matrix[:, 2], [0.0, 0.0, 1.0])
    assert_orthonormal_right_handed(rotation_matrix)


def test_same_underlying_axis_selectors_are_rejected():
    with pytest.raises(ValueError, match="different frame axes"):
        rotation_matrix_from_origin_primary_secondary(
            "+X",
            np.array([1.0, 0.0, 0.0]),
            "-X",
            np.array([0.0, 1.0, 0.0]),
        )


def test_collinear_primary_and_secondary_vectors_are_rejected():
    with pytest.raises(ValueError, match="parallel"):
        rotation_matrix_from_origin_primary_secondary(
            "+X",
            np.array([1.0, 0.0, 0.0]),
            "+Y",
            np.array([5.0, 0.0, 0.0]),
        )


def test_zero_vectors_are_rejected():
    with pytest.raises(ValueError, match="Primary direction vector"):
        rotation_matrix_from_origin_primary_secondary(
            "+X",
            np.array([0.0, 0.0, 0.0]),
            "+Y",
            np.array([0.0, 1.0, 0.0]),
        )

    with pytest.raises(ValueError, match="Secondary direction vector"):
        rotation_matrix_from_origin_primary_secondary(
            "+X",
            np.array([1.0, 0.0, 0.0]),
            "+Y",
            np.array([0.0, 0.0, 0.0]),
        )


def test_frame_from_origin_primary_secondary_uses_existing_frame_model():
    frame = frame_from_origin_primary_secondary(
        name="mount_face",
        origin=np.array([10.0, 20.0, 30.0]),
        primary_axis="+Z",
        primary_direction=np.array([0.0, 0.0, 1.0]),
        secondary_axis="+X",
        secondary_direction=np.array([1.0, 0.0, 0.0]),
        parent_frame_name="demo_bracket",
        frame_type="feature_preview",
        notes="face normal plus edge direction",
    )

    assert frame.body == "demo_bracket"
    assert frame.name == "mount_face"
    np.testing.assert_allclose(frame.origin, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))
    assert_orthonormal_right_handed(frame.rotation_matrix)
