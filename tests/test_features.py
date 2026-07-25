import numpy as np
import pytest

from chrono_frame_builder.core.features import (
    edge_midpoint,
    edge_tangent,
    mesh_edge_candidate,
    point_candidate,
    triangle_center,
    triangle_face_candidate,
    triangle_normal,
)


def test_triangle_face_center_and_normal_for_right_triangle():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([2.0, 0.0, 0.0])
    p2 = np.array([0.0, 2.0, 0.0])

    np.testing.assert_allclose(triangle_center(p0, p1, p2), [2.0 / 3.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(triangle_normal(p0, p1, p2), [0.0, 0.0, 1.0])


def test_reversed_triangle_winding_reverses_normal_direction():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([2.0, 0.0, 0.0])
    p2 = np.array([0.0, 2.0, 0.0])

    np.testing.assert_allclose(triangle_normal(p0, p2, p1), [0.0, 0.0, -1.0])


def test_edge_midpoint_and_tangent():
    p0 = np.array([1.0, 2.0, 3.0])
    p1 = np.array([5.0, 2.0, 3.0])

    np.testing.assert_allclose(edge_midpoint(p0, p1), [3.0, 2.0, 3.0])
    np.testing.assert_allclose(edge_tangent(p0, p1), [1.0, 0.0, 0.0])


def test_degenerate_triangle_is_rejected():
    with pytest.raises(ValueError, match="Degenerate triangle"):
        triangle_normal(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        )

    with pytest.raises(ValueError, match="Degenerate triangle"):
        triangle_face_candidate(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        )


def test_zero_length_edge_is_rejected():
    with pytest.raises(ValueError, match="Zero-length edge"):
        edge_tangent(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="Zero-length edge"):
        mesh_edge_candidate(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))


def test_triangle_face_candidate_contains_metadata_and_source_information():
    candidate = triangle_face_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        source_ids=(42,),
        metadata={"body": "demo_bracket", "mesh_file": "demo_bracket.stl"},
    )

    assert candidate.kind == "triangle_face"
    assert candidate.source_type == "mesh"
    assert candidate.source_ids == (42,)
    assert candidate.metadata["body"] == "demo_bracket"
    np.testing.assert_allclose(candidate.point, [1.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(candidate.direction, [0.0, 0.0, 1.0])


def test_mesh_edge_candidate_contains_midpoint_tangent_and_metadata():
    candidate = mesh_edge_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
        source_ids=("cell-7", "edge-2"),
        metadata={"source": "picked_mesh_edge"},
    )

    assert candidate.kind == "mesh_edge"
    assert candidate.source_type == "mesh"
    assert candidate.source_ids == ("cell-7", "edge-2")
    assert candidate.metadata["source"] == "picked_mesh_edge"
    np.testing.assert_allclose(candidate.point, [0.0, 0.0, 2.5])
    np.testing.assert_allclose(candidate.direction, [0.0, 0.0, 1.0])


def test_point_candidate_has_origin_point_only():
    candidate = point_candidate(
        np.array([4.0, 5.0, 6.0]),
        source_ids=("vertex-3",),
        metadata={"selection": "mesh_vertex"},
    )

    assert candidate.kind == "point"
    assert candidate.source_type == "mesh"
    assert candidate.direction is None
    assert candidate.source_ids == ("vertex-3",)
    np.testing.assert_allclose(candidate.point, [4.0, 5.0, 6.0])
