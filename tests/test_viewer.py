import inspect
import json

import numpy as np
import pytest

from chrono_frame_builder.core.body import Body
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.project import Project
from chrono_frame_builder.core.transform import frame_from_three_points
from chrono_frame_builder.viewer import (
    CreateFrameWorkflowState,
    FeatureInspectionState,
    ThreePointPickState,
    build_arg_parser,
    collect_geometry,
    create_frame_control_summary,
    create_frame_feature_labels,
    create_frame_preview_style_from_bounds,
    create_frame_status_summary,
    feature_candidate_from_pick,
    feature_candidate_signature,
    feature_direction_vector,
    feature_origin_point,
    feature_preview_labels,
    feature_preview_style_from_bounds,
    frame_axis_segments,
    geometry_bounds_diagonal,
    geometry_warnings,
    is_supported_geometry_file,
    load_project_for_viewer,
    point_feature_candidate_from_pick,
    resolve_project_file,
    snap_feature_candidate_from_pick,
    triangle_feature_candidate_from_vertices,
)
from chrono_frame_builder.viewer_qt import qt_ui_status, render_create_frame_qt

VIEWER_DEMO_PROJECT = "examples/viewer_demo/project.json"


def make_project(**overrides):
    data = {
        "schema_version": "0.1.0",
        "project_name": "viewer_test",
        "units": {"length": "mm", "angle": "deg"},
        "bodies": [],
        "frames": [],
    }
    data.update(overrides)
    return Project(
        schema_version=data["schema_version"],
        project_name=data["project_name"],
        units=data["units"],
        bodies=data["bodies"],
        frames=data["frames"],
    )


def test_frame_axis_segments_use_rotation_matrix_columns():
    frame = Frame(
        body="arm",
        name="tool",
        origin=np.array([10.0, 20.0, 30.0]),
        rotation_matrix=np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    segments = frame_axis_segments(frame, axis_length=5.0)

    assert [segment.axis_name for segment in segments] == ["x", "y", "z"]
    np.testing.assert_allclose(segments[0].start, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(segments[0].end, [10.0, 25.0, 30.0])
    np.testing.assert_allclose(segments[1].end, [5.0, 20.0, 30.0])
    np.testing.assert_allclose(segments[2].end, [10.0, 20.0, 35.0])


def test_frame_axis_segments_reject_non_positive_axis_length():
    frame = Frame(
        body="arm",
        name="tool",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )

    with pytest.raises(ValueError, match="axis_length"):
        frame_axis_segments(frame, axis_length=0.0)


def test_frame_from_three_points_creates_right_handed_frame():
    frame = frame_from_three_points(
        "arm",
        "preview",
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 3.0]),
        np.array([1.0, 5.0, 3.0]),
    )

    np.testing.assert_allclose(frame.origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_frame_from_three_points_rejects_degenerate_points():
    with pytest.raises(ValueError, match="distinct"):
        frame_from_three_points(
            "arm",
            "preview",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    with pytest.raises(ValueError, match="collinear"):
        frame_from_three_points(
            "arm",
            "preview",
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        )


def test_three_point_pick_state_returns_preview_after_third_valid_point():
    state = ThreePointPickState(points=[])

    assert state.next_point_label == "P0"
    assert state.add_point(np.array([0.0, 0.0, 0.0])) is None
    assert state.next_point_label == "P1"
    assert state.add_point(np.array([1.0, 0.0, 0.0])) is None
    frame = state.add_point(np.array([0.0, 1.0, 0.0]))

    assert frame is not None
    assert state.points == []
    assert state.warning is None
    np.testing.assert_allclose(frame.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_three_point_pick_state_resets_after_degenerate_points():
    state = ThreePointPickState(points=[])

    assert state.add_point(np.array([0.0, 0.0, 0.0])) is None
    assert state.add_point(np.array([1.0, 0.0, 0.0])) is None
    assert state.add_point(np.array([2.0, 0.0, 0.0])) is None

    assert state.points == []
    assert state.preview_frame is None
    assert state.warning is not None
    assert "collinear" in state.warning


def test_build_arg_parser_accepts_pick_frame():
    args = build_arg_parser().parse_args(
        ["examples/viewer_demo/project.json", "--pick-frame", "--axis-length", "12.5"]
    )

    assert args.project == "examples/viewer_demo/project.json"
    assert args.pick_frame is True
    assert args.axis_length == 12.5


def test_build_arg_parser_accepts_feature_inspection():
    args = build_arg_parser().parse_args(
        ["examples/viewer_demo/project.json", "--inspect-features"]
    )

    assert args.project == "examples/viewer_demo/project.json"
    assert args.inspect_features is True


def test_build_arg_parser_accepts_create_frame():
    args = build_arg_parser().parse_args(["examples/viewer_demo/project.json", "--create-frame"])

    assert args.project == "examples/viewer_demo/project.json"
    assert args.create_frame is True


def test_triangle_feature_candidate_from_vertices():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    candidate = triangle_feature_candidate_from_vertices(
        vertices,
        body_name="demo_bracket",
        mesh_file="demo_bracket.stl",
        cell_id=7,
    )

    assert candidate.kind == "triangle_face"
    assert candidate.source_type == "mesh"
    assert candidate.source_ids == (7,)
    assert candidate.metadata["body_name"] == "demo_bracket"
    assert candidate.metadata["mesh_file"] == "demo_bracket.stl"
    np.testing.assert_allclose(candidate.point, [2.0 / 3.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(candidate.direction, [0.0, 0.0, 1.0])


def test_triangle_feature_candidate_rejects_unsupported_cell_vertices():
    quad_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="triangle cell"):
        triangle_feature_candidate_from_vertices(quad_vertices)


def test_feature_candidate_from_pick_falls_back_for_missing_triangle_vertices():
    candidate = feature_candidate_from_pick(
        np.array([1.0, 2.0, 3.0]),
        body_name="demo_bracket",
        mesh_file="demo_bracket.stl",
    )

    assert candidate.kind == "point"
    assert candidate.direction is None
    assert candidate.metadata["body_name"] == "demo_bracket"
    assert candidate.metadata["mesh_file"] == "demo_bracket.stl"
    assert "fallback_reason" in candidate.metadata
    np.testing.assert_allclose(candidate.point, [1.0, 2.0, 3.0])


def test_feature_candidate_from_pick_falls_back_for_invalid_triangle_vertices():
    candidate = feature_candidate_from_pick(
        np.array([1.0, 2.0, 3.0]),
        triangle_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    assert candidate.kind == "point"
    assert "triangle cell" in candidate.metadata["fallback_reason"]
    np.testing.assert_allclose(candidate.point, [1.0, 2.0, 3.0])


def test_point_feature_candidate_from_pick_records_fallback_reason():
    candidate = point_feature_candidate_from_pick(
        np.array([4.0, 5.0, 6.0]),
        reason="no cell id available",
    )

    assert candidate.kind == "point"
    assert candidate.metadata["fallback_reason"] == "no cell id available"
    np.testing.assert_allclose(candidate.point, [4.0, 5.0, 6.0])


def test_snap_feature_candidate_from_pick_prefers_edge_near_triangle_edge():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    candidate = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]),
        triangle_vertices=vertices,
        body_name="demo_bracket",
        cell_id=4,
    )

    assert candidate.kind == "mesh_edge_snap"
    assert candidate.source_ids == (4, "edge:0")
    assert candidate.metadata["body_name"] == "demo_bracket"
    np.testing.assert_allclose(candidate.midpoint, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(candidate.tangent, [1.0, 0.0, 0.0])


def test_snap_feature_candidate_from_pick_prefers_vertex_near_triangle_vertex():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    candidate = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=vertices,
        body_name="demo_bracket",
        cell_id=4,
    )

    assert candidate.kind == "mesh_vertex"
    assert candidate.source_ids == (4, "vertex:0")
    assert candidate.metadata["body_name"] == "demo_bracket"
    np.testing.assert_allclose(candidate.point, [0.0, 0.0, 0.0])


def test_snap_feature_candidate_from_pick_uses_face_for_triangle_interior():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    candidate = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]),
        triangle_vertices=vertices,
        cell_id=4,
    )

    assert candidate.kind == "triangle_face"
    assert candidate.source_ids == (4,)
    np.testing.assert_allclose(candidate.point, [2.0 / 3.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(candidate.direction, [0.0, 0.0, 1.0])


def test_snap_feature_candidate_from_pick_falls_back_for_invalid_cell_data():
    candidate = snap_feature_candidate_from_pick(
        np.array([0.25, 0.25, 0.0]),
        triangle_vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    assert candidate.kind == "point"
    assert "not a triangle" in candidate.metadata["fallback_reason"]


def test_feature_inspection_state_suppresses_duplicate_candidates():
    state = FeatureInspectionState()
    candidate = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        cell_id=4,
    )

    assert state.should_print(candidate) is True
    assert state.should_print(candidate) is False
    assert feature_candidate_signature(candidate) == state.last_signature


def test_geometry_bounds_diagonal_combines_mesh_bounds():
    diagonal = geometry_bounds_diagonal(
        [
            (0.0, 3.0, 0.0, 4.0, 0.0, 0.0),
            (3.0, 6.0, 0.0, 4.0, 0.0, 8.0),
        ]
    )

    assert diagonal == pytest.approx(10.770329614269007)


def test_feature_preview_style_scales_from_mesh_bounds():
    style = feature_preview_style_from_bounds([(0.0, 6.0, 0.0, 8.0, 0.0, 0.0)])

    assert style.marker_radius == pytest.approx(0.25)
    assert style.vector_length == pytest.approx(1.8)
    assert style.edge_line_width == 8
    assert style.label_font_size == 14


def test_create_frame_preview_style_is_less_intrusive_than_debug_inspection_style():
    bounds = [(0.0, 6.0, 0.0, 8.0, 0.0, 0.0)]
    inspect_style = feature_preview_style_from_bounds(bounds)
    create_style = create_frame_preview_style_from_bounds(bounds)

    assert create_style.marker_radius < inspect_style.marker_radius
    assert create_style.vector_length < inspect_style.vector_length
    assert create_style.edge_line_width < inspect_style.edge_line_width


def test_feature_preview_labels_for_edge_face_and_vertex_candidates():
    style = feature_preview_style_from_bounds([(0.0, 10.0, 0.0, 0.0, 0.0, 0.0)])
    edge_candidate = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        cell_id=4,
    )
    edge_labels = feature_preview_labels(edge_candidate, style)

    assert [label for _point, label in edge_labels] == ["start", "midpoint", "end", "tangent"]
    np.testing.assert_allclose(edge_labels[3][0], [2.8, 0.0, 0.0])

    face_candidate = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    face_labels = feature_preview_labels(face_candidate, style)

    assert [label for _point, label in face_labels] == ["face center", "normal"]
    np.testing.assert_allclose(face_labels[1][0], [2.0 / 3.0, 2.0 / 3.0, 1.8])

    vertex_candidate = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    vertex_labels = feature_preview_labels(vertex_candidate, style)

    assert [label for _point, label in vertex_labels] == ["vertex"]
    np.testing.assert_allclose(vertex_labels[0][0], [0.0, 0.0, 0.0])


def test_create_frame_state_assigns_selected_feature_as_origin():
    state = CreateFrameWorkflowState()
    origin = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    assert state.select_feature(origin) is True
    state.assign_selected_as_origin()

    assert state.step == "Step 2: Primary axis"
    np.testing.assert_allclose(feature_origin_point(state.origin_feature), [0.0, 0.0, 0.0])


def test_create_frame_state_assigns_primary_and_secondary_vector_sources():
    state = CreateFrameWorkflowState()
    edge = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    face = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    state.select_feature(face)
    state.assign_selected_as_primary()
    state.select_feature(edge)
    state.assign_selected_as_secondary()

    assert state.step == "Step 4: Preview"
    np.testing.assert_allclose(state.preview_frame.rotation_matrix, np.eye(3))
    np.testing.assert_allclose(feature_direction_vector(state.primary_feature), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(feature_direction_vector(state.secondary_feature), [1.0, 0.0, 0.0])


def test_selected_feature_creates_global_preview_at_its_origin():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    assert state.select_feature(vertex) is True

    assert state.origin_feature is vertex
    assert state.origin_source == "feature"
    assert state.orientation_source == "global"
    np.testing.assert_allclose(state.preview_frame.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.preview_frame.rotation_matrix, np.eye(3))


def test_reset_orientation_restores_global_preview_after_vector_roles_are_assigned():
    state = CreateFrameWorkflowState()
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    origin = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]), triangle_vertices=triangle
    )
    face = snap_feature_candidate_from_pick(np.array([0.65, 0.65, 0.0]), triangle_vertices=triangle)
    edge = snap_feature_candidate_from_pick(np.array([0.75, 0.05, 0.0]), triangle_vertices=triangle)

    state.select_feature(origin)
    state.select_feature(face)
    state.assign_selected_as_primary()
    state.select_feature(edge)
    state.assign_selected_as_secondary()
    frame = state.reset_orientation_to_global()

    assert state.primary_feature is None
    assert state.secondary_feature is None
    np.testing.assert_allclose(frame.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_assigning_both_vectors_recomputes_the_feature_preview_frame():
    state = CreateFrameWorkflowState()
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    origin = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]), triangle_vertices=triangle
    )
    face = snap_feature_candidate_from_pick(np.array([0.65, 0.65, 0.0]), triangle_vertices=triangle)
    edge = snap_feature_candidate_from_pick(np.array([0.75, 0.05, 0.0]), triangle_vertices=triangle)

    state.select_feature(origin)
    state.select_feature(face)
    state.assign_selected_as_primary()
    state.select_feature(edge)
    state.assign_selected_as_secondary()

    assert state.preview_frame is not None
    assert state.orientation_source == "feature"
    np.testing.assert_allclose(state.preview_frame.rotation_matrix, np.eye(3))


def test_manual_transform_updates_preview_origin_and_rotation():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.select_feature(vertex)
    rotation_matrix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    frame = state.apply_manual_transform(np.array([4.0, 5.0, 6.0]), rotation_matrix)

    assert state.origin_source == "manual"
    assert state.orientation_source == "manual"
    np.testing.assert_allclose(frame.origin, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(frame.rotation_matrix, rotation_matrix)


def test_manual_position_override_preserves_orientation_and_updates_source():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.select_feature(vertex)

    frame = state.apply_manual_position(np.array([1.5, 2.5, 3.5]))

    assert state.origin_source == "manual"
    assert state.orientation_source == "global"
    np.testing.assert_allclose(frame.origin, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_manual_transform_rejects_non_orthonormal_and_reflected_matrices():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.select_feature(vertex)
    original_origin = state.preview_frame.origin.copy()
    original_rotation = state.preview_frame.rotation_matrix.copy()

    with pytest.raises(ValueError, match="orthonormal"):
        state.apply_manual_transform(np.array([1.0, 2.0, 3.0]), np.ones((3, 3)))
    with pytest.raises(ValueError, match="determinant"):
        state.apply_manual_transform(
            np.array([1.0, 2.0, 3.0]),
            np.diag([-1.0, 1.0, 1.0]),
        )

    np.testing.assert_allclose(state.preview_frame.origin, original_origin)
    np.testing.assert_allclose(state.preview_frame.rotation_matrix, original_rotation)


def test_marker_name_updates_preview_and_control_summary():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.select_feature(vertex)

    state.set_marker_name("tool_marker")
    summary = create_frame_control_summary(state)

    assert state.preview_frame.name == "tool_marker"
    assert summary["marker_name"] == "tool_marker"
    assert summary["position"] == "[0. 0. 0.]"
    assert "[[1." in summary["rotation_matrix"]


def test_create_frame_state_rejects_point_feature_as_vector_source():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    state.select_feature(vertex)

    with pytest.raises(ValueError, match="does not provide a direction"):
        state.assign_selected_as_primary()


def test_create_frame_state_builds_preview_frame_from_roles():
    state = CreateFrameWorkflowState()
    origin = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    face = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    edge = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    state.select_feature(origin)
    state.assign_selected_as_origin()
    state.select_feature(face)
    state.assign_selected_as_primary()
    state.select_feature(edge)
    state.assign_selected_as_secondary()
    frame = state.build_preview_frame()

    assert state.step == "Step 4: Preview"
    np.testing.assert_allclose(frame.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_create_frame_state_rejects_preview_with_missing_roles_or_same_axes():
    state = CreateFrameWorkflowState()

    with pytest.raises(ValueError, match="origin"):
        state.build_preview_frame()

    state.origin_feature = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.primary_feature = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.secondary_feature = state.primary_feature
    state.secondary_axis = "-Z"

    with pytest.raises(ValueError, match="different underlying axes"):
        state.build_preview_frame()


def test_create_frame_status_summary_includes_roles_and_axes():
    state = CreateFrameWorkflowState()

    summary = create_frame_status_summary(state)

    assert "Step 1: Origin" in summary
    assert "primary +Z" in summary
    assert "secondary +X" in summary


def test_create_frame_control_summary_exposes_panel_fields_without_tk():
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    state.select_feature(vertex)
    state.assign_selected_as_origin()

    summary = create_frame_control_summary(state)

    assert summary["step"] == "Step 2: Primary axis"
    assert "mesh_vertex" in summary["selected"]
    assert "mesh_vertex" in summary["origin"]
    assert summary["primary"].startswith("+Z:")
    assert summary["secondary"].startswith("+X:")
    assert "no project files" in summary["message"]


def test_create_frame_state_validates_axis_selectors_and_clears_preview():
    state = CreateFrameWorkflowState()
    state.preview_frame = Frame(
        body="__preview__",
        name="feature_preview",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )

    assert state.set_primary_axis("-Y") == "-Y"
    assert state.set_secondary_axis("+Z") == "+Z"
    assert state.preview_frame is None

    with pytest.raises(ValueError, match="Invalid primary axis selector"):
        state.set_primary_axis("X")
    with pytest.raises(ValueError, match="Invalid secondary axis selector"):
        state.set_secondary_axis("roll")


def test_create_frame_feature_labels_are_compact_for_edge_and_face_candidates():
    style = create_frame_preview_style_from_bounds([(0.0, 2.0, 0.0, 2.0, 0.0, 1.0)])
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    edge = snap_feature_candidate_from_pick(
        np.array([0.75, 0.05, 0.0]), triangle_vertices=triangle
    )
    face = snap_feature_candidate_from_pick(
        np.array([0.65, 0.65, 0.0]), triangle_vertices=triangle
    )

    assert [label for _point, label in create_frame_feature_labels(edge, style)] == [
        "edge",
        "tangent",
    ]
    assert [label for _point, label in create_frame_feature_labels(face, style)] == [
        "face",
        "normal",
    ]


def test_create_frame_workflow_state_does_not_mutate_project_file(tmp_path):
    project_path = tmp_path / "project.json"
    project_path.write_text('{"project_name": "read_only_preview"}\n', encoding="utf-8")
    original_contents = project_path.read_text(encoding="utf-8")
    state = CreateFrameWorkflowState()
    vertex = snap_feature_candidate_from_pick(
        np.array([0.03, 0.04, 0.0]),
        triangle_vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )

    state.select_feature(vertex)
    state.assign_selected_as_origin()
    state.set_primary_axis("-Y")
    state.reset()

    assert project_path.read_text(encoding="utf-8") == original_contents


def test_qt_ui_status_is_safe_without_optional_ui_dependencies():
    available, message = qt_ui_status()

    assert isinstance(available, bool)
    assert "Qt UI" in message


def test_qt_apply_properties_button_wiring_keeps_widget_and_handler_distinct():
    source = inspect.getsource(render_create_frame_qt)

    assert "apply_properties_button.clicked.connect(apply_properties)" in source


def test_viewer_input_validation_for_missing_project(tmp_path):
    with pytest.raises(FileNotFoundError, match="Project file was not found"):
        resolve_project_file(tmp_path / "missing.json")


def test_supported_geometry_extensions_are_case_insensitive():
    assert is_supported_geometry_file("body.STL")
    assert is_supported_geometry_file("body.obj")
    assert is_supported_geometry_file("body.vtp")
    assert not is_supported_geometry_file("body.step")


def test_collect_geometry_reports_missing_and_unsupported_geometry(tmp_path):
    project_path = tmp_path / "project.json"
    project_path.write_text("{}", encoding="utf-8")
    project = make_project(
        bodies=[
            Body(name="missing_mesh", cad_file="cad/missing.stl"),
            Body(name="unsupported_cad", cad_file="cad/body.step"),
            Body(name="no_metadata", cad_file=""),
        ]
    )

    geometry = collect_geometry(project, project_path)

    assert [item.status for item in geometry] == [
        "missing_file",
        "unsupported",
        "missing_metadata",
    ]
    assert geometry[0].path == tmp_path / "cad" / "missing.stl"
    warnings = geometry_warnings(geometry)
    assert "geometry file was not found" in warnings[0]
    assert "not a supported mesh format" in warnings[1]
    assert "no geometry file metadata" in warnings[2]


def test_collect_geometry_resolves_displayable_mesh_relative_to_project(tmp_path):
    mesh_dir = tmp_path / "cad"
    mesh_dir.mkdir()
    mesh_file = mesh_dir / "body.ply"
    mesh_file.write_text("ply\n", encoding="utf-8")
    project_path = tmp_path / "project.json"
    project_path.write_text("{}", encoding="utf-8")
    project = make_project(bodies=[Body(name="arm", cad_file="cad/body.ply")])

    geometry = collect_geometry(project, project_path)

    assert len(geometry) == 1
    assert geometry[0].status == "displayable"
    assert geometry[0].path == mesh_file


def test_viewer_demo_project_loads_with_displayable_geometry():
    project, project_path = load_project_for_viewer(VIEWER_DEMO_PROJECT)
    geometry = collect_geometry(project, project_path)

    assert project.project_name == "viewer_demo"
    assert project.list_body_names() == ["demo_bracket"]
    assert project.list_frame_names() == [
        "demo_bracket.body_origin",
        "demo_bracket.marker_tip",
    ]
    assert len(geometry) == 1
    assert geometry[0].status == "displayable"
    assert geometry[0].path is not None
    assert geometry[0].path.name == "demo_bracket.stl"


def test_viewer_demo_frames_are_available():
    project = Project.load(VIEWER_DEMO_PROJECT)

    body_origin = project.get_frame_by_full_name("demo_bracket.body_origin")
    marker_tip = project.get_frame_by_full_name("demo_bracket.marker_tip")

    np.testing.assert_allclose(body_origin.origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(marker_tip.origin, [80.0, 0.0, 12.0])


def test_load_project_for_viewer_accepts_project_directory(tmp_path):
    project_file = tmp_path / "project.json"
    project_file.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "project_name": "from_directory",
                "units": {"length": "mm", "angle": "deg"},
                "bodies": [],
                "frames": [],
            }
        ),
        encoding="utf-8",
    )

    project, resolved_path = load_project_for_viewer(tmp_path)

    assert project.project_name == "from_directory"
    assert resolved_path == project_file.resolve()
