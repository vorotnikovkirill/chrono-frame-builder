import numpy as np
import pytest

from chrono_frame_builder.core.features import (
    mesh_edge_snap_candidate,
    mesh_vertex_candidate,
    triangle_face_candidate,
)
from chrono_frame_builder.core.marker_editor import (
    CreateFrameEditorState,
    create_frame_editor_status_message,
    create_frame_editor_summary,
)


def make_vertex():
    return mesh_vertex_candidate(np.array([0.0, 0.0, 0.0]), source_ids=(1,))


def make_edge():
    return mesh_edge_snap_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        source_ids=(2,),
    )


def make_y_edge():
    return mesh_edge_snap_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        source_ids=(4,),
    )


def make_face():
    return triangle_face_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        source_ids=(3,),
    )


def test_click_creates_markers_with_incrementing_names_and_global_orientation():
    state = CreateFrameEditorState()

    first = state.create_marker_from_feature(make_vertex())
    second = state.create_marker_from_feature(make_edge())

    assert [marker.name for marker in state.markers] == ["marker_001", "marker_002"]
    assert state.selected_marker_id == second.id
    assert first.origin_source == "feature"
    assert first.orientation_source == "reference"
    np.testing.assert_allclose(first.rotation_matrix, np.eye(3))
    np.testing.assert_allclose(second.origin, [1.0, 0.0, 0.0])


def test_new_marker_mode_creates_one_marker_then_returns_to_select_edit_mode():
    state = CreateFrameEditorState()

    assert state.click_mode == "select_edit"
    state.begin_new_marker()
    assert state.click_mode == "create_marker"
    marker = state.handle_geometry_pick(make_vertex())

    assert marker.name == "marker_001"
    assert len(state.markers) == 1
    assert state.click_mode == "select_edit"
    assert state.handle_geometry_pick(make_edge()) is None
    assert len(state.markers) == 1


def test_new_marker_mode_can_intentionally_create_a_second_marker():
    state = CreateFrameEditorState()

    state.begin_new_marker()
    state.handle_geometry_pick(make_vertex())
    state.begin_new_marker()
    second = state.handle_geometry_pick(make_edge())

    assert [marker.name for marker in state.markers] == ["marker_001", "marker_002"]
    assert second.name == "marker_002"
    assert state.click_mode == "select_edit"


def test_marker_selection_rename_and_position_edit_update_selected_marker():
    state = CreateFrameEditorState()
    first = state.create_marker_from_feature(make_vertex())
    second = state.create_marker_from_feature(make_edge())

    selected = state.select_marker(first.id)
    state.rename_selected_marker("tool_tip")
    state.apply_selected_position(np.array([4.0, 5.0, 6.0]))
    summary = create_frame_editor_summary(state)

    assert selected is first
    assert second.name == "marker_002"
    assert first.name == "tool_tip"
    assert first.origin_source == "manual"
    np.testing.assert_allclose(first.origin, [4.0, 5.0, 6.0])
    assert summary["selected_marker"] == "tool_tip"


def test_editor_status_messages_match_the_active_workflow_step():
    state = CreateFrameEditorState()

    assert create_frame_editor_status_message(state) == "Press New Marker, then click geometry."
    state.begin_new_marker()
    assert create_frame_editor_status_message(state) == "Click geometry to create a preview marker."
    state.handle_geometry_pick(make_vertex())
    state.begin_pick_primary_vector()
    assert (
        create_frame_editor_status_message(state)
        == "Click a face for normal or edge for tangent."
    )
    state.handle_geometry_pick(make_face())
    assert (
        create_frame_editor_status_message(state)
        == "Review candidate, then Apply / Flip / Cancel."
    )


def test_manual_rotation_edit_accepts_valid_matrix_and_rejects_invalid_matrices():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    state.apply_selected_rotation(rotation)
    with pytest.raises(ValueError, match="orthonormal"):
        state.apply_selected_rotation(np.ones((3, 3)))
    with pytest.raises(ValueError, match="determinant"):
        state.apply_selected_rotation(np.diag([-1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        state.apply_selected_rotation(
            np.array(
                [
                    [np.nan, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    assert marker.orientation_source == "manual"
    np.testing.assert_allclose(marker.rotation_matrix, rotation)


def test_feature_axis_assignments_use_face_normal_and_edge_tangent_with_projection():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.select_feature(make_face())
    state.use_selected_feature_for_primary_axis()
    state.select_feature(make_edge())
    state.use_selected_feature_for_secondary_axis()

    assert marker.primary_axis_source_mode == "feature"
    assert marker.secondary_axis_source_mode == "feature"
    assert marker.orientation_source == "feature"
    np.testing.assert_allclose(marker.rotation_matrix, np.eye(3))


def test_secondary_feature_vector_is_projected_against_the_primary_direction():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    slanted_edge = mesh_edge_snap_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 1.0]),
    )

    state.select_feature(make_face())
    state.use_selected_feature_for_primary_axis()
    state.select_feature(slanted_edge)
    state.use_selected_feature_for_secondary_axis()

    np.testing.assert_allclose(marker.rotation_matrix, np.eye(3))


def test_selected_point_is_rejected_as_an_axis_vector_source():
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    state.select_feature(make_vertex())

    with pytest.raises(ValueError, match="does not provide a direction"):
        state.use_selected_feature_for_primary_axis()


def test_reference_axis_mode_recomputes_identity_orientation():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.set_primary_axis_source_mode("reference")
    state.set_secondary_axis_source_mode("reference")

    assert marker.orientation_source == "reference"
    np.testing.assert_allclose(marker.rotation_matrix, np.eye(3))


def test_reference_source_axes_align_marker_axes_with_global_axes():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.set_secondary_reference_axis("+Y")
    state.set_primary_reference_axis("+X")

    expected_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(marker.rotation_matrix, expected_rotation)
    np.testing.assert_allclose(marker.rotation_matrix[:, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(marker.rotation_matrix[:, 0], [0.0, 1.0, 0.0])


def test_reference_source_axis_changes_recompute_orientation_and_reject_invalid_pairs():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    initial_rotation = marker.rotation_matrix.copy()
    state.set_secondary_reference_axis("+Y")
    assert not np.allclose(marker.rotation_matrix, initial_rotation)
    state.set_primary_reference_axis("+X")
    rotated = marker.rotation_matrix.copy()
    state.set_primary_reference_axis("+Z")

    assert not np.allclose(marker.rotation_matrix, rotated)
    with pytest.raises(ValueError, match="parallel"):
        state.set_secondary_reference_axis("+Z")
    with pytest.raises(ValueError, match="different frame axes"):
        state.set_secondary_axis("+Z")


def test_flipping_geometric_axis_directions_changes_marker_orientation():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    state.select_feature(make_face())
    state.use_selected_feature_for_primary_axis()
    state.select_feature(make_edge())
    state.use_selected_feature_for_secondary_axis()

    state.flip_primary_direction()
    primary_flipped = marker.rotation_matrix.copy()
    state.flip_primary_direction()
    state.flip_secondary_direction()

    np.testing.assert_allclose(primary_flipped, np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(marker.rotation_matrix, np.diag([-1.0, -1.0, 1.0]))


def test_face_pick_creates_pending_primary_candidate_without_applying_it():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    original_rotation = marker.rotation_matrix.copy()

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())

    assert len(state.markers) == 1
    assert state.click_mode == "pick_primary_vector"
    assert state.pending_axis_candidate is not None
    assert state.pending_axis_candidate.role == "primary"
    assert state.pending_axis_candidate.candidate_type == "face_normal"
    assert marker.primary_feature is None
    np.testing.assert_allclose(marker.rotation_matrix, original_rotation)


def test_edge_pick_creates_pending_primary_candidate_without_creating_marker():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_edge())

    assert len(state.markers) == 1
    assert state.pending_axis_candidate is not None
    assert state.pending_axis_candidate.candidate_type == "edge_tangent"
    assert marker.primary_feature is None


def test_apply_pending_candidates_assigns_primary_and_secondary_features():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())
    state.apply_pending_axis_candidate()

    assert marker.primary_axis_source_mode == "feature"
    assert marker.primary_feature is not None
    assert state.pending_axis_candidate is None
    assert state.click_mode == "select_edit"

    state.begin_pick_secondary_vector()
    state.handle_geometry_pick(make_edge())
    state.apply_pending_axis_candidate()

    assert marker.secondary_feature is not None
    assert marker.secondary_axis_source_mode == "feature"


def test_flip_pending_candidate_reverses_preview_vector_without_changing_marker():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    original_rotation = marker.rotation_matrix.copy()

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())
    pending = state.pending_axis_candidate
    original_vector = pending.vector.copy()

    state.flip_pending_axis_candidate()

    np.testing.assert_allclose(pending.vector, -original_vector)
    assert pending.flipped is True
    np.testing.assert_allclose(marker.rotation_matrix, original_rotation)


def test_cancel_pending_candidate_leaves_marker_unchanged():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    original_rotation = marker.rotation_matrix.copy()

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())
    state.cancel_pending_axis_candidate()

    assert state.pending_axis_candidate is None
    assert state.click_mode == "select_edit"
    assert marker.primary_feature is None
    np.testing.assert_allclose(marker.rotation_matrix, original_rotation)


def test_quick_edge_and_face_picks_stage_primary_candidates_for_selected_local_axis():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_quick_primary_edge_direction("+Z")
    state.handle_geometry_pick(make_edge())

    assert marker.primary_axis == "+Z"
    assert state.click_mode == "quick_pick_primary_edge"
    assert state.pending_axis_candidate.candidate_type == "edge_tangent"
    assert len(state.markers) == 1

    state.cancel_pending_axis_candidate()
    state.begin_quick_primary_face_normal("-Y")
    state.handle_geometry_pick(make_face())

    assert marker.primary_axis == "-Y"
    assert state.pending_axis_candidate.candidate_type == "face_normal"
    assert len(state.markers) == 1


def test_quick_line_pick_stages_vector_from_point_a_to_point_b():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_quick_primary_line("+Z")
    state.handle_geometry_pick(mesh_vertex_candidate(np.array([1.0, 2.0, 3.0])))

    np.testing.assert_allclose(state.line_start_point, [1.0, 2.0, 3.0])
    assert state.click_mode == "pick_line_point_b"
    assert state.pending_axis_candidate is None

    state.handle_geometry_pick(mesh_vertex_candidate(np.array([1.0, 2.0, 5.0])))

    assert marker.primary_axis == "+Z"
    assert state.pending_axis_candidate.candidate_type == "line_between_points"
    np.testing.assert_allclose(state.pending_axis_candidate.anchor, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.pending_axis_candidate.vector, [0.0, 0.0, 1.0])
    assert len(state.markers) == 1


def test_quick_line_pick_rejects_identical_points_without_creating_marker():
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    point = mesh_vertex_candidate(np.array([1.0, 2.0, 3.0]))

    state.begin_quick_primary_line("+Z")
    state.handle_geometry_pick(point)
    with pytest.raises(ValueError, match="must be distinct"):
        state.handle_geometry_pick(point)

    assert state.pending_axis_candidate is None
    assert state.click_mode == "pick_line_point_b"
    assert len(state.markers) == 1


def test_applying_quick_edge_candidate_assigns_primary_source_and_selected_axis():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    state.set_secondary_reference_axis("+Y")

    state.begin_quick_primary_edge_direction("+Z")
    state.handle_geometry_pick(make_edge())
    state.apply_pending_axis_candidate()

    assert marker.primary_axis == "+Z"
    assert marker.primary_axis_source_mode == "feature"
    assert marker.primary_feature.kind == "mesh_edge_snap"
    assert state.click_mode == "select_edit"


def test_quick_primary_parallel_secondary_keeps_source_and_reports_roll_warning():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_quick_primary_edge_direction("+Z")
    state.handle_geometry_pick(make_edge())
    state.apply_pending_axis_candidate()

    assert marker.primary_axis_source_mode == "feature"
    assert marker.primary_feature is not None
    assert state.warning == (
        "Primary direction set. Choose a non-parallel Secondary Axis to fully define orientation."
    )


def test_quick_orientation_apply_marks_saved_marker_modified():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.saved_signature = marker._signature()
    marker.saved_frame_full_name = "body.marker_001"
    state.set_secondary_reference_axis("+Y")

    state.begin_quick_primary_edge_direction("+Z")
    state.handle_geometry_pick(make_edge())
    state.apply_pending_axis_candidate()

    assert marker.save_status == "Modified"


def test_invalid_vector_pick_does_not_create_marker_or_leave_pick_mode():
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    state.begin_pick_primary_vector()

    with pytest.raises(ValueError, match="Point/vertex features"):
        state.handle_geometry_pick(make_vertex())

    assert len(state.markers) == 1
    assert state.click_mode == "pick_primary_vector"
    assert state.pending_axis_candidate is None


def test_applying_pending_candidate_marks_a_saved_marker_modified(tmp_path):
    project_path = tmp_path / "project.json"
    project_path.write_text('{"project_name": "preview_only"}\n', encoding="utf-8")
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.saved_signature = marker._signature()
    marker.saved_frame_full_name = "body.marker_001"

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())
    state.apply_pending_axis_candidate()

    assert marker.save_status == "Modified"
    assert project_path.read_text(encoding="utf-8") == '{"project_name": "preview_only"}\n'


def test_inertia_axis_mode_is_explicitly_unavailable():
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())

    with pytest.raises(ValueError, match="not available"):
        state.set_primary_axis_source_mode("inertia")


def test_marker_editor_never_mutates_project_file(tmp_path):
    project_path = tmp_path / "project.json"
    project_path.write_text('{"project_name": "preview_only"}\n', encoding="utf-8")
    original_contents = project_path.read_text(encoding="utf-8")
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    state.rename_selected_marker("marker_manual")
    state.apply_selected_transform(np.array([1.0, 2.0, 3.0]), np.eye(3))

    assert project_path.read_text(encoding="utf-8") == original_contents
