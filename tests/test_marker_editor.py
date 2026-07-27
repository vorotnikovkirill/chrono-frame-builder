import numpy as np
import pytest

from chrono_frame_builder.core.features import (
    mesh_edge_snap_candidate,
    mesh_vertex_candidate,
    triangle_face_candidate,
)
from chrono_frame_builder.core.marker_editor import (
    CreateFrameEditorState,
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


def test_pick_vector_modes_assign_features_without_creating_extra_markers():
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    state.begin_pick_primary_vector()
    state.handle_geometry_pick(make_face())
    assert len(state.markers) == 1
    assert state.click_mode == "select_edit"
    assert marker.primary_feature is not None

    state.begin_pick_secondary_vector()
    state.handle_geometry_pick(make_edge())
    assert len(state.markers) == 1
    assert state.click_mode == "select_edit"
    assert marker.secondary_feature is not None


def test_invalid_vector_pick_does_not_create_marker_or_leave_pick_mode():
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    state.begin_pick_primary_vector()

    with pytest.raises(ValueError, match="does not provide a direction"):
        state.handle_geometry_pick(make_vertex())

    assert len(state.markers) == 1
    assert state.click_mode == "pick_primary_vector"


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
