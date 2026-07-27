import numpy as np
import pytest

from chrono_frame_builder.core.body import Body
from chrono_frame_builder.core.features import mesh_edge_snap_candidate, mesh_vertex_candidate
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.marker_editor import CreateFrameEditorState
from chrono_frame_builder.core.marker_persistence import (
    preview_marker_to_frame,
    save_selected_preview_marker,
)
from chrono_frame_builder.core.project import Project


def make_project(tmp_path, frames=None):
    project = Project(
        schema_version="0.1.0",
        project_name="marker_persistence",
        units={"length": "mm", "angle": "deg"},
        bodies=[Body(name="bracket", cad_file="bracket.stl")],
        frames=list(frames or []),
    )
    project_path = tmp_path / "project.json"
    project.save(project_path)
    return project, project_path


def make_vertex():
    return mesh_vertex_candidate(np.array([1.0, 2.0, 3.0]))


def test_preview_marker_converts_to_existing_frame_schema(tmp_path):
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.name = "mount_marker"

    frame = preview_marker_to_frame(marker, "bracket")

    assert frame.body == "bracket"
    assert frame.name == "mount_marker"
    np.testing.assert_allclose(frame.origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame.rotation_matrix, np.eye(3))


def test_save_selected_marker_appends_frame_and_preserves_existing_frames(tmp_path):
    existing = Frame(
        body="bracket",
        name="existing",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )
    project, project_path = make_project(tmp_path, frames=[existing])
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.name = "saved_marker"

    saved_frame = save_selected_preview_marker(state, project, project_path)
    reloaded = Project.load(project_path)

    assert saved_frame.full_name == "bracket.saved_marker"
    assert marker.save_status == "Saved"
    assert marker.saved_frame_full_name == "bracket.saved_marker"
    assert [frame.name for frame in reloaded.frames] == ["existing", "saved_marker"]
    np.testing.assert_allclose(reloaded.frames[-1].origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(reloaded.frames[-1].rotation_matrix, np.eye(3))


def test_save_selected_marker_rejects_duplicate_or_empty_names(tmp_path):
    existing = Frame(
        body="bracket",
        name="existing",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )
    project, project_path = make_project(tmp_path, frames=[existing])
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.name = "existing"

    with pytest.raises(ValueError, match="already exists"):
        save_selected_preview_marker(state, project, project_path)

    marker.name = ""
    with pytest.raises(ValueError, match="must not be empty"):
        save_selected_preview_marker(state, project, project_path)

    assert len(project.frames) == 1


def test_save_selected_marker_rejects_invalid_rotation_without_writing(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    marker.rotation_matrix = np.ones((3, 3))
    original_contents = project_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="orthonormal"):
        save_selected_preview_marker(state, project, project_path)

    assert project_path.read_text(encoding="utf-8") == original_contents
    assert marker.save_status == "Unsaved"


def test_save_selected_marker_only_persists_selected_preview_marker(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    first = state.create_marker_from_feature(make_vertex())
    second = state.create_marker_from_feature(
        mesh_vertex_candidate(np.array([4.0, 5.0, 6.0]))
    )

    save_selected_preview_marker(state, project, project_path)

    assert [frame.name for frame in project.frames] == [second.name]
    assert first.save_status == "Unsaved"
    assert second.save_status == "Saved"


def test_saved_marker_status_becomes_modified_after_transform_edit(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())

    save_selected_preview_marker(state, project, project_path)
    state.apply_selected_position(np.array([7.0, 8.0, 9.0]))

    assert state.selected_marker.save_status == "Modified"


def test_quick_orientation_edit_updates_the_linked_saved_frame(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)
    state.set_secondary_reference_axis("+Y")
    edge = mesh_edge_snap_candidate(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    )

    state.begin_quick_primary_edge_direction("+Z")
    state.handle_geometry_pick(edge)
    state.apply_pending_axis_candidate()
    save_selected_preview_marker(state, project, project_path)
    reloaded = Project.load(project_path)

    assert marker.save_status == "Saved"
    assert len(reloaded.frames) == 1
    np.testing.assert_allclose(reloaded.frames[0].rotation_matrix[:, 2], [1.0, 0.0, 0.0])


def test_saved_marker_save_without_changes_is_a_no_op(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())

    first_save = save_selected_preview_marker(state, project, project_path)
    second_save = save_selected_preview_marker(state, project, project_path)

    assert second_save is first_save
    assert [frame.name for frame in project.frames] == [marker.name]
    assert marker.save_status == "Saved"


def test_modified_saved_marker_updates_its_frame_and_preserves_unrelated_frames(tmp_path):
    existing = Frame(
        body="bracket",
        name="existing",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )
    project, project_path = make_project(tmp_path, frames=[existing])
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    state.apply_selected_transform(np.array([7.0, 8.0, 9.0]), rotation)
    assert state.selected_marker.save_status == "Modified"
    save_selected_preview_marker(state, project, project_path)
    reloaded = Project.load(project_path)

    assert [frame.name for frame in reloaded.frames] == ["existing", "marker_001"]
    np.testing.assert_allclose(reloaded.frames[0].origin, np.zeros(3))
    np.testing.assert_allclose(reloaded.frames[1].origin, [7.0, 8.0, 9.0])
    np.testing.assert_allclose(reloaded.frames[1].rotation_matrix, rotation)
    assert state.selected_marker.save_status == "Saved"


def test_saved_marker_rename_updates_its_linked_frame_in_place(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    marker = state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)

    state.rename_selected_marker("test_frame_001")
    save_selected_preview_marker(state, project, project_path)
    reloaded = Project.load(project_path)

    assert [frame.name for frame in reloaded.frames] == ["test_frame_001"]
    assert marker.saved_frame_full_name == "bracket.test_frame_001"
    assert marker.save_status == "Saved"


def test_saved_marker_rename_rejects_another_project_frame_name(tmp_path):
    existing = Frame(
        body="bracket",
        name="existing",
        origin=np.zeros(3),
        rotation_matrix=np.eye(3),
    )
    project, project_path = make_project(tmp_path, frames=[existing])
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)

    state.rename_selected_marker("existing")
    with pytest.raises(ValueError, match="already exists"):
        save_selected_preview_marker(state, project, project_path)

    assert [frame.name for frame in project.frames] == ["existing", "marker_001"]
    assert state.selected_marker.save_status == "Modified"


def test_saved_marker_with_missing_link_is_rejected_without_appending(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)
    project.frames.clear()
    state.apply_selected_position(np.array([7.0, 8.0, 9.0]))

    with pytest.raises(ValueError, match="was not found"):
        save_selected_preview_marker(state, project, project_path)

    assert project.frames == []
    assert state.selected_marker.save_status == "Modified"


def test_saved_marker_orientation_controls_mark_it_modified(tmp_path):
    project, project_path = make_project(tmp_path)
    state = CreateFrameEditorState()
    state.create_marker_from_feature(make_vertex())
    save_selected_preview_marker(state, project, project_path)

    state.set_primary_reference_axis("-Z")

    assert state.selected_marker.save_status == "Modified"
