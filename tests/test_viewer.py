import json

import numpy as np
import pytest

from chrono_frame_builder.core.body import Body
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.project import Project
from chrono_frame_builder.core.transform import frame_from_three_points
from chrono_frame_builder.viewer import (
    ThreePointPickState,
    build_arg_parser,
    collect_geometry,
    frame_axis_segments,
    geometry_warnings,
    is_supported_geometry_file,
    load_project_for_viewer,
    resolve_project_file,
)

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
