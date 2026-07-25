import json

import numpy as np
import pytest

from chrono_frame_builder.core.body import Body
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.project import Project
from chrono_frame_builder.viewer import (
    collect_geometry,
    frame_axis_segments,
    geometry_warnings,
    is_supported_geometry_file,
    load_project_for_viewer,
    resolve_project_file,
)


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
