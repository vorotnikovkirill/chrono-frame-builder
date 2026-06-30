import pytest

from chrono_frame_builder import FrameNotFoundError, Project


EXAMPLE_PROJECT = "examples/minimal_pendulum/project.json"


def test_load_project():
    project = Project.load(EXAMPLE_PROJECT)

    assert project.project_name == "minimal_pendulum"
    assert project.schema_version == "0.1.0"
    assert project.units["length"] == "mm"
    assert project.units["angle"] == "deg"


def test_list_bodies():
    project = Project.load(EXAMPLE_PROJECT)

    assert project.list_body_names() == ["base", "arm"]


def test_list_frames():
    project = Project.load(EXAMPLE_PROJECT)

    assert project.list_frame_names() == ["base.hinge_A", "arm.hinge_A"]


def test_get_frame():
    project = Project.load(EXAMPLE_PROJECT)

    frame = project.get_frame("base", "hinge_A")

    assert frame.body == "base"
    assert frame.name == "hinge_A"
    assert frame.full_name == "base.hinge_A"


def test_get_frame_by_full_name():
    project = Project.load(EXAMPLE_PROJECT)

    frame = project.get_frame_by_full_name("arm.hinge_A")

    assert frame.body == "arm"
    assert frame.name == "hinge_A"
    assert frame.full_name == "arm.hinge_A"


def test_missing_frame_error():
    project = Project.load(EXAMPLE_PROJECT)

    with pytest.raises(FrameNotFoundError) as error:
        project.get_frame("arm", "missing_frame")

    assert "arm.missing_frame" in str(error.value)
    assert "base.hinge_A" in str(error.value)
    assert "arm.hinge_A" in str(error.value)
