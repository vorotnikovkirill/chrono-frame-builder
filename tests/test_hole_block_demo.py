import subprocess
import sys
from pathlib import Path

from chrono_frame_builder.core.coordinate_context import coordinate_context_summary
from chrono_frame_builder.core.project import Project

ROOT = Path(__file__).resolve().parents[1]
HOLE_BLOCK_DIRECTORY = ROOT / "examples" / "hole_block"


def test_hole_block_project_and_mesh_load_from_existing_schema():
    project_path = HOLE_BLOCK_DIRECTORY / "project.json"

    project = Project.load(project_path)
    mesh_path = HOLE_BLOCK_DIRECTORY / project.bodies[0].cad_file

    assert project.project_name == "hole_block"
    assert project.bodies[0].name == "hole_block"
    assert [frame.name for frame in project.frames] == ["body_origin", "hole_reference"]
    assert mesh_path.is_file()
    assert "v " in mesh_path.read_text(encoding="utf-8")


def test_hole_block_generator_requires_only_python_standard_library(tmp_path):
    output_path = tmp_path / "generated_hole_block.obj"

    subprocess.run(
        [
            sys.executable,
            str(HOLE_BLOCK_DIRECTORY / "make_hole_block.py"),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    generated = output_path.read_text(encoding="utf-8")
    assert generated.startswith("# Faceted")
    assert generated.count("\nv ") == 192
    assert generated.count("\nf ") == 192


def test_coordinate_context_reports_current_body_local_convention():
    project = Project.load(HOLE_BLOCK_DIRECTORY / "project.json")

    context = coordinate_context_summary(project, "hole_block")

    assert context["parent_body"] == "hole_block"
    assert context["display_coordinates"] == "Body/part-local Cartesian XYZ"
    assert context["saved_coordinates"] == "Body/part-local Cartesian XYZ in project.json"
    assert "Not supported" in context["assembly_transform"]
