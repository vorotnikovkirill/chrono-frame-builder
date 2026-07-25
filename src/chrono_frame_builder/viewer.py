"""Minimal PyVista viewer for geometry and stored project frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.project import Project

SUPPORTED_GEOMETRY_EXTENSIONS = frozenset({".stl", ".obj", ".ply", ".vtk", ".vtp"})
AXIS_COLORS = {"x": "red", "y": "green", "z": "blue"}


@dataclass(frozen=True)
class GeometryStatus:
    """Resolved geometry metadata for one project body."""

    body_name: str
    source: str
    path: Path | None
    status: str
    color: tuple[float, float, float]
    opacity: float

    @property
    def is_displayable(self) -> bool:
        """Return whether this body has a supported mesh file on disk."""
        return self.status == "displayable"


@dataclass(frozen=True)
class AxisSegment:
    """One rendered frame axis as a line segment."""

    frame_name: str
    axis_name: str
    start: np.ndarray
    end: np.ndarray
    color: str


def resolve_project_file(path: str | Path) -> Path:
    """Resolve a viewer input path to an existing project JSON file."""
    project_path = Path(path).expanduser()
    if project_path.is_dir():
        project_path = project_path / "project.json"

    if not project_path.exists():
        raise FileNotFoundError(f"Project file was not found: {project_path}")

    if not project_path.is_file():
        raise ValueError(f"Project path is not a file: {project_path}")

    return project_path.resolve()


def load_project_for_viewer(path: str | Path) -> tuple[Project, Path]:
    """Load a project and return it with the resolved project file path."""
    project_path = resolve_project_file(path)
    return Project.load(project_path), project_path


def is_supported_geometry_file(path: str | Path) -> bool:
    """Return whether a path has a mesh extension supported by the B0 viewer."""
    return Path(path).suffix.lower() in SUPPORTED_GEOMETRY_EXTENSIONS


def collect_geometry(project: Project, project_path: str | Path) -> list[GeometryStatus]:
    """Collect body geometry metadata and resolve file paths relative to the project file."""
    project_dir = Path(project_path).expanduser().resolve().parent
    geometry = []

    for body in project.bodies:
        source = body.cad_file.strip()
        if not source:
            geometry.append(
                GeometryStatus(
                    body_name=body.name,
                    source=source,
                    path=None,
                    status="missing_metadata",
                    color=body.color,
                    opacity=body.opacity,
                )
            )
            continue

        geometry_path = Path(source).expanduser()
        if not geometry_path.is_absolute():
            geometry_path = project_dir / geometry_path

        if not is_supported_geometry_file(geometry_path):
            status = "unsupported"
        elif not geometry_path.is_file():
            status = "missing_file"
        else:
            status = "displayable"

        geometry.append(
            GeometryStatus(
                body_name=body.name,
                source=source,
                path=geometry_path,
                status=status,
                color=body.color,
                opacity=body.opacity,
            )
        )

    return geometry


def geometry_warnings(geometry: list[GeometryStatus]) -> list[str]:
    """Create user-facing warnings for geometry that cannot be displayed."""
    warnings = []
    for item in geometry:
        if item.status == "missing_metadata":
            warnings.append(f"Body '{item.body_name}' has no geometry file metadata.")
        elif item.status == "unsupported":
            warnings.append(
                f"Body '{item.body_name}' geometry '{item.source}' is not a supported mesh "
                "format for Stage B0."
            )
        elif item.status == "missing_file":
            warnings.append(f"Body '{item.body_name}' geometry file was not found: {item.path}")

    return warnings


def frame_axis_segments(frame: Frame, axis_length: float = 1.0) -> list[AxisSegment]:
    """Calculate x/y/z axis line segments for a stored frame."""
    if axis_length <= 0.0:
        raise ValueError("axis_length must be positive.")

    origin = np.asarray(frame.origin, dtype=float)
    rotation_matrix = np.asarray(frame.rotation_matrix, dtype=float)

    segments = []
    for index, axis_name in enumerate(("x", "y", "z")):
        axis_direction = rotation_matrix[:, index]
        segments.append(
            AxisSegment(
                frame_name=frame.full_name,
                axis_name=axis_name,
                start=origin.copy(),
                end=origin + axis_direction * axis_length,
                color=AXIS_COLORS[axis_name],
            )
        )

    return segments


def project_axis_segments(project: Project, axis_length: float = 1.0) -> list[AxisSegment]:
    """Calculate frame axis line segments for every frame in a project."""
    segments = []
    for frame in project.frames:
        segments.extend(frame_axis_segments(frame, axis_length=axis_length))
    return segments


def infer_axis_length(project: Project, geometry_bounds: list[tuple[float, ...]]) -> float:
    """Choose a readable default axis length from geometry bounds or frame positions."""
    extents = []
    for bounds in geometry_bounds:
        extents.extend(
            (
                abs(bounds[1] - bounds[0]),
                abs(bounds[3] - bounds[2]),
                abs(bounds[5] - bounds[4]),
            )
        )

    if project.frames:
        frame_points = np.array([frame.origin for frame in project.frames], dtype=float)
        if len(frame_points) > 1:
            extents.extend(np.ptp(frame_points, axis=0).tolist())

    reference_extent = max(extents, default=100.0)
    if reference_extent <= 0.0:
        reference_extent = 100.0

    return reference_extent * 0.1


def render_project(
    project: Project,
    project_path: str | Path,
    *,
    axis_length: float | None = None,
) -> None:
    """Render supported geometry files and stored frames with PyVista."""
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is required for the viewer. Install it with "
            "'pip install chrono-frame-builder[visualization]'."
        ) from error

    geometry = collect_geometry(project, project_path)
    plotter = pv.Plotter()
    geometry_bounds = []

    for item in geometry:
        if not item.is_displayable:
            continue

        mesh = pv.read(str(item.path))
        plotter.add_mesh(mesh, color=item.color, opacity=item.opacity, name=item.body_name)
        geometry_bounds.append(tuple(mesh.bounds))

    resolved_axis_length = axis_length or infer_axis_length(project, geometry_bounds)
    for segment in project_axis_segments(project, axis_length=resolved_axis_length):
        line = pv.Line(segment.start, segment.end)
        plotter.add_mesh(line, color=segment.color, line_width=4)

    if project.frames:
        origins = np.array([frame.origin for frame in project.frames], dtype=float)
        labels = [frame.full_name for frame in project.frames]
        plotter.add_point_labels(origins, labels, font_size=12, point_size=0)

    plotter.add_axes()
    plotter.show(title=f"chrono-frame-builder: {project.project_name}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Stage B0 viewer."""
    parser = argparse.ArgumentParser(
        prog="chrono-frame-viewer",
        description="Open a minimal PyVista view of mesh geometry and stored project frames.",
    )
    parser.add_argument(
        "project",
        help="Path to project.json, or a directory containing project.json.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=None,
        help="Override the frame axis length in project units.",
    )

    args = parser.parse_args(argv)

    try:
        project, project_path = load_project_for_viewer(args.project)
        geometry = collect_geometry(project, project_path)
        for warning in geometry_warnings(geometry):
            print(f"warning: {warning}", file=sys.stderr)

        render_project(project, project_path, axis_length=args.axis_length)
    except Exception as error:
        print(f"chrono-frame-viewer: error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
