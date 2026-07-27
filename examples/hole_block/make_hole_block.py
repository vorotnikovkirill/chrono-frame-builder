"""Generate a deterministic faceted block with a Z-axis through-hole OBJ mesh."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

BLOCK_X_MIN = -40.0
BLOCK_X_MAX = 40.0
BLOCK_Y_MIN = -20.0
BLOCK_Y_MAX = 20.0
HALF_THICKNESS = 8.0
HOLE_CENTER_X = 15.0
HOLE_CENTER_Y = 0.0
HOLE_RADIUS = 8.0
SEGMENTS = 48


def _outer_boundary_point(angle: float) -> tuple[float, float]:
    """Return the rectangle-boundary point reached from the hole center by one ray."""
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    candidates = []
    if direction_x > 0.0:
        candidates.append((BLOCK_X_MAX - HOLE_CENTER_X) / direction_x)
    elif direction_x < 0.0:
        candidates.append((BLOCK_X_MIN - HOLE_CENTER_X) / direction_x)
    if direction_y > 0.0:
        candidates.append((BLOCK_Y_MAX - HOLE_CENTER_Y) / direction_y)
    elif direction_y < 0.0:
        candidates.append((BLOCK_Y_MIN - HOLE_CENTER_Y) / direction_y)
    distance = min(value for value in candidates if value > 0.0)
    return HOLE_CENTER_X + distance * direction_x, HOLE_CENTER_Y + distance * direction_y


def _mesh_data() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    """Build a radial top/bottom annulus plus outer and faceted-hole side walls."""
    vertices = []
    faces = []
    for index in range(SEGMENTS):
        angle = 2.0 * math.pi * index / SEGMENTS
        inner_x = HOLE_CENTER_X + HOLE_RADIUS * math.cos(angle)
        inner_y = HOLE_CENTER_Y + HOLE_RADIUS * math.sin(angle)
        outer_x, outer_y = _outer_boundary_point(angle)
        vertices.extend(
            [
                (inner_x, inner_y, HALF_THICKNESS),
                (outer_x, outer_y, HALF_THICKNESS),
                (inner_x, inner_y, -HALF_THICKNESS),
                (outer_x, outer_y, -HALF_THICKNESS),
            ]
        )

    for index in range(SEGMENTS):
        next_index = (index + 1) % SEGMENTS
        inner_top = 4 * index + 1
        outer_top = inner_top + 1
        inner_bottom = inner_top + 2
        outer_bottom = inner_top + 3
        next_inner_top = 4 * next_index + 1
        next_outer_top = next_inner_top + 1
        next_inner_bottom = next_inner_top + 2
        next_outer_bottom = next_inner_top + 3
        faces.extend(
            [
                (inner_top, outer_top, next_outer_top, next_inner_top),
                (inner_bottom, next_inner_bottom, next_outer_bottom, outer_bottom),
                (inner_top, next_inner_top, next_inner_bottom, inner_bottom),
                (outer_top, outer_bottom, next_outer_bottom, next_outer_top),
            ]
        )
    return vertices, faces


def generate_hole_block(output_path: str | Path | None = None) -> Path:
    """Write the small, dependency-free hole-block OBJ and return its resolved path."""
    output = Path(output_path) if output_path is not None else Path(__file__).with_name("hole_block.obj")
    vertices, faces = _mesh_data()
    lines = [
        "# Faceted 80 x 40 x 16 block with a radius-8 Z-axis through-hole.",
        "# Hole center: (15, 0, 0); 48 segments intentionally expose mesh faceting.",
        "o hole_block",
    ]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(vertex) for vertex in face) for face in faces)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output.resolve()


def main() -> None:
    """Generate the default example mesh or an explicitly requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output OBJ path.")
    arguments = parser.parse_args()
    print(generate_hole_block(arguments.output))


if __name__ == "__main__":
    main()
