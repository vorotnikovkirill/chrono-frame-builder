"""Current coordinate-convention descriptions for viewer and marker-editor UI."""

from __future__ import annotations

from chrono_frame_builder.core.project import Project


def coordinate_context_summary(
    project: Project,
    parent_body_name: str | None = None,
) -> dict[str, str]:
    """Return honest coordinate context for the current body-attached frame schema."""
    body_names = project.list_body_names()
    if parent_body_name in body_names:
        parent_body = parent_body_name
    elif len(body_names) == 1:
        parent_body = body_names[0]
    else:
        parent_body = "<select a body>"

    return {
        "parent_body": parent_body,
        "display_coordinates": "Body/part-local Cartesian XYZ",
        "saved_coordinates": "Body/part-local Cartesian XYZ in project.json",
        "assembly_transform": "Not supported; current projects assume identity body transforms.",
    }
