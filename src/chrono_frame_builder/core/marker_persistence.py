"""Explicit persistence helpers for preview markers created by the Qt editor."""

from __future__ import annotations

import os
from pathlib import Path

from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.marker_editor import (
    CreateFrameEditorState,
    PreviewMarker,
    validate_preview_origin,
    validate_preview_rotation_matrix,
)
from chrono_frame_builder.core.project import Project


def preview_marker_to_frame(marker: PreviewMarker, body_name: str) -> Frame:
    """Convert one validated preview marker into the existing persisted frame model."""
    name = marker.name.strip()
    if not name:
        raise ValueError("Marker name must not be empty.")
    if not body_name:
        raise ValueError("A parent body is required to save a marker.")

    return Frame(
        body=body_name,
        name=name,
        origin=validate_preview_origin(marker.origin),
        rotation_matrix=validate_preview_rotation_matrix(marker.rotation_matrix),
    )


def save_selected_preview_marker(
    editor_state: CreateFrameEditorState,
    project: Project,
    project_path: str | Path,
) -> Frame:
    """Create or update the selected preview marker after full validation."""
    marker = editor_state.selected_marker
    if marker is None:
        raise ValueError("Select a preview marker before saving.")

    _validate_writable_project_path(project_path)
    linked_index = _linked_frame_index(marker, project)

    if linked_index is None:
        body_name = _resolve_marker_body_name(marker, project)
        frame = preview_marker_to_frame(marker, body_name)
        _validate_unique_frame_name(marker, frame, editor_state, project)
        project.frames.append(frame)
        try:
            project.save(project_path)
        except OSError as error:
            project.frames.pop()
            raise ValueError(f"Could not save project file: {error}") from error

        marker.parent_body_name = body_name
        marker.mark_saved(frame)
        return frame

    linked_frame = project.frames[linked_index]
    if marker.save_status == "Saved":
        return linked_frame

    frame = preview_marker_to_frame(marker, linked_frame.body)
    _validate_unique_frame_name(
        marker,
        frame,
        editor_state,
        project,
        excluded_frame_index=linked_index,
    )
    project.frames[linked_index] = frame
    try:
        project.save(project_path)
    except OSError as error:
        project.frames[linked_index] = linked_frame
        raise ValueError(f"Could not save project file: {error}") from error

    marker.parent_body_name = frame.body
    marker.mark_saved(frame)
    return frame


def _linked_frame_index(marker: PreviewMarker, project: Project) -> int | None:
    """Return the linked saved-frame index, or validate that a new marker is unlinked."""
    if marker.saved_frame_full_name is None:
        if marker.saved_signature is not None:
            raise ValueError("Saved marker has no linked project frame.")
        return None

    for index, frame in enumerate(project.frames):
        if frame.full_name == marker.saved_frame_full_name:
            return index
    raise ValueError(
        f"Saved marker link '{marker.saved_frame_full_name}' was not found in the project."
    )


def _resolve_marker_body_name(marker: PreviewMarker, project: Project) -> str:
    """Resolve a marker parent body from feature metadata or the one-body project convention."""
    if marker.parent_body_name and marker.parent_body_name in project.list_body_names():
        return marker.parent_body_name
    if len(project.bodies) == 1:
        return project.bodies[0].name
    if not project.bodies:
        raise ValueError("Project has no bodies to own the saved marker frame.")
    raise ValueError(
        "Marker parent body is ambiguous; select geometry from one project body first."
    )


def _validate_unique_frame_name(
    marker: PreviewMarker,
    frame: Frame,
    editor_state: CreateFrameEditorState,
    project: Project,
    *,
    excluded_frame_index: int | None = None,
) -> None:
    """Reject ambiguous persisted or preview marker names before writing the project."""
    if any(
        index != excluded_frame_index and existing.name == frame.name
        for index, existing in enumerate(project.frames)
    ):
        raise ValueError(f"A project frame named '{frame.name}' already exists.")
    if any(
        other.id != marker.id and other.name == frame.name
        for other in editor_state.markers
    ):
        raise ValueError(f"Another preview marker named '{frame.name}' already exists.")


def _validate_writable_project_path(path: str | Path) -> None:
    """Validate that a known project file can be explicitly rewritten."""
    project_path = Path(path)
    if not project_path.is_file():
        raise ValueError(f"Project file was not found: {project_path}")
    if not os.access(project_path, os.W_OK):
        raise ValueError(f"Project file is not writable: {project_path}")
