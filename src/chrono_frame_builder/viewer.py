"""Minimal PyVista viewer for geometry and stored project frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chrono_frame_builder.core.features import (
    FeatureCandidate,
    MeshEdgeSnapCandidate,
    mesh_vertex_candidate,
    nearest_triangle_edge_snap_candidate,
    point_candidate,
    point_to_segment_distance,
    triangle_face_candidate,
)
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.project import Project
from chrono_frame_builder.core.transform import (
    frame_from_origin_primary_secondary,
    frame_from_three_points,
)

SUPPORTED_GEOMETRY_EXTENSIONS = frozenset({".stl", ".obj", ".ply", ".vtk", ".vtp"})
AXIS_COLORS = {"x": "red", "y": "green", "z": "blue"}
AXIS_SELECTOR_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


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


@dataclass
class ThreePointPickState:
    """Collect three picked points and convert them into a preview frame."""

    points: list[np.ndarray]
    preview_frame: Frame | None = None
    warning: str | None = None

    @property
    def next_point_label(self) -> str:
        """Return the label for the next point expected from the user."""
        return f"P{len(self.points)}"

    def add_point(self, point: np.ndarray) -> Frame | None:
        """Add one picked point and return a preview frame when P0/P1/P2 are complete."""
        self.warning = None
        self.points.append(np.asarray(point, dtype=float))

        if len(self.points) < 3:
            return None

        try:
            self.preview_frame = frame_from_three_points(
                "__preview__",
                "three_point",
                self.points[0],
                self.points[1],
                self.points[2],
            )
        except ValueError as error:
            self.warning = str(error)
            self.preview_frame = None
            self.points.clear()
            return None

        self.points.clear()
        return self.preview_frame


@dataclass
class FeatureInspectionState:
    """Track printed feature candidates to avoid repeated terminal output."""

    last_signature: tuple | None = None

    def should_print(self, candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> bool:
        """Return whether a candidate differs from the last printed candidate."""
        signature = feature_candidate_signature(candidate)
        if signature == self.last_signature:
            return False

        self.last_signature = signature
        return True


@dataclass(frozen=True)
class FeaturePreviewStyle:
    """Scale-aware sizes for feature inspection preview actors."""

    marker_radius: float
    vector_length: float
    edge_line_width: int
    label_font_size: int


@dataclass
class CreateFrameWorkflowState:
    """Preview-only origin/primary/secondary frame creation workflow state."""

    selected_feature: FeatureCandidate | MeshEdgeSnapCandidate | None = None
    origin_feature: FeatureCandidate | MeshEdgeSnapCandidate | None = None
    primary_feature: FeatureCandidate | MeshEdgeSnapCandidate | None = None
    secondary_feature: FeatureCandidate | MeshEdgeSnapCandidate | None = None
    primary_axis: str = "+Z"
    secondary_axis: str = "+X"
    marker_name: str = "marker_preview"
    origin_source: str = "global"
    orientation_source: str = "global"
    manual_origin: np.ndarray | None = None
    manual_rotation_matrix: np.ndarray | None = None
    preview_frame: Frame | None = None
    warning: str | None = None
    last_selected_signature: tuple | None = None

    @property
    def step(self) -> str:
        """Return the current workflow step label."""
        if self.origin_feature is None:
            return "Step 1: Origin"
        if self.primary_feature is None:
            return "Step 2: Primary axis"
        if self.secondary_feature is None:
            return "Step 3: Secondary axis"
        return "Step 4: Preview"

    def select_feature(self, candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> bool:
        """Lock a preselected feature and initialize a global-origin preview when needed."""
        self.warning = None
        signature = feature_candidate_signature(candidate)
        self.selected_feature = candidate
        if signature == self.last_selected_signature:
            return False

        self.last_selected_signature = signature
        if self.origin_feature is None:
            self.origin_feature = candidate
            self.manual_origin = None
            self.origin_source = "feature"
            self.reset_orientation_to_global()
        return True

    def assign_selected_as_origin(self) -> None:
        """Assign the selected feature point as the frame origin source."""
        self.warning = None
        if self.selected_feature is None:
            raise ValueError("Select a mesh feature before assigning the origin.")

        self.origin_feature = self.selected_feature
        self.manual_origin = None
        self.origin_source = "feature"
        self.reset_orientation_to_global()

    def assign_selected_as_primary(self) -> None:
        """Assign the selected feature direction as the primary axis vector source."""
        self.warning = None
        if self.selected_feature is None:
            raise ValueError("Select a mesh feature before assigning the primary vector.")

        feature_direction_vector(self.selected_feature)
        self.primary_feature = self.selected_feature
        self.manual_rotation_matrix = None
        self.recompute_preview()

    def assign_selected_as_secondary(self) -> None:
        """Assign the selected feature direction as the secondary axis vector source."""
        self.warning = None
        if self.selected_feature is None:
            raise ValueError("Select a mesh feature before assigning the secondary vector.")

        feature_direction_vector(self.selected_feature)
        self.secondary_feature = self.selected_feature
        self.manual_rotation_matrix = None
        self.recompute_preview()

    def cycle_primary_axis(self) -> str:
        """Cycle through primary signed-axis selectors."""
        return self.set_primary_axis(_next_axis_selector(self.primary_axis))

    def cycle_secondary_axis(self) -> str:
        """Cycle through secondary signed-axis selectors."""
        return self.set_secondary_axis(_next_axis_selector(self.secondary_axis))

    def set_primary_axis(self, axis_selector: str) -> str:
        """Set the primary signed-axis selector for the next preview."""
        self.primary_axis = _validate_axis_selector(axis_selector, role="primary")
        self.recompute_preview()
        return self.primary_axis

    def set_secondary_axis(self, axis_selector: str) -> str:
        """Set the secondary signed-axis selector for the next preview."""
        self.secondary_axis = _validate_axis_selector(axis_selector, role="secondary")
        self.recompute_preview()
        return self.secondary_axis

    def reset_orientation_to_global(self) -> Frame | None:
        """Reset vector roles and preview a globally aligned frame at the assigned origin."""
        self.warning = None
        self.primary_feature = None
        self.secondary_feature = None
        self.manual_rotation_matrix = None
        self.orientation_source = "global"
        return self._global_preview()

    def _global_preview(self) -> Frame | None:
        """Build a global-orientation preview without changing assigned feature roles."""
        if self.origin_feature is None:
            self.preview_frame = None
            return None

        self.preview_frame = Frame(
            body="__preview__",
            name=self.marker_name,
            origin=self._active_origin(),
            rotation_matrix=np.eye(3),
        )
        self.orientation_source = "global"
        return self.preview_frame

    def recompute_preview(self) -> Frame | None:
        """Use feature vectors when complete, otherwise retain a global-origin preview."""
        if self.origin_feature is None:
            self.preview_frame = None
            return None
        if self.manual_rotation_matrix is not None:
            self.orientation_source = "manual"
            self.preview_frame = Frame(
                body="__preview__",
                name=self.marker_name,
                origin=self._active_origin(),
                rotation_matrix=self.manual_rotation_matrix.copy(),
            )
            return self.preview_frame
        if self.primary_feature is None or self.secondary_feature is None:
            return self._global_preview()

        return self.build_preview_frame()

    def build_preview_frame(
        self,
        *,
        parent_frame_name: str = "__preview__",
        frame_name: str | None = None,
    ) -> Frame:
        """Build a preview frame from assigned origin, primary, and secondary sources."""
        self.warning = None
        if self.origin_feature is None:
            raise ValueError("Assign an origin feature before previewing a frame.")
        if self.primary_feature is None:
            raise ValueError("Assign a primary vector feature before previewing a frame.")
        if self.secondary_feature is None:
            raise ValueError("Assign a secondary vector feature before previewing a frame.")
        if self.primary_axis[1] == self.secondary_axis[1]:
            raise ValueError("Primary and secondary axes must use different underlying axes.")

        self.preview_frame = frame_from_origin_primary_secondary(
            name=frame_name or self.marker_name,
            origin=self._active_origin(),
            primary_axis=self.primary_axis,
            primary_direction=feature_direction_vector(self.primary_feature),
            secondary_axis=self.secondary_axis,
            secondary_direction=feature_direction_vector(self.secondary_feature),
            parent_frame_name=parent_frame_name,
            frame_type="preview",
            notes="Preview-only frame; not saved.",
        )
        self.orientation_source = "feature"
        return self.preview_frame

    @property
    def preview_origin(self) -> np.ndarray | None:
        """Return a copy of the current preview origin, when a preview exists."""
        if self.preview_frame is None:
            return None
        return self.preview_frame.origin.copy()

    @property
    def preview_rotation_matrix(self) -> np.ndarray | None:
        """Return a copy of the current preview rotation matrix, when a preview exists."""
        if self.preview_frame is None:
            return None
        return self.preview_frame.rotation_matrix.copy()

    def set_marker_name(self, marker_name: str) -> None:
        """Rename the preview marker without persisting a project frame."""
        normalized_name = marker_name.strip()
        if not normalized_name:
            raise ValueError("Marker name must not be empty.")

        self.marker_name = normalized_name
        if self.preview_frame is not None:
            self.preview_frame = Frame(
                body=self.preview_frame.body,
                name=self.marker_name,
                origin=self.preview_frame.origin.copy(),
                rotation_matrix=self.preview_frame.rotation_matrix.copy(),
            )

    def apply_manual_position(self, origin: np.ndarray) -> Frame:
        """Override the preview position while retaining its current orientation."""
        if self.preview_frame is None:
            raise ValueError("Create a preview origin before applying a manual position.")

        self.manual_origin = _validate_preview_origin(origin)
        self.origin_source = "manual"
        self.preview_frame = Frame(
            body="__preview__",
            name=self.marker_name,
            origin=self.manual_origin.copy(),
            rotation_matrix=self.preview_frame.rotation_matrix.copy(),
        )
        return self.preview_frame

    def apply_manual_transform(self, origin: np.ndarray, rotation_matrix: np.ndarray) -> Frame:
        """Validate and apply a manual marker transform to the preview only."""
        if self.preview_frame is None:
            raise ValueError("Create a preview origin before applying a manual transform.")

        manual_origin = _validate_preview_origin(origin)
        manual_rotation_matrix = _validate_preview_rotation_matrix(rotation_matrix)
        self.manual_origin = manual_origin
        self.manual_rotation_matrix = manual_rotation_matrix
        self.origin_source = "manual"
        self.orientation_source = "manual"
        self.preview_frame = Frame(
            body="__preview__",
            name=self.marker_name,
            origin=self.manual_origin.copy(),
            rotation_matrix=self.manual_rotation_matrix.copy(),
        )
        return self.preview_frame

    def _active_origin(self) -> np.ndarray:
        """Return the manual origin override or the assigned feature origin."""
        if self.manual_origin is not None:
            return self.manual_origin.copy()
        if self.origin_feature is None:
            raise ValueError("Assign an origin feature before previewing a frame.")
        return feature_origin_point(self.origin_feature).copy()

    def reset(self) -> None:
        """Reset assigned roles and preview frame, preserving axis selector choices."""
        self.selected_feature = None
        self.origin_feature = None
        self.primary_feature = None
        self.secondary_feature = None
        self.marker_name = "marker_preview"
        self.origin_source = "global"
        self.orientation_source = "global"
        self.manual_origin = None
        self.manual_rotation_matrix = None
        self.preview_frame = None
        self.warning = None
        self.last_selected_signature = None


def _next_axis_selector(axis_selector: str) -> str:
    index = AXIS_SELECTOR_ORDER.index(_validate_axis_selector(axis_selector, role="axis"))
    return AXIS_SELECTOR_ORDER[(index + 1) % len(AXIS_SELECTOR_ORDER)]


def _validate_axis_selector(axis_selector: str, *, role: str) -> str:
    """Validate one signed frame-axis selector used by the preview workflow."""
    if axis_selector not in AXIS_SELECTOR_ORDER:
        expected = ", ".join(AXIS_SELECTOR_ORDER)
        raise ValueError(
            f"Invalid {role} axis selector '{axis_selector}'; expected one of: {expected}."
        )
    return axis_selector


def _validate_preview_origin(origin: np.ndarray) -> np.ndarray:
    """Validate one finite manual preview position."""
    validated_origin = np.asarray(origin, dtype=float)
    if validated_origin.shape != (3,):
        raise ValueError("Manual preview position must contain exactly three values.")
    if not np.all(np.isfinite(validated_origin)):
        raise ValueError("Manual preview position must contain only finite values.")
    return validated_origin.copy()


def _validate_preview_rotation_matrix(
    rotation_matrix: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate a finite right-handed orthonormal preview rotation matrix."""
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Manual rotation matrix must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Manual rotation matrix must contain only finite values.")

    identity = np.eye(3)
    if not np.allclose(matrix.T @ matrix, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Manual rotation matrix columns must be orthonormal.")
    if not np.allclose(matrix @ matrix.T, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Manual rotation matrix rows must be orthonormal.")

    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Manual rotation matrix determinant must be +1.")
    return matrix.copy()


def feature_origin_point(candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> np.ndarray:
    """Return the point to use when a feature is assigned as frame origin."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return candidate.midpoint

    return candidate.point


def feature_direction_vector(candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> np.ndarray:
    """Return the vector to use when a feature is assigned as an axis source."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return candidate.tangent
    if candidate.direction is None:
        raise ValueError(f"Feature '{candidate.kind}' does not provide a direction vector.")

    return candidate.direction


def feature_summary(candidate: FeatureCandidate | MeshEdgeSnapCandidate | None) -> str:
    """Return a concise terminal/control summary for a feature candidate."""
    if candidate is None:
        return "<none>"
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return f"{candidate.kind} midpoint={_format_vector(candidate.midpoint)}"

    return f"{candidate.kind} point={_format_vector(candidate.point)}"


def create_frame_status_summary(state: CreateFrameWorkflowState) -> str:
    """Return a concise summary of the create-frame workflow state."""
    return (
        f"{state.step}\n"
        f"origin: {feature_summary(state.origin_feature)}\n"
        f"primary {state.primary_axis}: {feature_summary(state.primary_feature)}\n"
        f"secondary {state.secondary_axis}: {feature_summary(state.secondary_feature)}"
    )


def create_frame_control_summary(state: CreateFrameWorkflowState) -> dict[str, str]:
    """Return panel-ready create-frame workflow text without a GUI dependency."""
    preview_origin = "<none>"
    preview_rotation = "<none>"
    if state.preview_frame is not None:
        preview_origin = _format_vector(state.preview_frame.origin)
        preview_rotation = np.array2string(
            state.preview_frame.rotation_matrix,
            precision=6,
            suppress_small=True,
        )

    return {
        "step": state.step,
        "marker_name": state.marker_name,
        "selected": feature_summary(state.selected_feature),
        "origin": feature_summary(state.origin_feature),
        "origin_source": state.origin_source,
        "primary": f"{state.primary_axis}: {feature_summary(state.primary_feature)}",
        "secondary": f"{state.secondary_axis}: {feature_summary(state.secondary_feature)}",
        "preview": "ready" if state.preview_frame is not None else "<none>",
        "orientation_source": state.orientation_source,
        "position": preview_origin,
        "rotation_matrix": preview_rotation,
        "message": state.warning or "Preview only: no project files are changed.",
    }


class CreateFrameControlPanel:
    """Small Tk control panel for the preview-only create-frame workflow.

    Tk is imported lazily by ``create_frame_control_panel`` so the core viewer module stays
    usable in environments where Tk is not installed.
    """

    def __init__(self, tk, ttk, state: CreateFrameWorkflowState, actions) -> None:
        self._tk = tk
        self._ttk = ttk
        self._state = state
        self._actions = actions
        self._closed = False
        self.window = tk.Tk()
        self.window.title("chrono-frame-builder: Create Frame")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        content = ttk.Frame(self.window, padding=12)
        content.grid(sticky="nsew")
        self._step = tk.StringVar()
        self._selected = tk.StringVar()
        self._origin = tk.StringVar()
        self._primary = tk.StringVar()
        self._secondary = tk.StringVar()
        self._preview = tk.StringVar()
        self._message = tk.StringVar()
        self._primary_axis = tk.StringVar(value=state.primary_axis)
        self._secondary_axis = tk.StringVar(value=state.secondary_axis)

        ttk.Label(content, textvariable=self._step, font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self._add_summary_row(content, 1, "Selected", self._selected)
        self._add_summary_row(content, 2, "Origin", self._origin)
        self._add_summary_row(content, 3, "Primary source", self._primary)
        self._add_summary_row(content, 4, "Secondary source", self._secondary)
        self._add_summary_row(content, 5, "Preview", self._preview)

        ttk.Separator(content).grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(content, text="Primary axis").grid(row=7, column=0, sticky="w", pady=2)
        primary_selector = ttk.Combobox(
            content,
            textvariable=self._primary_axis,
            values=AXIS_SELECTOR_ORDER,
            state="readonly",
            width=8,
        )
        primary_selector.grid(row=7, column=1, sticky="ew", pady=2)
        primary_selector.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._actions["set_primary_axis"](self._primary_axis.get()),
        )

        ttk.Label(content, text="Secondary axis").grid(row=8, column=0, sticky="w", pady=2)
        secondary_selector = ttk.Combobox(
            content,
            textvariable=self._secondary_axis,
            values=AXIS_SELECTOR_ORDER,
            state="readonly",
            width=8,
        )
        secondary_selector.grid(row=8, column=1, sticky="ew", pady=2)
        secondary_selector.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._actions["set_secondary_axis"](self._secondary_axis.get()),
        )

        buttons = ttk.Frame(content)
        buttons.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for row, (label, action_name) in enumerate(
            (
                ("Use selected as origin", "assign_origin"),
                ("Use selected as primary vector", "assign_primary"),
                ("Use selected as secondary vector", "assign_secondary"),
                ("Preview frame", "preview_frame"),
                ("Reset", "reset"),
            )
        ):
            ttk.Button(buttons, text=label, command=self._actions[action_name]).grid(
                row=row, column=0, sticky="ew", pady=2
            )

        ttk.Label(
            content,
            textvariable=self._message,
            foreground="#8b0000",
            justify="left",
            wraplength=360,
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        content.columnconfigure(1, weight=1)
        buttons.columnconfigure(0, weight=1)
        self.refresh()

    def _add_summary_row(self, parent, row: int, label: str, value) -> None:
        self._ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="nw", pady=2)
        self._ttk.Label(parent, textvariable=value, justify="left", wraplength=280).grid(
            row=row, column=1, sticky="w", pady=2
        )

    def refresh(self, message: str | None = None) -> None:
        """Synchronize displayed panel values from the workflow state."""
        if self._closed:
            return

        summary = create_frame_control_summary(self._state)
        self._step.set(summary["step"])
        self._selected.set(summary["selected"])
        self._origin.set(summary["origin"])
        self._primary.set(summary["primary"])
        self._secondary.set(summary["secondary"])
        self._preview.set(summary["preview"])
        self._message.set(message or summary["message"])
        self._primary_axis.set(self._state.primary_axis)
        self._secondary_axis.set(self._state.secondary_axis)

    def process_events(self) -> bool:
        """Process Tk events while the PyVista window owns the main event loop."""
        if self._closed:
            return False
        try:
            self.window.update_idletasks()
            self.window.update()
        except self._tk.TclError:
            self._closed = True
        return not self._closed

    def close(self) -> None:
        """Close the optional panel without affecting the viewer keyboard fallback."""
        if self._closed:
            return
        self._closed = True
        self.window.destroy()


def create_frame_control_panel(state: CreateFrameWorkflowState, actions):
    """Create a Tk control panel, or return ``None`` when Tk cannot be used."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return None

    try:
        return CreateFrameControlPanel(tk, ttk, state, actions)
    except tk.TclError:
        return None


def feature_candidate_signature(
    candidate: FeatureCandidate | MeshEdgeSnapCandidate,
    *,
    decimals: int = 9,
) -> tuple:
    """Create a stable comparison signature for duplicate print suppression."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return (
            candidate.kind,
            candidate.source_type,
            tuple(np.round(candidate.start, decimals)),
            tuple(np.round(candidate.midpoint, decimals)),
            tuple(np.round(candidate.end, decimals)),
            tuple(np.round(candidate.tangent, decimals)),
            candidate.source_ids,
        )

    direction = None
    if candidate.direction is not None:
        direction = tuple(np.round(candidate.direction, decimals))

    return (
        candidate.kind,
        candidate.source_type,
        tuple(np.round(candidate.point, decimals)),
        direction,
        candidate.source_ids,
    )


def geometry_bounds_diagonal(bounds: list[tuple[float, ...]]) -> float:
    """Return the diagonal length of the combined mesh bounds."""
    if not bounds:
        return 100.0

    bounds_array = np.asarray(bounds, dtype=float)
    mins = np.array(
        [
            np.min(bounds_array[:, 0]),
            np.min(bounds_array[:, 2]),
            np.min(bounds_array[:, 4]),
        ]
    )
    maxs = np.array(
        [
            np.max(bounds_array[:, 1]),
            np.max(bounds_array[:, 3]),
            np.max(bounds_array[:, 5]),
        ]
    )
    diagonal = float(np.linalg.norm(maxs - mins))
    if diagonal <= 0.0:
        return 100.0

    return diagonal


def feature_preview_style_from_bounds(bounds: list[tuple[float, ...]]) -> FeaturePreviewStyle:
    """Choose feature-preview visual sizes from the mesh bounding-box diagonal."""
    diagonal = geometry_bounds_diagonal(bounds)
    return FeaturePreviewStyle(
        marker_radius=diagonal * 0.025,
        vector_length=diagonal * 0.18,
        edge_line_width=8,
        label_font_size=14,
    )


def create_frame_preview_style_from_bounds(bounds: list[tuple[float, ...]]) -> FeaturePreviewStyle:
    """Choose less intrusive visual sizes for create-frame hover/selection previews."""
    diagonal = geometry_bounds_diagonal(bounds)
    return FeaturePreviewStyle(
        marker_radius=diagonal * 0.007,
        vector_length=diagonal * 0.1,
        edge_line_width=3,
        label_font_size=10,
    )


def feature_preview_labels(
    candidate: FeatureCandidate | MeshEdgeSnapCandidate,
    style: FeaturePreviewStyle,
) -> list[tuple[np.ndarray, str]]:
    """Return label locations and text for a feature-inspection candidate."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return [
            (candidate.start, "start"),
            (candidate.midpoint, "midpoint"),
            (candidate.end, "end"),
            (candidate.midpoint + candidate.tangent * style.vector_length, "tangent"),
        ]

    if candidate.kind == "triangle_face" and candidate.direction is not None:
        return [
            (candidate.point, "face center"),
            (candidate.point + candidate.direction * style.vector_length, "normal"),
        ]

    if candidate.kind == "mesh_vertex":
        return [(candidate.point, "vertex")]

    return [(candidate.point, "picked point")]


def create_frame_feature_labels(
    candidate: FeatureCandidate | MeshEdgeSnapCandidate,
    style: FeaturePreviewStyle,
) -> list[tuple[np.ndarray, str]]:
    """Return the intentionally compact labels used by create-frame mode."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return [
            (candidate.midpoint, "edge"),
            (candidate.midpoint + candidate.tangent * style.vector_length, "tangent"),
        ]
    if candidate.kind == "triangle_face" and candidate.direction is not None:
        return [
            (candidate.point, "face"),
            (candidate.point + candidate.direction * style.vector_length, "normal"),
        ]
    if candidate.kind == "mesh_vertex":
        return [(candidate.point, "vertex")]
    return [(candidate.point, "point")]


def triangle_feature_candidate_from_vertices(
    vertices: np.ndarray,
    *,
    body_name: str | None = None,
    mesh_file: str | None = None,
    cell_id: int | str | None = None,
) -> FeatureCandidate:
    """Convert a picked mesh triangle's vertices into a face feature candidate."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 3):
        raise ValueError("Feature inspection requires a triangle cell with exactly 3 vertices.")

    metadata = {}
    if body_name is not None:
        metadata["body_name"] = body_name
    if mesh_file is not None:
        metadata["mesh_file"] = mesh_file
    metadata["triangle_vertices"] = tuple(
        tuple(float(value) for value in vertex) for vertex in vertices
    )

    source_ids = () if cell_id is None else (cell_id,)
    return triangle_face_candidate(
        vertices[0],
        vertices[1],
        vertices[2],
        source_ids=source_ids,
        metadata=metadata,
    )


def point_feature_candidate_from_pick(
    point: np.ndarray,
    *,
    body_name: str | None = None,
    mesh_file: str | None = None,
    reason: str = "triangle cell data unavailable",
) -> FeatureCandidate:
    """Create a point fallback candidate from a picked surface position."""
    metadata = {"fallback_reason": reason}
    if body_name is not None:
        metadata["body_name"] = body_name
    if mesh_file is not None:
        metadata["mesh_file"] = mesh_file

    return point_candidate(point, metadata=metadata)


def feature_candidate_from_pick(
    point: np.ndarray,
    *,
    triangle_vertices: np.ndarray | None = None,
    body_name: str | None = None,
    mesh_file: str | None = None,
    cell_id: int | str | None = None,
) -> FeatureCandidate:
    """Create the best available feature candidate for a mesh pick."""
    if triangle_vertices is None:
        return point_feature_candidate_from_pick(
            point,
            body_name=body_name,
            mesh_file=mesh_file,
        )

    try:
        return triangle_feature_candidate_from_vertices(
            triangle_vertices,
            body_name=body_name,
            mesh_file=mesh_file,
            cell_id=cell_id,
        )
    except ValueError as error:
        return point_feature_candidate_from_pick(
            point,
            body_name=body_name,
            mesh_file=mesh_file,
            reason=str(error),
        )


def snap_feature_candidate_from_pick(
    point: np.ndarray,
    *,
    triangle_vertices: np.ndarray | None = None,
    body_name: str | None = None,
    mesh_file: str | None = None,
    cell_id: int | str | None = None,
    edge_snap_tolerance_ratio: float = 0.15,
) -> FeatureCandidate | MeshEdgeSnapCandidate:
    """Prefer edge snap handles near triangle edges, otherwise return face or point fallback."""
    if triangle_vertices is None:
        return point_feature_candidate_from_pick(
            point,
            body_name=body_name,
            mesh_file=mesh_file,
        )

    vertices = np.asarray(triangle_vertices, dtype=float)
    if vertices.shape != (3, 3):
        return point_feature_candidate_from_pick(
            point,
            body_name=body_name,
            mesh_file=mesh_file,
            reason="selected cell is not a triangle",
        )

    metadata = {}
    if body_name is not None:
        metadata["body_name"] = body_name
    if mesh_file is not None:
        metadata["mesh_file"] = mesh_file

    source_ids = () if cell_id is None else (cell_id,)
    edge_lengths = (
        np.linalg.norm(vertices[1] - vertices[0]),
        np.linalg.norm(vertices[2] - vertices[1]),
        np.linalg.norm(vertices[0] - vertices[2]),
    )
    nearest_vertex_index = int(np.argmin(np.linalg.norm(vertices - point, axis=1)))
    nearest_vertex_distance = np.linalg.norm(vertices[nearest_vertex_index] - point)
    vertex_snap_tolerance = min(edge_lengths) * edge_snap_tolerance_ratio
    if nearest_vertex_distance <= vertex_snap_tolerance:
        return mesh_vertex_candidate(
            vertices[nearest_vertex_index],
            source_ids=source_ids + (f"vertex:{nearest_vertex_index}",),
            metadata=metadata,
        )

    try:
        edge_candidate = nearest_triangle_edge_snap_candidate(
            point,
            vertices,
            source_ids=source_ids,
            metadata=metadata,
        )
    except ValueError as error:
        return point_feature_candidate_from_pick(
            point,
            body_name=body_name,
            mesh_file=mesh_file,
            reason=str(error),
        )

    snap_distance = point_to_segment_distance(point, edge_candidate.start, edge_candidate.end)
    snap_tolerance = min(edge_lengths) * edge_snap_tolerance_ratio
    if snap_distance <= snap_tolerance:
        return edge_candidate

    return triangle_feature_candidate_from_vertices(
        vertices,
        body_name=body_name,
        mesh_file=mesh_file,
        cell_id=cell_id,
    )


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
    pick_frame: bool = False,
    inspect_features: bool = False,
    create_frame: bool = False,
) -> None:
    """Render supported geometry files and stored frames with PyVista."""
    if create_frame:
        from chrono_frame_builder.viewer_qt import qt_ui_status, render_create_frame_qt

        qt_available, qt_message = qt_ui_status()
        if qt_available:
            render_create_frame_qt(project, project_path, axis_length=axis_length)
            return
        print(f"warning: {qt_message} Using the standalone viewer fallback.", file=sys.stderr)

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
    displayable_meshes = []

    for item in geometry:
        if not item.is_displayable:
            continue

        mesh = pv.read(str(item.path))
        mesh_options = {
            "color": item.color,
            "opacity": item.opacity,
            "name": f"geometry_{item.body_name}",
        }
        if create_frame:
            mesh_options.update(
                {"show_edges": True, "edge_color": "#202020", "line_width": 1}
            )
        plotter.add_mesh(mesh, **mesh_options)
        geometry_bounds.append(tuple(mesh.bounds))
        displayable_meshes.append((item, mesh))

    resolved_axis_length = axis_length or infer_axis_length(project, geometry_bounds)
    for segment in project_axis_segments(project, axis_length=resolved_axis_length):
        line = pv.Line(segment.start, segment.end)
        plotter.add_mesh(line, color=segment.color, line_width=4)

    if project.frames:
        origins = np.array([frame.origin for frame in project.frames], dtype=float)
        labels = [frame.full_name for frame in project.frames]
        plotter.add_point_labels(origins, labels, font_size=12, point_size=0)

    if pick_frame:
        enable_three_point_preview(plotter, pv, resolved_axis_length)

    if inspect_features:
        preview_style = feature_preview_style_from_bounds(geometry_bounds)
        enable_feature_inspection(plotter, pv, displayable_meshes, preview_style)

    if create_frame:
        preview_style = create_frame_preview_style_from_bounds(geometry_bounds)
        enable_create_frame_workflow(
            plotter,
            pv,
            displayable_meshes,
            preview_style,
            axis_length=resolved_axis_length,
        )

    plotter.add_axes()
    plotter.show(title=f"chrono-frame-builder: {project.project_name}")


def enable_three_point_preview(plotter, pv, axis_length: float) -> None:
    """Enable PyVista point picking for previewing a frame from P0/P1/P2."""
    pick_state = ThreePointPickState(points=[])
    preview_actor_names = [
        "preview_frame_x",
        "preview_frame_y",
        "preview_frame_z",
        "preview_origin",
    ]

    def clear_preview() -> None:
        for actor_name in preview_actor_names:
            if hasattr(plotter, "actors") and actor_name not in plotter.actors:
                continue
            plotter.remove_actor(actor_name, reset_camera=False)

    def draw_preview(frame: Frame) -> None:
        clear_preview()
        for segment in frame_axis_segments(frame, axis_length=axis_length):
            line = pv.Line(segment.start, segment.end)
            plotter.add_mesh(
                line,
                color=segment.color,
                line_width=6,
                name=f"preview_frame_{segment.axis_name}",
            )

        marker = pv.Sphere(radius=axis_length * 0.06, center=frame.origin)
        plotter.add_mesh(marker, color="yellow", name="preview_origin")
        plotter.render()

    def print_preview(frame: Frame) -> None:
        print("Preview frame from picked points:")
        print(f"origin: {np.array2string(frame.origin, precision=6, suppress_small=True)}")
        print("rotation_matrix:")
        print(np.array2string(frame.rotation_matrix, precision=6, suppress_small=True))
        print("Preview only: project.json was not modified.")

    def on_pick(point) -> None:
        if point is None:
            return

        picked_point = np.asarray(point, dtype=float)
        point_label = pick_state.next_point_label
        frame = pick_state.add_point(picked_point)
        print(f"{point_label}: {np.array2string(picked_point, precision=6, suppress_small=True)}")

        if pick_state.warning:
            print(f"warning: {pick_state.warning}", file=sys.stderr)
            print("Pick P0, P1, and P2 again.", file=sys.stderr)
            return

        if frame is not None:
            draw_preview(frame)
            print_preview(frame)

    picking_options = {
        "callback": on_pick,
        "show_message": "Pick P0 origin, P1 +X direction, then P2 in the XY plane.",
        "left_clicking": True,
        "show_point": True,
    }

    try:
        plotter.enable_surface_point_picking(**picking_options)
    except TypeError:
        picking_options.pop("show_point")
        plotter.enable_surface_point_picking(**picking_options)


def _mesh_cell_vertices(mesh, cell_id: int) -> np.ndarray | None:
    """Best-effort extraction of vertices for a PyVista mesh cell."""
    if cell_id is None or cell_id < 0:
        return None

    try:
        cell = mesh.get_cell(cell_id)
        return np.asarray(cell.points, dtype=float)
    except Exception:
        pass

    try:
        return np.asarray(mesh.extract_cells(cell_id).points, dtype=float)
    except Exception:
        return None


def _closest_mesh_triangle(point: np.ndarray, displayable_meshes) -> tuple | None:
    """Return the nearest recoverable cell vertices for a picked point, if available."""
    best_match = None
    best_distance = np.inf

    for geometry, mesh in displayable_meshes:
        try:
            cell_id = int(mesh.find_closest_cell(point))
        except Exception:
            continue

        vertices = _mesh_cell_vertices(mesh, cell_id)
        if vertices is None or len(vertices) == 0:
            continue

        distance = np.linalg.norm(np.mean(vertices, axis=0) - point)
        if distance < best_distance:
            best_match = (geometry, vertices, cell_id)
            best_distance = distance

    return best_match


def snap_candidate_from_displayable_mesh_pick(
    point: np.ndarray,
    displayable_meshes,
) -> tuple[FeatureCandidate | MeshEdgeSnapCandidate, str | None]:
    """Return a snap candidate and optional warning for a picked mesh location."""
    picked_point = np.asarray(point, dtype=float)
    match = _closest_mesh_triangle(picked_point, displayable_meshes)
    if match is None:
        return (
            point_feature_candidate_from_pick(
                picked_point,
                reason="no recoverable mesh triangle near pick",
            ),
            "warning: using point candidate fallback; no triangle cell was recovered.",
        )

    geometry, vertices, cell_id = match
    candidate = snap_feature_candidate_from_pick(
        picked_point,
        triangle_vertices=vertices,
        body_name=geometry.body_name,
        mesh_file=str(geometry.path) if geometry.path is not None else None,
        cell_id=cell_id,
    )
    if candidate.kind == "point":
        return candidate, "warning: using point candidate fallback; edge or face was not recovered."

    return candidate, None


def _format_vector(vector: np.ndarray) -> str:
    return np.array2string(np.asarray(vector, dtype=float), precision=6, suppress_small=True)


def _add_vector_arrow(
    plotter,
    pv,
    start: np.ndarray,
    direction: np.ndarray,
    style,
    name: str,
    color: str = "dodgerblue",
    opacity: float = 1.0,
) -> None:
    """Add a clear vector arrow, with a line fallback for PyVista variants."""
    try:
        arrow = pv.Arrow(
            start=start,
            direction=direction,
            scale=style.vector_length,
            shaft_radius=style.marker_radius * 0.16,
            tip_radius=style.marker_radius * 0.36,
        )
    except Exception:
        arrow = pv.Line(start, start + direction * style.vector_length)

    plotter.add_mesh(arrow, color=color, opacity=opacity, name=name)


def _add_feature_labels(
    plotter,
    labels,
    style: FeaturePreviewStyle,
    *,
    name: str = "feature_candidate_labels",
) -> None:
    """Add labels for feature preview handles."""
    if not labels:
        return

    points = np.array([point for point, _label in labels], dtype=float)
    text = [label for _point, label in labels]
    label_options = {
        "name": name,
        "font_size": style.label_font_size,
        "point_size": 0,
        "always_visible": True,
    }
    try:
        plotter.add_point_labels(points, text, **label_options)
    except TypeError:
        label_options.pop("name")
        plotter.add_point_labels(points, text, **label_options)


def _remove_named_actors(plotter, actor_names: list[str]) -> None:
    """Remove named PyVista actors when present."""
    for actor_name in actor_names:
        if hasattr(plotter, "actors") and actor_name not in plotter.actors:
            continue
        plotter.remove_actor(actor_name, reset_camera=False)


def create_frame_actor_names(prefix: str) -> list[str]:
    """Return actor names for one create-frame preview layer."""
    return [
        f"{prefix}_point",
        f"{prefix}_vector",
        f"{prefix}_labels",
        f"{prefix}_edge",
        f"{prefix}_edge_start",
        f"{prefix}_edge_midpoint",
        f"{prefix}_edge_end",
    ]


def frame_preview_actor_names(prefix: str) -> list[str]:
    """Return actor names for preview frame axes."""
    return [
        f"{prefix}_x",
        f"{prefix}_y",
        f"{prefix}_z",
        f"{prefix}_origin",
        f"{prefix}_label",
    ]


def print_feature_candidate(candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> None:
    """Print a feature candidate in a compact terminal-friendly form."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        print(f"Feature candidate: {candidate.kind} from {candidate.source_type}")
        print(f"start: {_format_vector(candidate.start)}")
        print(f"midpoint: {_format_vector(candidate.midpoint)}")
        print(f"end: {_format_vector(candidate.end)}")
        print(f"tangent: {_format_vector(candidate.tangent)}")
        if candidate.source_ids:
            print(f"source_ids: {candidate.source_ids}")
        if candidate.metadata:
            print(f"metadata: {candidate.metadata}")
        print("Preview only: project.json was not modified.")
        return

    print(f"Feature candidate: {candidate.kind} from {candidate.source_type}")
    print(f"point: {_format_vector(candidate.point)}")
    if candidate.direction is not None:
        print(f"direction: {_format_vector(candidate.direction)}")
    if candidate.source_ids:
        print(f"source_ids: {candidate.source_ids}")
    if candidate.metadata:
        print(f"metadata: {candidate.metadata}")
    print("Preview only: project.json was not modified.")


def enable_feature_inspection(
    plotter,
    pv,
    displayable_meshes,
    style: FeaturePreviewStyle,
) -> None:
    """Enable preview-only feature-candidate inspection from mesh surface picks."""
    inspection_state = FeatureInspectionState()
    actor_names = [
        "feature_candidate_point",
        "feature_candidate_direction",
        "feature_candidate_labels",
        "feature_edge_highlight",
        "feature_edge_start",
        "feature_edge_midpoint",
        "feature_edge_end",
    ]

    def clear_feature_preview() -> None:
        for actor_name in actor_names:
            if hasattr(plotter, "actors") and actor_name not in plotter.actors:
                continue
            plotter.remove_actor(actor_name, reset_camera=False)

    def draw_candidate(candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> None:
        clear_feature_preview()
        if isinstance(candidate, MeshEdgeSnapCandidate):
            edge = pv.Line(candidate.start, candidate.end)
            plotter.add_mesh(
                edge,
                color="orange",
                line_width=style.edge_line_width,
                name="feature_edge_highlight",
            )

            handle_specs = (
                ("feature_edge_start", candidate.start, "cyan"),
                ("feature_edge_midpoint", candidate.midpoint, "yellow"),
                ("feature_edge_end", candidate.end, "cyan"),
            )
            for actor_name, center, color in handle_specs:
                marker = pv.Sphere(radius=style.marker_radius, center=center)
                plotter.add_mesh(marker, color=color, name=actor_name)

            _add_vector_arrow(
                plotter,
                pv,
                candidate.midpoint,
                name="feature_candidate_direction",
                direction=candidate.tangent,
                style=style,
            )
            _add_feature_labels(plotter, feature_preview_labels(candidate, style), style)
            plotter.render()
            return

        marker = pv.Sphere(radius=style.marker_radius, center=candidate.point)
        plotter.add_mesh(marker, color="yellow", name="feature_candidate_point")

        if candidate.direction is not None:
            _add_vector_arrow(
                plotter,
                pv,
                candidate.point,
                candidate.direction,
                style,
                name="feature_candidate_direction",
            )

        _add_feature_labels(plotter, feature_preview_labels(candidate, style), style)
        plotter.render()

    def on_pick(point) -> None:
        if point is None:
            return

        picked_point = np.asarray(point, dtype=float)
        match = _closest_mesh_triangle(picked_point, displayable_meshes)
        if match is None:
            candidate = point_feature_candidate_from_pick(
                picked_point,
                reason="no recoverable mesh triangle near pick",
            )
            warning_message = (
                "warning: using point candidate fallback; no triangle cell was recovered."
            )
        else:
            warning_message = None
            geometry, vertices, cell_id = match
            candidate = snap_feature_candidate_from_pick(
                picked_point,
                triangle_vertices=vertices,
                body_name=geometry.body_name,
                mesh_file=str(geometry.path) if geometry.path is not None else None,
                cell_id=cell_id,
            )
            if candidate.kind == "point":
                warning_message = (
                    "warning: using point candidate fallback; edge or face was not recovered."
                )

        if inspection_state.should_print(candidate):
            if warning_message is not None:
                print(warning_message)
            print_feature_candidate(candidate)
        draw_candidate(candidate)

    picking_options = {
        "callback": on_pick,
        "show_message": (
            "Feature inspection preview: click mesh vertex, edge, or face handles. "
            "No frames are created or saved."
        ),
        "left_clicking": True,
        "show_point": True,
    }

    try:
        plotter.enable_surface_point_picking(**picking_options)
    except TypeError:
        picking_options.pop("show_point")
        plotter.enable_surface_point_picking(**picking_options)


def _draw_create_feature_candidate(
    plotter,
    pv,
    candidate: FeatureCandidate | MeshEdgeSnapCandidate,
    style: FeaturePreviewStyle,
    *,
    prefix: str,
    selected: bool,
) -> None:
    """Draw one create-frame feature candidate layer."""
    opacity = 1.0 if selected else 0.35
    marker_scale = 1.0 if selected else 0.55
    line_width = style.edge_line_width if selected else 1

    _remove_named_actors(plotter, create_frame_actor_names(prefix))
    if isinstance(candidate, MeshEdgeSnapCandidate):
        edge = pv.Line(candidate.start, candidate.end)
        plotter.add_mesh(
            edge,
            color="orange" if selected else "white",
            opacity=opacity,
            line_width=line_width,
            name=f"{prefix}_edge",
        )

        handle_specs = [
            (f"{prefix}_edge_midpoint", candidate.midpoint, "yellow", marker_scale),
        ]
        if selected:
            handle_specs.extend(
                (
                    (f"{prefix}_edge_start", candidate.start, "cyan", 0.55),
                    (f"{prefix}_edge_end", candidate.end, "cyan", 0.55),
                )
            )
        for actor_name, center, color, scale in handle_specs:
            marker = pv.Sphere(radius=style.marker_radius * scale, center=center)
            plotter.add_mesh(marker, color=color, opacity=opacity, name=actor_name)

        if selected:
            _add_feature_labels(
                plotter,
                create_frame_feature_labels(candidate, style),
                style,
                name=f"{prefix}_labels",
            )
        plotter.render()
        return

    marker = pv.Sphere(radius=style.marker_radius * marker_scale, center=candidate.point)
    color = "yellow" if selected else "white"
    plotter.add_mesh(marker, color=color, opacity=opacity, name=f"{prefix}_point")

    if selected:
        _add_feature_labels(
            plotter,
            create_frame_feature_labels(candidate, style),
            style,
            name=f"{prefix}_labels",
        )
    plotter.render()


def _draw_create_frame_preview(
    plotter,
    pv,
    frame: Frame,
    axis_length: float,
    *,
    prefix: str = "create_frame_preview",
) -> None:
    """Draw preview frame axes."""
    _remove_named_actors(plotter, frame_preview_actor_names(prefix))
    for segment in frame_axis_segments(frame, axis_length=axis_length):
        line = pv.Line(segment.start, segment.end)
        plotter.add_mesh(
            line,
            color=segment.color,
            line_width=7,
            name=f"{prefix}_{segment.axis_name}",
        )

    marker = pv.Sphere(radius=axis_length * 0.045, center=frame.origin)
    plotter.add_mesh(marker, color="white", name=f"{prefix}_origin")
    plotter.render()


def _print_create_frame_keyboard_fallback(state: CreateFrameWorkflowState) -> None:
    """Print the keyboard controls used when a Tk control panel is unavailable."""
    print("Create-frame preview keyboard fallback:")
    print("  click: lock current feature")
    print("  o: use selected feature as origin")
    print("  p: use selected feature as primary vector")
    print("  s: use selected feature as secondary vector")
    print("  [: cycle primary axis selector")
    print("  ]: cycle secondary axis selector")
    print("  v: preview frame")
    print("  r: reset")
    print("Preview only: project.json will not be modified.")
    print(create_frame_status_summary(state))


def enable_create_frame_workflow(
    plotter,
    pv,
    displayable_meshes,
    style: FeaturePreviewStyle,
    *,
    axis_length: float,
) -> None:
    """Enable preview-only Simscape-like frame creation workflow."""
    workflow_state = CreateFrameWorkflowState()
    hover_state = {"signature": None, "candidate": None}
    control_panel = None

    def refresh_control(message: str | None = None) -> None:
        if control_panel is not None:
            control_panel.refresh(message)

    def report_error(error: ValueError) -> None:
        workflow_state.warning = str(error)
        print(f"warning: {error}", file=sys.stderr)
        refresh_control()

    def update_hover(point) -> None:
        if point is None:
            return

        candidate, _warning = snap_candidate_from_displayable_mesh_pick(point, displayable_meshes)
        signature = feature_candidate_signature(candidate)
        if signature == hover_state["signature"]:
            return

        hover_state["signature"] = signature
        hover_state["candidate"] = candidate
        _draw_create_feature_candidate(
            plotter,
            pv,
            candidate,
            style,
            prefix="create_hover",
            selected=False,
        )

    def on_hover(*_args) -> None:
        if not hasattr(plotter, "pick_mouse_position"):
            return

        try:
            point = plotter.pick_mouse_position()
        except Exception:
            return

        update_hover(point)

    def on_pick(point) -> None:
        if point is None:
            return

        candidate, warning = snap_candidate_from_displayable_mesh_pick(point, displayable_meshes)
        changed = workflow_state.select_feature(candidate)
        _draw_create_feature_candidate(
            plotter,
            pv,
            candidate,
            style,
            prefix="create_selected",
            selected=True,
        )
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(
                plotter,
                pv,
                workflow_state.preview_frame,
                axis_length,
            )
        if changed:
            if warning is not None:
                print(warning)
            print(f"Selected feature: {feature_summary(candidate)}")
            print(create_frame_status_summary(workflow_state))
        refresh_control()

    def assign_origin() -> None:
        try:
            workflow_state.assign_selected_as_origin()
            print(f"Origin assigned: {feature_summary(workflow_state.origin_feature)}")
            print(create_frame_status_summary(workflow_state))
        except ValueError as error:
            report_error(error)
            return
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(plotter, pv, workflow_state.preview_frame, axis_length)
        refresh_control()

    def assign_primary() -> None:
        try:
            workflow_state.assign_selected_as_primary()
            print(
                f"Primary vector assigned ({workflow_state.primary_axis}): "
                f"{feature_summary(workflow_state.primary_feature)}"
            )
            print(create_frame_status_summary(workflow_state))
        except ValueError as error:
            report_error(error)
            return
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(plotter, pv, workflow_state.preview_frame, axis_length)
        refresh_control()

    def assign_secondary() -> None:
        try:
            workflow_state.assign_selected_as_secondary()
            print(
                f"Secondary vector assigned ({workflow_state.secondary_axis}): "
                f"{feature_summary(workflow_state.secondary_feature)}"
            )
            print(create_frame_status_summary(workflow_state))
        except ValueError as error:
            report_error(error)
            return
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(plotter, pv, workflow_state.preview_frame, axis_length)
        refresh_control()

    def set_primary_axis(axis_selector: str) -> None:
        try:
            print(f"Primary axis selector: {workflow_state.set_primary_axis(axis_selector)}")
            print(create_frame_status_summary(workflow_state))
        except ValueError as error:
            report_error(error)
            return
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(plotter, pv, workflow_state.preview_frame, axis_length)
        refresh_control()

    def set_secondary_axis(axis_selector: str) -> None:
        try:
            print(f"Secondary axis selector: {workflow_state.set_secondary_axis(axis_selector)}")
            print(create_frame_status_summary(workflow_state))
        except ValueError as error:
            report_error(error)
            return
        if workflow_state.preview_frame is not None:
            _draw_create_frame_preview(plotter, pv, workflow_state.preview_frame, axis_length)
        refresh_control()

    def cycle_primary() -> None:
        set_primary_axis(_next_axis_selector(workflow_state.primary_axis))

    def cycle_secondary() -> None:
        set_secondary_axis(_next_axis_selector(workflow_state.secondary_axis))

    def preview_frame() -> None:
        try:
            frame = workflow_state.recompute_preview()
            if frame is None:
                raise ValueError("Assign an origin feature before previewing a frame.")
            _draw_create_frame_preview(plotter, pv, frame, axis_length)
            print("Preview frame:")
            print(f"origin: {_format_vector(frame.origin)}")
            print("rotation_matrix:")
            print(np.array2string(frame.rotation_matrix, precision=6, suppress_small=True))
            print("Preview only: project.json was not modified.")
        except ValueError as error:
            report_error(error)
            return
        refresh_control("Preview frame drawn. Preview only: no project files are changed.")

    def reset() -> None:
        workflow_state.reset()
        hover_state["signature"] = None
        hover_state["candidate"] = None
        _remove_named_actors(plotter, create_frame_actor_names("create_hover"))
        _remove_named_actors(plotter, create_frame_actor_names("create_selected"))
        _remove_named_actors(plotter, frame_preview_actor_names("create_frame_preview"))
        plotter.render()
        print("Create-frame workflow reset.")
        print(create_frame_status_summary(workflow_state))
        refresh_control()

    control_panel = create_frame_control_panel(
        workflow_state,
        {
            "assign_origin": assign_origin,
            "assign_primary": assign_primary,
            "assign_secondary": assign_secondary,
            "set_primary_axis": set_primary_axis,
            "set_secondary_axis": set_secondary_axis,
            "preview_frame": preview_frame,
            "reset": reset,
        },
    )
    if control_panel is not None:
        try:
            plotter.add_timer_event(
                max_steps=1_000_000,
                duration=100,
                callback=lambda *_args: control_panel.process_events(),
            )
            print(
                "Create-frame control panel opened. Keyboard shortcuts remain available "
                "as a fallback."
            )
            print(create_frame_status_summary(workflow_state))
        except Exception:
            control_panel.close()
            control_panel = None
            print(
                "warning: Tk control panel could not be integrated with this PyVista version; "
                "using keyboard fallback.",
                file=sys.stderr,
            )

    if control_panel is None:
        _print_create_frame_keyboard_fallback(workflow_state)

    if hasattr(plotter, "track_mouse_position"):
        try:
            plotter.track_mouse_position(on_hover)
        except Exception:
            print("warning: hover preselection is unavailable in this PyVista version.")
    else:
        print("warning: hover preselection is unavailable in this PyVista version.")

    picking_options = {
        "callback": on_pick,
        "show_message": False,
        "left_clicking": True,
        "show_point": False,
    }
    try:
        plotter.enable_surface_point_picking(**picking_options)
    except TypeError:
        picking_options.pop("show_point")
        plotter.enable_surface_point_picking(**picking_options)

    key_actions = {
        "o": assign_origin,
        "p": assign_primary,
        "s": assign_secondary,
        "bracketleft": cycle_primary,
        "[": cycle_primary,
        "bracketright": cycle_secondary,
        "]": cycle_secondary,
        "v": preview_frame,
        "r": reset,
    }
    for key, action in key_actions.items():
        try:
            plotter.add_key_event(key, action)
        except Exception:
            continue


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the viewer CLI argument parser."""
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
    parser.add_argument(
        "--pick-frame",
        action="store_true",
        help="Preview a new frame by picking P0 origin, P1 +X, and P2 in the XY plane.",
    )
    parser.add_argument(
        "--inspect-features",
        action="store_true",
        help="Preview mesh feature candidates from surface picks without saving frames.",
    )
    parser.add_argument(
        "--create-frame",
        action="store_true",
        help="Preview Simscape-like origin/primary/secondary frame creation without saving.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Stage B0/B1 viewer."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        project, project_path = load_project_for_viewer(args.project)
        geometry = collect_geometry(project, project_path)
        for warning in geometry_warnings(geometry):
            print(f"warning: {warning}", file=sys.stderr)

        render_project(
            project,
            project_path,
            axis_length=args.axis_length,
            pick_frame=args.pick_frame,
            inspect_features=args.inspect_features,
            create_frame=args.create_frame,
        )
    except Exception as error:
        print(f"chrono-frame-viewer: error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
