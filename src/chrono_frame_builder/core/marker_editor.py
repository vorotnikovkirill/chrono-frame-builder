"""Preview-only marker collection and Simscape-like frame-axis editing state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chrono_frame_builder.core.features import FeatureCandidate, MeshEdgeSnapCandidate
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.core.transform import (
    AXIS_INDEX,
    AXIS_SELECTORS,
    frame_from_origin_primary_secondary,
)

FeatureSource = FeatureCandidate | MeshEdgeSnapCandidate
AXIS_SOURCE_MODES = frozenset({"reference", "feature", "inertia"})
CLICK_MODES = frozenset(
    {
        "create_marker",
        "select_edit",
        "select_feature",
        "pick_primary_vector",
        "pick_secondary_vector",
        "quick_pick_primary_edge",
        "quick_pick_primary_face",
        "pick_line_point_a",
        "pick_line_point_b",
    }
)


def feature_origin_point(candidate: FeatureSource) -> np.ndarray:
    """Return the origin point represented by a mesh feature candidate."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return candidate.midpoint.copy()
    return candidate.point.copy()


def feature_direction_vector(candidate: FeatureSource) -> np.ndarray:
    """Return the orientation vector represented by a directional feature candidate."""
    if isinstance(candidate, MeshEdgeSnapCandidate):
        return candidate.tangent.copy()
    if candidate.direction is None:
        raise ValueError(f"Feature '{candidate.kind}' does not provide a direction vector.")
    return candidate.direction.copy()


def feature_source_metadata(candidate: FeatureSource) -> dict[str, Any]:
    """Return serializable provenance for a preview-only feature assignment."""
    return {
        "kind": candidate.kind,
        "source_type": candidate.source_type,
        "source_ids": tuple(candidate.source_ids),
        "metadata": dict(candidate.metadata),
    }


def _candidate_type(candidate: FeatureSource) -> str:
    """Classify the directional feature kinds used by quick orientation picks."""
    if candidate.kind == "line_between_points":
        return "line_between_points"
    if isinstance(candidate, MeshEdgeSnapCandidate) or candidate.kind == "mesh_edge":
        return "edge_tangent"
    return "face_normal"


def _is_roll_ambiguity(error: ValueError) -> bool:
    """Return whether frame construction needs a different secondary-axis definition."""
    message = str(error).lower()
    return "parallel" in message or "different frame axes" in message


def validate_preview_origin(origin: np.ndarray) -> np.ndarray:
    """Validate a finite marker position."""
    validated_origin = np.asarray(origin, dtype=float)
    if validated_origin.shape != (3,):
        raise ValueError("Marker position must contain exactly three values.")
    if not np.all(np.isfinite(validated_origin)):
        raise ValueError("Marker position must contain only finite values.")
    return validated_origin.copy()


def validate_preview_rotation_matrix(
    rotation_matrix: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate a finite right-handed orthonormal marker rotation matrix."""
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Marker rotation matrix must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Marker rotation matrix must contain only finite values.")

    identity = np.eye(3)
    if not np.allclose(matrix.T @ matrix, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Marker rotation matrix columns must be orthonormal.")
    if not np.allclose(matrix @ matrix.T, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Marker rotation matrix rows must be orthonormal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Marker rotation matrix determinant must be +1.")
    return matrix.copy()


def reference_axis_direction(axis_selector: str) -> np.ndarray:
    """Return a signed global direction for one frame-axis selector."""
    normalized_selector = axis_selector.upper()
    if normalized_selector not in AXIS_SELECTORS:
        valid = ", ".join(sorted(AXIS_SELECTORS))
        raise ValueError(f"Axis selector must be one of: {valid}.")

    direction = np.zeros(3, dtype=float)
    direction[AXIS_INDEX[normalized_selector[1]]] = 1.0 if normalized_selector[0] == "+" else -1.0
    return direction


@dataclass
class PreviewMarker:
    """One editable, non-persistent preview marker and its axis-source assignments."""

    id: int
    name: str
    origin: np.ndarray
    rotation_matrix: np.ndarray
    origin_source: str = "feature"
    orientation_source: str = "reference"
    parent_body_name: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    primary_axis: str = "+Z"
    secondary_axis: str = "+X"
    primary_reference_axis: str = "+Z"
    secondary_reference_axis: str = "+X"
    primary_axis_source_mode: str = "reference"
    secondary_axis_source_mode: str = "reference"
    primary_feature: FeatureSource | None = None
    secondary_feature: FeatureSource | None = None
    primary_direction_sign: int = 1
    secondary_direction_sign: int = 1
    saved_signature: tuple | None = None
    saved_frame_full_name: str | None = None

    def as_frame(self) -> Frame:
        """Return this preview marker in the existing frame representation."""
        return Frame(
            body="__preview__",
            name=self.name,
            origin=self.origin.copy(),
            rotation_matrix=self.rotation_matrix.copy(),
        )

    @property
    def save_status(self) -> str:
        """Return whether this preview marker has been explicitly persisted."""
        if self.saved_signature is None:
            return "Unsaved"
        return "Saved" if self.saved_signature == self._signature() else "Modified"

    def mark_saved(self, frame: Frame) -> None:
        """Record the transform snapshot that was successfully written to a project file."""
        self.saved_signature = self._signature()
        self.saved_frame_full_name = frame.full_name

    def _signature(self) -> tuple:
        return (
            self.name,
            self.parent_body_name,
            tuple(np.asarray(self.origin, dtype=float)),
            tuple(np.asarray(self.rotation_matrix, dtype=float).ravel()),
            self.origin_source,
            self.orientation_source,
            self.primary_axis,
            self.secondary_axis,
            self.primary_reference_axis,
            self.secondary_reference_axis,
            self.primary_axis_source_mode,
            self.secondary_axis_source_mode,
            _feature_signature(self.primary_feature),
            _feature_signature(self.secondary_feature),
            self.primary_direction_sign,
            self.secondary_direction_sign,
        )


@dataclass
class PendingAxisCandidate:
    """A directional mesh feature awaiting explicit application to a marker axis."""

    role: str
    feature: FeatureSource
    vector: np.ndarray
    anchor: np.ndarray
    marker_id: int
    flipped: bool = False

    @property
    def candidate_type(self) -> str:
        """Return a compact human-readable directional feature type."""
        if self.feature.kind == "line_between_points":
            return "line_between_points"
        if isinstance(self.feature, MeshEdgeSnapCandidate) or self.feature.kind == "mesh_edge":
            return "edge_tangent"
        return "face_normal"

    @property
    def role_label(self) -> str:
        """Return the panel label for the target frame-axis role."""
        return self.role.capitalize()

    @property
    def type_label(self) -> str:
        """Return the panel label for the feature-derived direction."""
        labels = {
            "edge_tangent": "Edge tangent",
            "face_normal": "Face normal",
            "line_between_points": "Line direction",
        }
        return labels[self.candidate_type]


def _feature_signature(candidate: FeatureSource | None) -> tuple | None:
    """Return a stable comparison value for one selected feature source."""
    if candidate is None:
        return None
    if isinstance(candidate, MeshEdgeSnapCandidate):
        geometry = (candidate.start, candidate.midpoint, candidate.end, candidate.tangent)
    else:
        geometry = (candidate.point, candidate.direction)
    return (
        candidate.kind,
        candidate.source_type,
        tuple(candidate.source_ids),
        tuple(
            None if value is None else tuple(np.asarray(value, dtype=float).ravel())
            for value in geometry
        ),
        _freeze_metadata(candidate.metadata),
    )


def _freeze_metadata(value: Any) -> tuple | float | int | str | bool | None:
    """Convert candidate metadata into a deterministic, numpy-safe comparison value."""
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_metadata(item)) for key, item in value.items()))
    if isinstance(value, np.ndarray):
        return tuple(_freeze_metadata(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class CreateFrameEditorState:
    """Qt-independent multi-marker editor state for the preview-only create-frame workflow."""

    markers: list[PreviewMarker] = field(default_factory=list)
    selected_marker_id: int | None = None
    selected_feature: FeatureSource | None = None
    pending_axis_candidate: PendingAxisCandidate | None = None
    line_start_point: np.ndarray | None = None
    click_mode: str = "select_edit"
    warning: str | None = None
    _next_marker_id: int = 1

    @property
    def selected_marker(self) -> PreviewMarker | None:
        """Return the selected marker, when one is selected."""
        if self.selected_marker_id is None:
            return None
        return next(
            (marker for marker in self.markers if marker.id == self.selected_marker_id),
            None,
        )

    def create_marker_from_feature(self, candidate: FeatureSource) -> PreviewMarker:
        """Create and select a global-orientation marker at a selected feature origin."""
        self.warning = None
        self.pending_axis_candidate = None
        self.line_start_point = None
        marker_id = self._next_marker_id
        self._next_marker_id += 1
        marker = PreviewMarker(
            id=marker_id,
            name=f"marker_{marker_id:03d}",
            origin=feature_origin_point(candidate),
            rotation_matrix=np.eye(3),
            origin_source="feature",
            orientation_source="reference",
            parent_body_name=candidate.metadata.get("body_name"),
            source_metadata={"origin": feature_source_metadata(candidate)},
        )
        self.markers.append(marker)
        self.selected_marker_id = marker.id
        self.selected_feature = candidate
        return marker

    def select_marker(self, marker_id: int) -> PreviewMarker:
        """Select an existing preview marker by its stable identifier."""
        marker = next((item for item in self.markers if item.id == marker_id), None)
        if marker is None:
            raise ValueError(f"Preview marker {marker_id} was not found.")
        self.selected_marker_id = marker.id
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.warning = None
        return marker

    def select_feature(self, candidate: FeatureSource) -> None:
        """Store a geometry feature for a later frame-origin or axis assignment."""
        self.selected_feature = candidate
        self.warning = None

    def set_click_mode(self, click_mode: str) -> None:
        """Set the viewer click behavior to marker creation or feature selection."""
        normalized_mode = "select_edit" if click_mode == "select_feature" else click_mode
        if normalized_mode not in CLICK_MODES:
            valid = ", ".join(sorted(CLICK_MODES))
            raise ValueError(f"Click mode must be one of: {valid}.")
        self.click_mode = normalized_mode
        if normalized_mode not in {
            "pick_primary_vector",
            "pick_secondary_vector",
            "quick_pick_primary_edge",
            "quick_pick_primary_face",
            "pick_line_point_a",
            "pick_line_point_b",
        }:
            self.pending_axis_candidate = None
            self.line_start_point = None

    def begin_pick_primary_vector(self) -> None:
        """Enter the one-click geometry-picking mode for the primary vector source."""
        self._require_selected_marker()
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.click_mode = "pick_primary_vector"
        self.warning = None

    def begin_new_marker(self) -> None:
        """Enter the one-click marker creation mode."""
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.click_mode = "create_marker"
        self.warning = None

    def begin_pick_secondary_vector(self) -> None:
        """Enter the one-click geometry-picking mode for the secondary vector source."""
        self._require_selected_marker()
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.click_mode = "pick_secondary_vector"
        self.warning = None

    def begin_quick_primary_edge_direction(self, axis_selector: str) -> None:
        """Stage a one-click edge-tangent source for the selected local primary axis."""
        self._begin_quick_primary_pick(axis_selector, "quick_pick_primary_edge")

    def begin_quick_primary_face_normal(self, axis_selector: str) -> None:
        """Stage a one-click face-normal source for the selected local primary axis."""
        self._begin_quick_primary_pick(axis_selector, "quick_pick_primary_face")

    def begin_quick_primary_line(self, axis_selector: str) -> None:
        """Start a two-point direction pick for the selected local primary axis."""
        self._begin_quick_primary_pick(axis_selector, "pick_line_point_a")

    def use_quick_primary_reference_axis(self, axis_selector: str) -> PreviewMarker:
        """Align the selected primary marker axis with the same signed global reference axis."""
        marker = self._require_selected_marker()
        previous = (
            marker.primary_axis,
            marker.primary_reference_axis,
            marker.primary_axis_source_mode,
        )
        selector = self._validate_axis_selector(axis_selector, role="quick primary")
        marker.primary_axis = selector
        marker.primary_reference_axis = selector
        marker.primary_axis_source_mode = "reference"
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            (
                marker.primary_axis,
                marker.primary_reference_axis,
                marker.primary_axis_source_mode,
            ) = previous
            raise

    def handle_geometry_pick(self, candidate: FeatureSource) -> PreviewMarker | None:
        """Create markers or stage directional candidates for explicit axis application."""
        if self.click_mode == "create_marker":
            marker = self.create_marker_from_feature(candidate)
            self.click_mode = "select_edit"
            return marker

        self.select_feature(candidate)
        if self.click_mode == "select_edit":
            return None
        if self.click_mode == "pick_primary_vector":
            self._create_pending_axis_candidate("primary", candidate)
            return None
        if self.click_mode == "pick_secondary_vector":
            self._create_pending_axis_candidate("secondary", candidate)
            return None
        if self.click_mode == "quick_pick_primary_edge":
            self._create_pending_axis_candidate(
                "primary",
                candidate,
                expected_type="edge_tangent",
            )
            return None
        if self.click_mode == "quick_pick_primary_face":
            self._create_pending_axis_candidate(
                "primary",
                candidate,
                expected_type="face_normal",
            )
            return None
        if self.click_mode == "pick_line_point_a":
            self.line_start_point = feature_origin_point(candidate)
            self.click_mode = "pick_line_point_b"
            return None
        if self.click_mode == "pick_line_point_b":
            self._create_line_pending_axis_candidate(candidate)
            return None
        raise ValueError(f"Unsupported click mode: {self.click_mode}.")

    def apply_pending_axis_candidate(self) -> PreviewMarker:
        """Apply the pending geometric direction to its selected marker-axis role."""
        pending = self._require_pending_axis_candidate()
        marker = self._require_selected_marker()
        if marker.id != pending.marker_id:
            raise ValueError("Pending axis candidate belongs to a different preview marker.")

        direction_sign = -1 if pending.flipped else 1
        self._assign_feature_axis(
            marker,
            pending.feature,
            pending.role,
            direction_sign,
            allow_primary_roll_warning=pending.role == "primary",
        )
        self.selected_feature = pending.feature
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.click_mode = "select_edit"
        return marker

    def flip_pending_axis_candidate(self) -> PendingAxisCandidate:
        """Reverse the displayed pending direction without changing the marker yet."""
        pending = self._require_pending_axis_candidate()
        pending.vector = -pending.vector
        pending.flipped = not pending.flipped
        self.warning = None
        return pending

    def cancel_pending_axis_candidate(self) -> None:
        """Discard the staged direction and return to marker selection/editing."""
        self.pending_axis_candidate = None
        self.line_start_point = None
        if self.click_mode in {
            "pick_primary_vector",
            "pick_secondary_vector",
            "quick_pick_primary_edge",
            "quick_pick_primary_face",
            "pick_line_point_a",
            "pick_line_point_b",
        }:
            self.click_mode = "select_edit"
        self.warning = None

    def rename_selected_marker(self, name: str) -> PreviewMarker:
        """Rename the selected preview marker."""
        marker = self._require_selected_marker()
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Marker name must not be empty.")
        marker.name = normalized_name
        return marker

    def apply_selected_position(self, origin: np.ndarray) -> PreviewMarker:
        """Apply a manual XYZ position override to the selected marker."""
        marker = self._require_selected_marker()
        marker.origin = validate_preview_origin(origin)
        marker.origin_source = "manual"
        marker.source_metadata["origin"] = {"kind": "manual_xyz"}
        return marker

    def apply_selected_rotation(self, rotation_matrix: np.ndarray) -> PreviewMarker:
        """Apply a validated manual rotation override to the selected marker."""
        marker = self._require_selected_marker()
        marker.rotation_matrix = validate_preview_rotation_matrix(rotation_matrix)
        marker.orientation_source = "manual"
        return marker

    def apply_selected_transform(
        self,
        origin: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> PreviewMarker:
        """Atomically apply a validated manual transform to the selected marker."""
        validated_origin = validate_preview_origin(origin)
        validated_rotation = validate_preview_rotation_matrix(rotation_matrix)
        marker = self._require_selected_marker()
        marker.origin = validated_origin
        marker.rotation_matrix = validated_rotation
        marker.origin_source = "manual"
        marker.orientation_source = "manual"
        marker.source_metadata["origin"] = {"kind": "manual_xyz"}
        return marker

    def set_primary_axis(self, axis_selector: str) -> PreviewMarker:
        """Set the selected marker's signed primary frame axis and recompute orientation."""
        marker = self._require_selected_marker()
        previous_axis = marker.primary_axis
        marker.primary_axis = self._validate_axis_selector(axis_selector, role="primary")
        if marker.primary_axis_source_mode == "feature" and marker.primary_feature is None:
            return marker
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.primary_axis = previous_axis
            raise

    def set_secondary_axis(self, axis_selector: str) -> PreviewMarker:
        """Set the selected marker's signed secondary frame axis and recompute orientation."""
        marker = self._require_selected_marker()
        previous_axis = marker.secondary_axis
        marker.secondary_axis = self._validate_axis_selector(axis_selector, role="secondary")
        if marker.secondary_axis_source_mode == "feature" and marker.secondary_feature is None:
            return marker
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.secondary_axis = previous_axis
            raise

    def set_primary_reference_axis(self, axis_selector: str) -> PreviewMarker:
        """Set the primary reference/source axis and recompute when applicable."""
        marker = self._require_selected_marker()
        previous_axis = marker.primary_reference_axis
        marker.primary_reference_axis = self._validate_axis_selector(
            axis_selector,
            role="primary source",
        )
        if marker.primary_axis_source_mode == "feature":
            return marker
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.primary_reference_axis = previous_axis
            raise

    def set_secondary_reference_axis(self, axis_selector: str) -> PreviewMarker:
        """Set the secondary reference/source axis and recompute when applicable."""
        marker = self._require_selected_marker()
        previous_axis = marker.secondary_reference_axis
        marker.secondary_reference_axis = self._validate_axis_selector(
            axis_selector,
            role="secondary source",
        )
        if marker.secondary_axis_source_mode == "feature":
            return marker
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.secondary_reference_axis = previous_axis
            raise

    def set_primary_axis_source_mode(self, mode: str) -> PreviewMarker:
        """Set the primary-axis source mode and recompute when it is available."""
        marker = self._require_selected_marker()
        marker.primary_axis_source_mode = self._validate_axis_source_mode(mode)
        if mode == "feature" and marker.primary_feature is None:
            return marker
        return self.recompute_selected_marker_orientation()

    def set_secondary_axis_source_mode(self, mode: str) -> PreviewMarker:
        """Set the secondary-axis source mode and recompute when it is available."""
        marker = self._require_selected_marker()
        marker.secondary_axis_source_mode = self._validate_axis_source_mode(mode)
        if mode == "feature" and marker.secondary_feature is None:
            return marker
        return self.recompute_selected_marker_orientation()

    def use_selected_feature_for_primary_axis(self) -> PreviewMarker:
        """Assign the selected directional feature as the selected marker's primary vector."""
        marker = self._require_selected_marker()
        candidate = self._require_selected_direction_feature()
        return self._assign_feature_axis(marker, candidate, "primary", 1)

    def use_selected_feature_for_secondary_axis(self) -> PreviewMarker:
        """Assign the selected directional feature as the selected marker's secondary vector."""
        marker = self._require_selected_marker()
        candidate = self._require_selected_direction_feature()
        return self._assign_feature_axis(marker, candidate, "secondary", 1)

    def flip_primary_direction(self) -> PreviewMarker:
        """Flip the selected marker's primary geometric vector direction."""
        marker = self._require_selected_marker()
        if marker.primary_axis_source_mode != "feature" or marker.primary_feature is None:
            raise ValueError("Assign a geometric feature for the primary axis before flipping it.")
        marker.primary_direction_sign *= -1
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.primary_direction_sign *= -1
            raise

    def flip_secondary_direction(self) -> PreviewMarker:
        """Flip the selected marker's secondary geometric vector direction."""
        marker = self._require_selected_marker()
        if marker.secondary_axis_source_mode != "feature" or marker.secondary_feature is None:
            raise ValueError(
                "Assign a geometric feature for the secondary axis before flipping it."
            )
        marker.secondary_direction_sign *= -1
        try:
            return self.recompute_selected_marker_orientation()
        except ValueError:
            marker.secondary_direction_sign *= -1
            raise

    def reset_selected_marker_axes_to_reference(self) -> PreviewMarker:
        """Restore reference-axis orientation for the selected marker."""
        marker = self._require_selected_marker()
        marker.primary_axis_source_mode = "reference"
        marker.secondary_axis_source_mode = "reference"
        marker.primary_reference_axis = marker.primary_axis
        marker.secondary_reference_axis = marker.secondary_axis
        marker.primary_feature = None
        marker.secondary_feature = None
        marker.primary_direction_sign = 1
        marker.secondary_direction_sign = 1
        return self.recompute_selected_marker_orientation()

    def recompute_selected_marker_orientation(self) -> PreviewMarker:
        """Compute selected-marker orientation from reference or feature axis sources."""
        marker = self._require_selected_marker()
        primary_direction = self._axis_direction(marker, primary=True)
        secondary_direction = self._axis_direction(marker, primary=False)
        frame = frame_from_origin_primary_secondary(
            name=marker.name,
            origin=marker.origin,
            primary_axis=marker.primary_axis,
            primary_direction=primary_direction,
            secondary_axis=marker.secondary_axis,
            secondary_direction=secondary_direction,
            parent_frame_name="__preview__",
            frame_type="preview",
            notes="Preview-only marker; not saved.",
        )
        marker.rotation_matrix = frame.rotation_matrix
        marker.orientation_source = (
            "reference"
            if (
                marker.primary_axis_source_mode == "reference"
                and marker.secondary_axis_source_mode == "reference"
            )
            else "feature"
        )
        return marker

    def _axis_direction(self, marker: PreviewMarker, *, primary: bool) -> np.ndarray:
        mode = marker.primary_axis_source_mode if primary else marker.secondary_axis_source_mode
        reference_axis = (
            marker.primary_reference_axis if primary else marker.secondary_reference_axis
        )
        feature = marker.primary_feature if primary else marker.secondary_feature
        direction_sign = (
            marker.primary_direction_sign if primary else marker.secondary_direction_sign
        )
        axis_name = "primary" if primary else "secondary"
        if mode == "reference":
            return reference_axis_direction(reference_axis)
        if mode == "inertia":
            raise ValueError("Principal inertia axis support is not available yet.")
        if feature is None:
            raise ValueError(f"Assign a geometric feature for the {axis_name} axis first.")
        return direction_sign * feature_direction_vector(feature)

    def _create_pending_axis_candidate(
        self,
        role: str,
        candidate: FeatureSource,
        *,
        expected_type: str | None = None,
    ) -> PendingAxisCandidate:
        """Stage a face normal or edge tangent for a later explicit assignment."""
        marker = self._require_selected_marker()
        if role not in {"primary", "secondary"}:
            raise ValueError("Pending axis candidate role must be primary or secondary.")
        try:
            vector = feature_direction_vector(candidate)
        except ValueError as error:
            raise ValueError(
                "Point/vertex features do not provide an axis direction; choose a face or edge."
            ) from error

        candidate_type = _candidate_type(candidate)
        if expected_type is not None and candidate_type != expected_type:
            required = "edge" if expected_type == "edge_tangent" else "face"
            raise ValueError(f"Click an {required} to create this quick orientation candidate.")

        pending = PendingAxisCandidate(
            role=role,
            feature=candidate,
            vector=vector,
            anchor=feature_origin_point(candidate),
            marker_id=marker.id,
        )
        self.selected_feature = candidate
        self.pending_axis_candidate = pending
        self.warning = None
        return pending

    def _create_line_pending_axis_candidate(self, candidate: FeatureSource) -> PendingAxisCandidate:
        """Create a primary candidate from two selected feature-origin points."""
        if self.line_start_point is None:
            raise ValueError(
                "Click the first point for the line direction before choosing the second."
            )
        point_a = self.line_start_point.copy()
        point_b = feature_origin_point(candidate)
        vector = point_b - point_a
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            raise ValueError("Line points must be distinct to define a direction.")
        line_candidate = FeatureCandidate(
            kind="line_between_points",
            source_type="mesh",
            point=point_a,
            direction=vector / length,
            source_ids=tuple(candidate.source_ids),
            metadata={
                "point_a": tuple(point_a),
                "point_b": tuple(point_b),
            },
        )
        return self._create_pending_axis_candidate("primary", line_candidate)

    def _assign_feature_axis(
        self,
        marker: PreviewMarker,
        candidate: FeatureSource,
        role: str,
        direction_sign: int,
        *,
        allow_primary_roll_warning: bool = False,
    ) -> PreviewMarker:
        """Assign one directional feature while restoring prior state on invalid geometry."""
        if role == "primary":
            previous = (
                marker.primary_axis_source_mode,
                marker.primary_feature,
                marker.primary_direction_sign,
            )
            marker.primary_axis_source_mode = "feature"
            marker.primary_feature = candidate
            marker.primary_direction_sign = direction_sign
        else:
            previous = (
                marker.secondary_axis_source_mode,
                marker.secondary_feature,
                marker.secondary_direction_sign,
            )
            marker.secondary_axis_source_mode = "feature"
            marker.secondary_feature = candidate
            marker.secondary_direction_sign = direction_sign
        try:
            marker = self.recompute_selected_marker_orientation()
        except ValueError as error:
            if allow_primary_roll_warning and role == "primary" and _is_roll_ambiguity(error):
                marker.orientation_source = "feature"
                marker.source_metadata["primary_axis"] = feature_source_metadata(candidate)
                self.warning = (
                    "Primary direction set. Choose a non-parallel Secondary Axis to fully define "
                    "orientation."
                )
                return marker
            if role == "primary":
                (
                    marker.primary_axis_source_mode,
                    marker.primary_feature,
                    marker.primary_direction_sign,
                ) = previous
            else:
                (
                    marker.secondary_axis_source_mode,
                    marker.secondary_feature,
                    marker.secondary_direction_sign,
                ) = previous
            raise
        marker.source_metadata[f"{role}_axis"] = feature_source_metadata(candidate)
        return marker

    def _begin_quick_primary_pick(self, axis_selector: str, click_mode: str) -> None:
        """Set the intended local primary axis and enter a constrained quick-pick mode."""
        marker = self._require_selected_marker()
        marker.primary_axis = self._validate_axis_selector(axis_selector, role="quick primary")
        self.pending_axis_candidate = None
        self.line_start_point = None
        self.click_mode = click_mode
        self.warning = None

    def _require_selected_marker(self) -> PreviewMarker:
        marker = self.selected_marker
        if marker is None:
            raise ValueError("Select or create a preview marker first.")
        return marker

    def _require_pending_axis_candidate(self) -> PendingAxisCandidate:
        if self.pending_axis_candidate is None:
            raise ValueError("Pick a face normal or edge tangent before applying it.")
        return self.pending_axis_candidate

    def _require_selected_direction_feature(self) -> FeatureSource:
        if self.selected_feature is None:
            raise ValueError("Select a face normal or edge tangent before assigning an axis.")
        feature_direction_vector(self.selected_feature)
        return self.selected_feature

    @staticmethod
    def _validate_axis_selector(axis_selector: str, *, role: str) -> str:
        normalized_selector = axis_selector.upper()
        if normalized_selector not in AXIS_SELECTORS:
            valid = ", ".join(sorted(AXIS_SELECTORS))
            raise ValueError(f"Invalid {role} axis selector; expected one of: {valid}.")
        return normalized_selector

    @staticmethod
    def _validate_axis_source_mode(mode: str) -> str:
        if mode not in AXIS_SOURCE_MODES:
            valid = ", ".join(sorted(AXIS_SOURCE_MODES))
            raise ValueError(f"Axis source mode must be one of: {valid}.")
        if mode == "inertia":
            raise ValueError("Principal inertia axis support is not available yet.")
        return mode


def create_frame_editor_summary(state: CreateFrameEditorState) -> dict[str, str]:
    """Return panel-ready text for the currently selected preview marker."""
    marker = state.selected_marker
    pending = state.pending_axis_candidate
    pending_summary = {
        "pending_role": pending.role_label if pending is not None else "<none>",
        "pending_type": pending.type_label if pending is not None else "<none>",
        "pending_vector": (
            np.array2string(pending.vector, precision=6, suppress_small=True)
            if pending is not None
            else "<none>"
        ),
    }
    if marker is None:
        return {
            "selected_marker": "<none>",
            "origin": "<none>",
            "origin_source": "<none>",
            "primary_feature": "<none>",
            "secondary_feature": "<none>",
            "message": create_frame_editor_status_message(state),
            **pending_summary,
        }

    primary_kind = marker.primary_feature.kind if marker.primary_feature is not None else "<none>"
    secondary_kind = (
        marker.secondary_feature.kind if marker.secondary_feature is not None else "<none>"
    )
    return {
        "selected_marker": marker.name,
        "origin": np.array2string(marker.origin, precision=6, suppress_small=True),
        "origin_source": marker.origin_source,
        "save_status": marker.save_status,
        "primary_reference_axis": marker.primary_reference_axis,
        "secondary_reference_axis": marker.secondary_reference_axis,
        "primary_feature": primary_kind,
        "secondary_feature": secondary_kind,
        "message": create_frame_editor_status_message(state),
        **pending_summary,
    }


def create_frame_editor_status_message(state: CreateFrameEditorState) -> str:
    """Return workflow-specific guidance without implying accidental marker creation."""
    if state.warning:
        return state.warning
    if state.pending_axis_candidate is not None:
        return "Review candidate, then Apply / Flip / Cancel."
    if state.click_mode == "create_marker":
        return "Click geometry to create a preview marker."
    if state.click_mode in {"pick_primary_vector", "pick_secondary_vector"}:
        return "Click a face for normal or edge for tangent."
    if state.click_mode == "quick_pick_primary_edge":
        marker = state.selected_marker
        return f"Click an edge to align local {marker.primary_axis} with its tangent."
    if state.click_mode == "quick_pick_primary_face":
        marker = state.selected_marker
        return f"Click a face to align local {marker.primary_axis} with its normal."
    if state.click_mode == "pick_line_point_a":
        return "Click first point for line direction."
    if state.click_mode == "pick_line_point_b":
        return "Click second point for line direction."
    return "Press New Marker, then click geometry."
