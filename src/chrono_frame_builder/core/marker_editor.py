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
    source_metadata: dict[str, Any] = field(default_factory=dict)
    primary_axis: str = "+Z"
    secondary_axis: str = "+X"
    primary_axis_source_mode: str = "reference"
    secondary_axis_source_mode: str = "reference"
    primary_feature: FeatureSource | None = None
    secondary_feature: FeatureSource | None = None

    def as_frame(self) -> Frame:
        """Return this preview marker in the existing frame representation."""
        return Frame(
            body="__preview__",
            name=self.name,
            origin=self.origin.copy(),
            rotation_matrix=self.rotation_matrix.copy(),
        )


@dataclass
class CreateFrameEditorState:
    """Qt-independent multi-marker editor state for the preview-only create-frame workflow."""

    markers: list[PreviewMarker] = field(default_factory=list)
    selected_marker_id: int | None = None
    selected_feature: FeatureSource | None = None
    click_mode: str = "create_marker"
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
        marker_id = self._next_marker_id
        self._next_marker_id += 1
        marker = PreviewMarker(
            id=marker_id,
            name=f"marker_{marker_id:03d}",
            origin=feature_origin_point(candidate),
            rotation_matrix=np.eye(3),
            origin_source="feature",
            orientation_source="reference",
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
        self.warning = None
        return marker

    def select_feature(self, candidate: FeatureSource) -> None:
        """Store a geometry feature for a later frame-origin or axis assignment."""
        self.selected_feature = candidate
        self.warning = None

    def set_click_mode(self, click_mode: str) -> None:
        """Set the viewer click behavior to marker creation or feature selection."""
        if click_mode not in {"create_marker", "select_feature"}:
            raise ValueError("Click mode must be 'create_marker' or 'select_feature'.")
        self.click_mode = click_mode

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
        previous_mode = marker.primary_axis_source_mode
        previous_feature = marker.primary_feature
        marker.primary_axis_source_mode = "feature"
        marker.primary_feature = candidate
        try:
            marker = self.recompute_selected_marker_orientation()
        except ValueError:
            marker.primary_axis_source_mode = previous_mode
            marker.primary_feature = previous_feature
            raise
        marker.source_metadata["primary_axis"] = feature_source_metadata(candidate)
        return marker

    def use_selected_feature_for_secondary_axis(self) -> PreviewMarker:
        """Assign the selected directional feature as the selected marker's secondary vector."""
        marker = self._require_selected_marker()
        candidate = self._require_selected_direction_feature()
        previous_mode = marker.secondary_axis_source_mode
        previous_feature = marker.secondary_feature
        marker.secondary_axis_source_mode = "feature"
        marker.secondary_feature = candidate
        try:
            marker = self.recompute_selected_marker_orientation()
        except ValueError:
            marker.secondary_axis_source_mode = previous_mode
            marker.secondary_feature = previous_feature
            raise
        marker.source_metadata["secondary_axis"] = feature_source_metadata(candidate)
        return marker

    def reset_selected_marker_axes_to_reference(self) -> PreviewMarker:
        """Restore reference-axis orientation for the selected marker."""
        marker = self._require_selected_marker()
        marker.primary_axis_source_mode = "reference"
        marker.secondary_axis_source_mode = "reference"
        marker.primary_feature = None
        marker.secondary_feature = None
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
        axis_selector = marker.primary_axis if primary else marker.secondary_axis
        feature = marker.primary_feature if primary else marker.secondary_feature
        axis_name = "primary" if primary else "secondary"
        if mode == "reference":
            return reference_axis_direction(axis_selector)
        if mode == "inertia":
            raise ValueError("Principal inertia axis support is not available yet.")
        if feature is None:
            raise ValueError(f"Assign a geometric feature for the {axis_name} axis first.")
        return feature_direction_vector(feature)

    def _require_selected_marker(self) -> PreviewMarker:
        marker = self.selected_marker
        if marker is None:
            raise ValueError("Select or create a preview marker first.")
        return marker

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
    if marker is None:
        return {
            "selected_marker": "<none>",
            "origin": "<none>",
            "origin_source": "<none>",
            "primary_feature": "<none>",
            "secondary_feature": "<none>",
            "message": state.warning or "Click geometry to create a preview marker.",
        }

    primary_kind = marker.primary_feature.kind if marker.primary_feature is not None else "<none>"
    secondary_kind = (
        marker.secondary_feature.kind if marker.secondary_feature is not None else "<none>"
    )
    return {
        "selected_marker": marker.name,
        "origin": np.array2string(marker.origin, precision=6, suppress_small=True),
        "origin_source": marker.origin_source,
        "primary_feature": primary_kind,
        "secondary_feature": secondary_kind,
        "message": state.warning or "Preview only: no project files are changed.",
    }
