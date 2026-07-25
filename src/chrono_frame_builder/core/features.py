"""Feature-candidate helpers for future visual frame construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureCandidate:
    """A geometry-derived candidate point and optional direction for frame construction."""

    kind: str
    source_type: str
    point: np.ndarray
    direction: np.ndarray | None = None
    source_ids: tuple[int | str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_point(point: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")

    return point


def _normalized(vector: np.ndarray, message: str, tolerance: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("Feature direction vectors must have shape (3,).")

    length = np.linalg.norm(vector)
    if length <= tolerance:
        raise ValueError(message)

    return vector / length


def triangle_center(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return the centroid of a triangle."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    p2 = _as_point(p2, "p2")
    return (p0 + p1 + p2) / 3.0


def triangle_normal(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Return the normalized triangle normal using the p0, p1, p2 winding."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    p2 = _as_point(p2, "p2")
    normal = np.cross(p1 - p0, p2 - p0)
    return _normalized(normal, "Degenerate triangle cannot define a face normal.", tolerance)


def edge_midpoint(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Return the midpoint of an edge."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    return (p0 + p1) / 2.0


def edge_tangent(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Return the normalized edge tangent from p0 to p1."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    return _normalized(p1 - p0, "Zero-length edge cannot define a tangent.", tolerance)


def point_candidate(
    point: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> FeatureCandidate:
    """Create a point candidate suitable for a frame origin."""
    return FeatureCandidate(
        kind="point",
        source_type=source_type,
        point=_as_point(point, "point"),
        source_ids=tuple(source_ids),
        metadata=dict(metadata or {}),
    )


def triangle_face_candidate(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> FeatureCandidate:
    """Create a triangle-face candidate with face center and normal."""
    return FeatureCandidate(
        kind="triangle_face",
        source_type=source_type,
        point=triangle_center(p0, p1, p2),
        direction=triangle_normal(p0, p1, p2),
        source_ids=tuple(source_ids),
        metadata=dict(metadata or {}),
    )


def mesh_edge_candidate(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> FeatureCandidate:
    """Create a mesh-edge candidate with edge midpoint and tangent."""
    return FeatureCandidate(
        kind="mesh_edge",
        source_type=source_type,
        point=edge_midpoint(p0, p1),
        direction=edge_tangent(p0, p1),
        source_ids=tuple(source_ids),
        metadata=dict(metadata or {}),
    )
