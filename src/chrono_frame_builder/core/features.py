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


@dataclass(frozen=True)
class MeshEdgeSnapCandidate:
    """A mesh edge with selectable start, midpoint, end, and tangent handles."""

    kind: str
    source_type: str
    start: np.ndarray
    midpoint: np.ndarray
    end: np.ndarray
    tangent: np.ndarray
    source_ids: tuple[int | str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_candidate(self) -> FeatureCandidate:
        """Return the edge midpoint plus tangent as a feature candidate."""
        return FeatureCandidate(
            kind="mesh_edge",
            source_type=self.source_type,
            point=self.midpoint,
            direction=self.tangent,
            source_ids=self.source_ids,
            metadata=dict(self.metadata),
        )

    @property
    def start_candidate(self) -> FeatureCandidate:
        """Return the start vertex as a point candidate."""
        return FeatureCandidate(
            kind="mesh_edge_start",
            source_type=self.source_type,
            point=self.start,
            source_ids=self.source_ids,
            metadata=dict(self.metadata),
        )

    @property
    def midpoint_candidate(self) -> FeatureCandidate:
        """Return the edge midpoint as a point candidate."""
        return FeatureCandidate(
            kind="mesh_edge_midpoint",
            source_type=self.source_type,
            point=self.midpoint,
            source_ids=self.source_ids,
            metadata=dict(self.metadata),
        )

    @property
    def end_candidate(self) -> FeatureCandidate:
        """Return the end vertex as a point candidate."""
        return FeatureCandidate(
            kind="mesh_edge_end",
            source_type=self.source_type,
            point=self.end,
            source_ids=self.source_ids,
            metadata=dict(self.metadata),
        )


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


def unique_mesh_vertices(vertices: np.ndarray) -> np.ndarray:
    """Return unique mesh vertices while preserving first occurrence order."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must have shape (n, 3).")

    _, first_indices = np.unique(vertices, axis=0, return_index=True)
    return vertices[np.sort(first_indices)]


def unique_mesh_edges(triangle_connectivity: np.ndarray) -> np.ndarray:
    """Return unique undirected mesh edges from triangle vertex connectivity."""
    triangle_connectivity = np.asarray(triangle_connectivity, dtype=int)
    if triangle_connectivity.ndim != 2 or triangle_connectivity.shape[1] != 3:
        raise ValueError("Triangle connectivity must have shape (n, 3).")

    if len(triangle_connectivity) == 0:
        return np.empty((0, 2), dtype=int)

    edges = np.vstack(
        (
            triangle_connectivity[:, [0, 1]],
            triangle_connectivity[:, [1, 2]],
            triangle_connectivity[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def mesh_edge_snap_candidate(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> MeshEdgeSnapCandidate:
    """Create snap handles for a mesh edge."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    return MeshEdgeSnapCandidate(
        kind="mesh_edge_snap",
        source_type=source_type,
        start=p0,
        midpoint=edge_midpoint(p0, p1),
        end=p1,
        tangent=edge_tangent(p0, p1),
        source_ids=tuple(source_ids),
        metadata=dict(metadata or {}),
    )


def triangle_edge_snap_candidates(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> list[MeshEdgeSnapCandidate]:
    """Return snap candidates for the three edges of a triangle."""
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    p2 = _as_point(p2, "p2")
    vertices = (p0, p1, p2)
    edge_indices = ((0, 1), (1, 2), (2, 0))
    candidates = []

    for edge_number, (start_index, end_index) in enumerate(edge_indices):
        candidates.append(
            mesh_edge_snap_candidate(
                vertices[start_index],
                vertices[end_index],
                source_type=source_type,
                source_ids=tuple(source_ids) + (f"edge:{edge_number}",),
                metadata=metadata,
            )
        )

    return candidates


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return the shortest distance from a point to a finite line segment."""
    point = _as_point(point, "point")
    start = _as_point(start, "start")
    end = _as_point(end, "end")
    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared == 0.0:
        raise ValueError("Zero-length segment cannot be used for distance calculation.")

    t = np.clip(float(np.dot(point - start, segment) / segment_length_squared), 0.0, 1.0)
    closest_point = start + t * segment
    return float(np.linalg.norm(point - closest_point))


def nearest_triangle_edge_snap_candidate(
    point: np.ndarray,
    triangle_vertices: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> MeshEdgeSnapCandidate:
    """Return the triangle edge closest to a picked point."""
    point = _as_point(point, "point")
    triangle_vertices = np.asarray(triangle_vertices, dtype=float)
    if triangle_vertices.shape != (3, 3):
        raise ValueError("Nearest edge selection requires triangle vertices with shape (3, 3).")

    candidates = triangle_edge_snap_candidates(
        triangle_vertices[0],
        triangle_vertices[1],
        triangle_vertices[2],
        source_type=source_type,
        source_ids=source_ids,
        metadata=metadata,
    )
    return min(
        candidates,
        key=lambda candidate: point_to_segment_distance(
            point,
            candidate.start,
            candidate.end,
        ),
    )


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


def mesh_vertex_candidate(
    point: np.ndarray,
    *,
    source_type: str = "mesh",
    source_ids: tuple[int | str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> FeatureCandidate:
    """Create a mesh-vertex point candidate."""
    return FeatureCandidate(
        kind="mesh_vertex",
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
