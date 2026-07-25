# Frame math convention

This document defines the frame transform convention used by chrono-frame-builder.

## Frame definition

Each frame has:

- origin
- rotation_matrix

In schema v0, both are stored relative to the project/global coordinate system.

## Rotation matrix convention

For a frame A:

R_GA means:

- rotation matrix from frame A to global frame G
- columns of R_GA are the local axes of frame A expressed in global coordinates

Therefore:

v_G = R_GA * v_A

where:

- v_A is a vector expressed in frame A coordinates
- v_G is the same vector expressed in global coordinates

## Homogeneous transform

A frame A relative to global frame G is represented as:

T_GA = [ R_GA  p_GA ]
       [ 0     1    ]

where:

- R_GA is the 3x3 rotation matrix
- p_GA is the origin of frame A expressed in global coordinates

## Relative transform from frame A to frame B

Given:

T_GA
T_GB

The transform from A to B is:

T_AB = inverse(T_GA) * T_GB

This gives:

R_AB = transpose(R_GA) * R_GB

p_AB_A = transpose(R_GA) * (p_GB - p_GA)

where:

- R_AB is the orientation of frame B relative to frame A
- p_AB_A is the position of frame B origin relative to frame A origin, expressed in frame A coordinates

## Practical interpretation

If two frames are identical in space:

p_AB_A = [0, 0, 0]

R_AB = identity matrix

This is useful for checking whether two frames are compatible for a fixed joint.

## Units

The project may store length in millimeters.

Transform math does not convert units.

Unit conversion should happen in adapters, for example when exporting to Chrono.

## Feature-based frame construction

The intended custom-frame workflow is feature based rather than based on three arbitrary
points. A future viewer should let the user choose:

- an origin feature, such as a vertex, face center, edge midpoint, or inertial feature
- a primary signed frame axis, such as `+Z`, aligned to a selected vector such as a face normal
- a secondary signed frame axis, such as `+X`, aligned to a selected vector such as an edge
  direction after projection onto the plane normal to the primary axis

The remaining axis is constructed automatically with cross products so the rotation matrix is
right-handed and orthonormal. The core helper for this is
`frame_from_origin_primary_secondary(...)`.

The three-point preview helper is a temporary convenience built on the same math:

- P0 is the origin
- P1 defines `+X`
- P2 supplies a direction that is projected to define `+Y`

## Feature candidates

Stage B1c introduces feature candidates as the bridge between raw viewer picking and future
feature-based frame creation. A candidate records:

- feature kind, such as `point`, `triangle_face`, or `mesh_edge`
- source type, currently `mesh`
- a candidate point, such as a picked point, triangle center, or edge midpoint
- an optional normalized direction, such as a triangle normal or edge tangent
- optional source IDs and metadata for tracing the candidate back to mesh cells, edges, files,
  or later BRep entities

Mesh and STL features are approximate because they come from tessellated triangles and edges.
They are useful for visual workflows, but future STEP/BRep features should provide more exact
surface normals, edge directions, circle centers, and analytic feature metadata.
