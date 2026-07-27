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

Stage B1d connects the PyVista viewer to this layer in preview-only mode. Surface picks try
to recover the picked triangle. Clicks near an edge report a `mesh_edge_snap` candidate with
start, midpoint, end, and tangent handles. Picks in the triangle interior report a
`triangle_face` candidate with center and normal. If triangle-cell information is unavailable
or unsupported, the viewer creates a `point` fallback candidate at the picked position rather
than inventing a normal.

These snap handles are still mesh features, not true CAD topology. STL triangle edges may be
tessellation artifacts. Future STEP/BRep support should provide true CAD edges, faces, circle
centers, arcs, and cylinder axes.

The inspection viewer is a preview/debug mode. It helps the user understand available mesh
candidates, but it does not assign origin, primary-axis, or secondary-axis roles and does not
save frames.

The `--create-frame` viewer mode starts the Simscape-like workflow. With the optional
`.[visualization,ui]` dependencies installed, it opens one Qt application window with the
PyVista view in the center and an embedded right-side control dock. The default Create marker
click mode creates a new named preview marker at each selected mesh origin, so multiple compact
RGB triads and labels remain visible at once. Selecting a marker in the list loads its editable
name, `X/Y/Z` origin, and `3 x 3` rotation matrix. Manual matrices must be finite, orthonormal,
and right-handed (`det(R) = +1`) before the selected preview marker is updated.

The dock separates **Frame Origin**, **Primary Axis**, and **Secondary Axis**. Each axis chooses
a signed frame axis and either follows the reference frame or uses a selected face normal or edge
tangent. The secondary vector is projected by the existing feature-based math layer. Principal
inertia-axis options are visibly disabled until an inertia backend exists. If Qt is unavailable,
the viewer reports the condition and uses its standalone PyVista/Tk or keyboard fallback. None
of these preview paths save frames or create final project data.
