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
PyVista view in the center and an embedded right-side control dock. The marker-editor workflow
starts in Select/Edit mode. Press **New Marker** to enter one-click Create Marker mode;
the next selected mesh origin creates a named preview marker and immediately returns to Select/Edit
mode. This prevents ordinary geometry clicks and vector-picking clicks from creating accidental
markers. Multiple compact RGB triads and labels remain visible at once. Selecting a marker in the
list loads its editable name, `X/Y/Z` origin, and `3 x 3` rotation matrix. Manual matrices must
be finite, orthonormal, and right-handed (`det(R) = +1`) before the selected preview marker is
updated.

The dock separates **Frame Origin**, **Primary Axis**, and **Secondary Axis**. Each axis chooses
a signed marker axis and an independent signed reference/source axis. In reference mode, primary
marker `+Z` aligned to reference `+X`, together with secondary marker `+X` aligned to reference
`+Y`, produces columns `X=global Y`, `Y=global Z`, and `Z=global X`:
`[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`.

For geometric mode, **Pick Primary Feature** and **Pick Secondary Feature** switch the viewer to
an explicit one-click vector-picking mode. A selected face/triangle is highlighted with a normal
arrow, and a selected edge is highlighted with a tangent arrow. This creates a pending candidate,
not an immediate orientation change: **Apply Candidate** assigns it, **Flip Candidate** reverses
the displayed direction, and **Cancel Candidate** discards it. Points and vertices do not provide
face/edge axis directions; they can be used as the two points of a Quick Orientation line. The
secondary vector is projected by
the existing feature-based math layer to remove roll ambiguity. Principal inertia-axis options are
visibly disabled until an inertia backend exists.

**Quick Orientation** provides a more direct primary-axis path: choose a local axis such as `+Z`,
then stage an edge tangent, a face normal, or a line from point A to point B. The selected vector
is still previewed before applying it. The Secondary Axis remains necessary to remove roll
ambiguity. When it is parallel to the quick primary direction, the editor retains the primary
source, keeps the last valid triad, and asks for a non-parallel secondary definition. Mesh
directions remain tessellated STL/OBJ approximations, not true CAD circle or cylinder recognition.

**Save Selected Marker** is an explicit persistence action. It validates the selected marker,
creates a compatible `Frame` in the loaded project's `frames` list, then updates that linked frame
after later edits. The status is `Unsaved`, `Saved`, or `Modified`; no edits are auto-saved.
Mesh-origin and vector candidates remain approximate STL/mesh features; STEP/BRep extraction plus
analytic holes, cylinders, and circle centers are future work. If Qt is unavailable, the viewer
reports the condition and uses its standalone PyVista/Tk or keyboard fallback.

## Coordinate hierarchy and mesh-hole demo

The intended transform hierarchy is `world/assembly -> body/part -> marker/frame`. The current
`Body` schema has CAD metadata but no placement transform, while `Frame` stores its origin and
rotation with a parent body name. Therefore the current viewer displays and saves body/part-local
Cartesian XYZ coordinates. The included examples use a single body with an implicit identity body
transform; full assembly/body transform conversion is planned and is not implied by the UI.

`examples/hole_block` provides a deterministic OBJ block with a 48-sided through-hole. It is useful
for testing planar triangle normals and faceted hole-boundary edge tangents. As with other STL/OBJ
meshes, this is tessellated geometry only: no true circle center, hole feature, or cylinder-axis
recognition is implemented yet.
