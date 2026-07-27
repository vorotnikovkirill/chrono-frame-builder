# chrono-frame-builder

chrono-frame-builder is a lightweight engineering tool for defining, storing, inspecting, and validating body-attached frames on CAD-derived rigid bodies, with a focus on preparing clean Project Chrono / PyChrono multibody models.

## Purpose

Project Chrono is powerful, but building mechanisms directly in code requires careful management of:

- body coordinate systems
- local frames / markers
- joint locations
- joint orientations
- relative transforms
- initial constraint consistency

This project aims to provide a small and transparent layer between CAD geometry and Chrono scripts.

## Core idea

Workflow:

CAD bodies
-> define body-attached frames / markers
-> save project.json
-> inspect relative frame transforms
-> use the saved frames in PyChrono / Chrono code

The source of truth is not the GUI and not the generated code.

The source of truth is the project file:

project.json

## Initial scope

Version 0.1.0 focuses on the core data model:

- bodies
- frames
- project loading/saving
- frame lookup
- missing-frame diagnostics
- relative frame measurement
- rotation matrix / quaternion / Euler angle output

CAD import, 3D visualization, interactive picking, and Chrono code generation are planned for later stages.

## Stage B0 viewer

The package includes a minimal, non-editing PyVista viewer for visual inspection of stored
frames and mesh geometry already referenced by `project.json`.

Install the optional visualization dependency:

```bash
pip install chrono-frame-builder[visualization]
```

Open the reproducible viewer demo:

```bash
chrono-frame-viewer examples/viewer_demo/project.json
```

The demo project contains one small ASCII STL mesh and two predefined frames:
`demo_bracket.body_origin` and `demo_bracket.marker_tip`.

The Stage B0 viewer supports STL, OBJ, PLY, VTK, and VTP mesh files listed in each body's
`cad_file` field. Unsupported or missing geometry is reported as a warning, and stored frame
axes are still drawn.

## Stage B1a three-point frame preview

The viewer can preview a new frame from three picked mesh points:

```bash
python3 -m chrono_frame_builder.viewer examples/viewer_demo/project.json --pick-frame
```

Pick `P0` for the frame origin, `P1` for the positive X direction, and `P2` for a point in
the frame XY plane. After the third valid pick, the viewer draws preview axes and prints the
computed origin and rotation matrix to the terminal.

This mode is preview-only. It does not name a new frame, modify `project.json`, or save
anything yet.

## Stage B1d mesh snap feature inspection

The viewer can inspect mesh snap feature candidates from surface picks as a debug mode:

```bash
python3 -m chrono_frame_builder.viewer examples/viewer_demo/project.json --inspect-features
```

When the picked mesh cell can be recovered as a triangle, clicks near a triangle edge report
a `mesh_edge_snap` candidate with start, midpoint, end, and tangent handles. Picks in the
triangle interior report a `triangle_face` candidate with face center and normal. If
triangle-cell data is unavailable, the viewer falls back to a `point` candidate at the picked
position and says so.

Repeated picks of the same candidate do not reprint the same block of terminal output. This
debug mode does not create frames, name frames, assign origin/primary/secondary roles, or save
anything. Mesh and STL features are tessellated approximations; future STEP/BRep support
should provide more exact analytic feature candidates.

## Stage B1 create-frame preview

The beginning of the Simscape-like frame workflow is available in preview-only mode:

```bash
python3 -m pip install -e '.[visualization,ui]'
python3 -m chrono_frame_builder.viewer examples/viewer_demo/project.json --create-frame
```

With the optional UI dependencies installed, `--create-frame` opens one integrated Qt window:
the PyVista model view fills the center and a Create Frame dock is attached on the right. Click
**New Marker**, then click geometry to create `marker_001`. The editor immediately returns to
**Select/Edit Marker** mode, so subsequent geometry clicks cannot create extra markers. Use
**New Marker** again to create `marker_002` intentionally. Each marker has a compact
global-orientation RGB triad and label, and remains in the preview marker list until the window
closes. Select a marker in that list to edit its name, position, rotation matrix, and source
status.

The **Frame Origin** section reports the geometry-origin source or manual XYZ override. The
**Primary Axis** and **Secondary Axis** sections follow the Simscape-style workflow: choose a
signed marker axis and a separate signed source/reference axis, then select **Along Reference
Frame Axis** or **Based on Geometric Feature**. For example, setting primary marker axis `+Z` to
reference axis `+X`, and secondary marker axis `+X` to reference axis `+Y`, makes local `+Z`
point along global `+X` and local `+X` point along global `+Y`. The resulting rotation matrix is
`[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`.

For geometric orientation, choose **Pick Primary Feature** or **Pick Secondary Feature**. The
active click mode changes to vector picking, so the next face normal or edge tangent click updates
the selected marker instead of creating another marker, then returns to **Select/Edit Marker**.
**Flip Primary Direction** and **Flip Secondary Direction** reverse mesh normal/tangent direction
when needed. Principal inertia-axis controls are shown as planned but disabled. The primary
direction remains dominant; the core math projects the secondary direction to remove roll
ambiguity while preserving a right-handed frame.

The selected mesh feature is only a thin visual cue; marker triads and labels remain the primary
visual objects. If the optional Qt dependencies are unavailable, the viewer reports that and
falls back to the standalone PyVista/Tk panel or its keyboard controls.

The Qt dock also includes an editable marker transform table: marker name, position `X/Y/Z`, and
the full `3 x 3` rotation matrix. **Apply marker properties** validates a finite, right-handed
orthonormal rotation before moving or rotating the selected preview marker. This gives numerical
editing equal footing with visual feature selection.

This mode previews the frame from origin, primary, and secondary feature roles using the core
feature-based frame math. It does not save `project.json`, name a new frame, or create final
frame data yet; persistent naming and saving come later.

## What this project is not

This is not a replacement for Simscape Multibody.

This is not a full CAD system.

This is not a dynamics solver.

This is a frame/marker preparation and inspection tool for Chrono-based mechanism models.

## Status

Pre-alpha. The project structure and data model are being defined.
