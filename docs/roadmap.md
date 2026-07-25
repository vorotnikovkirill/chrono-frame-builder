# Roadmap

## Stage 0 - Project foundation

Goal:

Create the repository structure and define the project concept, data schema, and first example project file.

Scope:

- README
- project concept
- project schema
- roadmap
- minimal example project.json
- git initialization

## Stage 1 - Core data model

Goal:

Implement the Python core without GUI.

Features:

- Body data class
- Frame data class
- Project data class
- load project.json
- save project.json
- add body
- add frame
- get frame by body.frame name
- missing-frame diagnostics

## Stage 2 - Transform math

Goal:

Implement reliable frame transform operations.

Features:

- 3x3 rotation matrix validation
- homogeneous transform
- inverse transform
- relative transform
- distance between frames
- quaternion output
- Euler angle output

## Stage 3 - CLI tools

Goal:

Allow the user to inspect a project from the terminal.

Commands:

- list bodies
- list frames
- inspect frame
- measure frame-to-frame transform

## Stage 4 - Basic 3D viewer

Goal:

Show bodies and frames in a simple viewer.

Features:

- rotate
- zoom
- pan
- show/hide bodies
- color control
- opacity control
- show frame axes

## Stage B0 - Minimal visual inspection viewer

Goal:

Provide a small PyVista-based viewer before the full interactive viewer stage.

Features:

- load `project.json`
- display supported mesh files referenced by body `cad_file`
- draw axes for frames already stored in the project
- report missing or unsupported geometry without editing the project

Out of scope:

- point picking
- frame editing
- saving from the viewer
- STEP or Parasolid import
- FEM, MNF, Adams, and Chrono code generation

## Stage B0.1 - Reproducible viewer demo

Goal:

Provide a tiny project that opens in the Stage B0 viewer and shows both mesh geometry and
stored frames.

Features:

- `examples/viewer_demo/project.json`
- small human-readable ASCII STL geometry
- body/reference frame and offset marker frame

## Stage 5 - Interactive frame creation

Goal:

Allow the user to create frames on imported geometry.

Initial features:

- create frame from picked point
- orientation parallel to global frame
- manual frame orientation

Later features:

- edge midpoint
- face center
- circular edge center
- face normal
- edge direction
- cylinder axis

## Stage 6 - Frame inspector

Goal:

Compare two frames.

Outputs:

- dx, dy, dz
- distance
- rotation matrix
- quaternion
- Euler angles
- expressed in global / first frame / second frame

## Stage 7 - Joint compatibility checks

Goal:

Check whether two frames are suitable for a joint.

Joint types:

- fixed
- revolute
- prismatic
- spherical

Outputs:

- PASS
- WARNING
- FAIL

## Stage 8 - PyChrono adapter

Goal:

Use saved frames directly in PyChrono scripts.

Features:

- create ChFrame from stored frame
- convert mm to meters
- helper functions for joints
- missing-frame errors with useful messages

## Stage 9 - Code generation

Goal:

Generate clean PyChrono code templates from the project file.

This stage comes after the data model, measurement tools, and adapter are stable.
