# Project schema v0

This document describes the first project file structure.

The project file is named:

project.json

## Top-level fields

Required top-level fields:

- schema_version
- project_name
- units
- bodies
- frames

Example structure:

{
  "schema_version": "0.1.0",
  "project_name": "minimal_pendulum",
  "units": {},
  "bodies": [],
  "frames": []
}

## Units

Example:

{
  "length": "mm",
  "angle": "deg"
}

The project may store CAD-style units such as millimeters.

Chrono adapters should convert values to SI units when needed.

## Body object

A body represents a CAD-derived rigid body or a future Chrono body.

Example:

{
  "name": "arm",
  "cad_file": "cad/arm.step",
  "color": [0.2, 0.5, 0.9],
  "opacity": 0.8,
  "fixed": false
}

Body fields:

- name: unique body name
- cad_file: relative path to CAD or mesh file
- color: RGB values from 0.0 to 1.0
- opacity: value from 0.0 to 1.0
- fixed: whether this body is intended to be fixed in the simulation

## Frame object

A frame is a local coordinate system attached to a body.

Example:

{
  "body": "arm",
  "name": "hinge_A",
  "origin": [0.0, 0.0, 0.0],
  "rotation_matrix": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
}

Frame fields:

- body: parent body name
- name: frame name on the body
- origin: frame origin coordinates
- rotation_matrix: frame orientation as a 3x3 rotation matrix

## Frame naming convention

Frames are referenced by:

body.frame

Examples:

base.hinge_A
arm.hinge_A
tool.mount_frame

## Notes

In v0, all frame origins and rotations are assumed to be expressed in the project/global coordinate system.

Later versions may add explicit local/global frame storage modes.
