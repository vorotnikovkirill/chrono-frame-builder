# Concept

## Problem

Project Chrono models are usually created directly in code.

For simple systems this is fine, but for real mechanisms the engineer must manually manage:

- rigid bodies
- body-attached frames
- marker locations
- marker orientations
- joint connection points
- relative transforms
- initial joint consistency

This can become error-prone, especially when the geometry comes from CAD.

## Proposed solution

chrono-frame-builder provides a small preparation layer before writing the main Chrono simulation script.

The tool stores CAD-derived bodies and user-defined body-attached frames in a project file.

The main simulation script can then refer to frames by name instead of hard-coded coordinates.

Example:

base.hinge_A
arm.hinge_A

Instead of manually writing raw coordinates everywhere, the user can load these frames from project.json.

## Main workflow

1. Export CAD bodies from an assembly.
2. Add the bodies to a chrono-frame-builder project.
3. Define body-attached frames / markers.
4. Save the project as project.json.
5. Inspect relative transforms between frames.
6. Use named frames in PyChrono / Chrono scripts.

## Source of truth

The source of truth is project.json.

The GUI, viewer, and generated code are secondary interfaces.

## First goal

The first goal is not CAD import or visualization.

The first goal is a reliable frame database:

- bodies
- frames
- transforms
- load/save
- frame lookup
- relative frame measurement
- diagnostics for missing frames

## Later goals

Later versions may include:

- CAD import
- 3D body viewer
- interactive frame creation
- color and opacity control
- frame-to-frame measurement GUI
- joint compatibility checks
- PyChrono helper functions
- PyChrono code generation
