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

Open a project:

```bash
chrono-frame-viewer path/to/project.json
```

The Stage B0 viewer supports STL, OBJ, PLY, VTK, and VTP mesh files listed in each body's
`cad_file` field. Unsupported or missing geometry is reported as a warning, and stored frame
axes are still drawn.

## What this project is not

This is not a replacement for Simscape Multibody.

This is not a full CAD system.

This is not a dynamics solver.

This is a frame/marker preparation and inspection tool for Chrono-based mechanism models.

## Status

Pre-alpha. The project structure and data model are being defined.
