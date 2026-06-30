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
