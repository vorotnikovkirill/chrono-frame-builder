# Hole-Block Viewer Demo

Run the demo with the optional viewer dependencies installed:

```bash
python3 -m chrono_frame_builder.viewer examples/hole_block/project.json --create-frame
```

`hole_block.obj` is an 80 x 40 x 16 block with a radius-8 through-hole centered at `(15, 0)` and
running along Z. It is generated deterministically by `make_hole_block.py` without third-party
dependencies:

```bash
python3 examples/hole_block/make_hole_block.py
```

The 48-sided hole is deliberately faceted. The current mesh picker can expose triangle face
normals, vertices, edge midpoints, and edge tangents around it. It does not yet recognize a true
circle center or cylinder axis from STL/OBJ geometry.
