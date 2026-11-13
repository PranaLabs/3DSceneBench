"""A draped sheet (the cloth shape in nonrigid.json) over a rail along x: front panel toward +y, back panel toward the
wall."""

import numpy as np, trimesh

MAX_VERTICES = 200_000  # a 10 m x 12 m sheet at the 2.5 cm spacing; a room's curtain is a few thousand


def draped_sheet(rail_c, rail_r, width, front, back, spacing=0.025, wall_gap=0.06):
    n = (
        (max(front, 0) + max(back, 0)) / spacing * (max(width, 0) / spacing + 1)
    )  # vertices, before any is built: the numbers are a copy's
    if not n < MAX_VERTICES:
        raise ValueError(f"a draped sheet of about {n:.3g} vertices (at most {MAX_VERTICES})")
    r = rail_r + 0.006
    arc = np.linspace(0, np.pi, 9)  # over the top, from wall side (-y) to front (+y)
    prof = [(rail_c[1] - r * np.cos(a), rail_c[2] + r * np.sin(a)) for a in arc]
    prof = [(rail_c[1] - r, rail_c[2] - s) for s in np.arange(back, 0, -spacing)] + prof
    prof += [(rail_c[1] + r, rail_c[2] - s) for s in np.arange(spacing, front + 1e-9, spacing)]
    xs = np.arange(-width / 2, width / 2 + 1e-9, spacing) + rail_c[0]
    V = np.array([(x, y, z) for (y, z) in prof for x in xs])
    nx = len(xs)
    F = []
    for i in range(len(prof) - 1):
        for j in range(nx - 1):
            a, b, c, d = i * nx + j, i * nx + j + 1, (i + 1) * nx + j, (i + 1) * nx + j + 1
            F += [(a, c, b), (b, c, d)]
    return trimesh.Trimesh(V, np.array(F), process=False)
