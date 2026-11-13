"""Private, disposable reference surface samples. A miss never changes the scoring path."""

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from sbv.common import scene as scene_module
from sbv.geometry import g2_shape as G


def cache_key(scene, names, count):
    """Hash only the geometry actually read by surface_samples, without textures or model serialization."""
    m, d = scene.m, scene.d
    h = hashlib.sha256(json.dumps([names, count, np.__version__, mujoco.__version__, trimesh.__version__]).encode())

    def add(a):
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))

    meshes = set()
    for n in names:
        gs = scene.visible(scene.ogeoms[n])
        add(gs)
        for g in gs:
            add(m.geom_type[g : g + 1])
            add(m.geom_size[g])
            add(d.geom_xpos[g])
            add(d.geom_xmat[g])
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            i = int(m.geom_dataid[g])
            add([i])
            if i not in meshes:
                meshes.add(i)
                add(m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]])
                add(m.mesh_face[m.mesh_faceadr[i] : m.mesh_faceadr[i] + m.mesh_facenum[i]])
    for source in (__file__, G.__file__, scene_module.__file__):
        h.update(Path(source).read_bytes())
    return h.hexdigest()


def samples(scene, slots, count):
    names = [o["slot"] for o in slots]

    def compute():
        return {n: G.surface_samples(scene, n, count) for n in names}

    folder = os.environ.get("SBV_GEOMETRY_CACHE", "off")
    if not folder or folder == "off":
        return compute()
    path = Path(folder) / (cache_key(scene, names, count) + ".npz")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = [data[f"p{i}"] for i in range(len(names))]
            if all(a.dtype == np.float64 and a.shape in ((0, 3), (count, 3)) and np.isfinite(a).all() for a in arrays):
                return dict(zip(names, arrays))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    result = compute()
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            tmp = f.name
            np.savez(f, **{f"p{i}": result[n] for i, n in enumerate(names)})
        os.replace(tmp, path)  # concurrent writers publish complete identical entries
    except OSError:
        pass  # read-only/full cache is not an evaluation failure
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return result
