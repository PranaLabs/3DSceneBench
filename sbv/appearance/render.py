"""Appearance renders (doc, Appearance: rendering protocol). A MuJoCo scene is rebuilt in Blender at its submitted
initial state, from the compiled model: visual geoms (group below 3 when the scene has any, else all geoms), their
colours, and image textures: on meshes that carry UVs by their UVs, on shapes without UVs by a box projection in room
coordinates that tiles texrepeat[0] times per metre on every axis. That is MuJoCo's texuniform="true" (repeats per
spatial unit); with MuJoCo's default texuniform="false" a 2D texture repeats texrepeat times per face whatever its size,
so a material without texuniform="true" on a shape without UVs renders here at a different tile size than MuJoCo shows
(a 2 m face with texrepeat 1: one tile in MuJoCo, two here). texrepeat[1] is not read. Every scene gets the same
lighting (a ceiling area light over the case's room bounds plus a dim grey world), the same colour management (Standard
view, no look, exposure 0), and Cycles at 512 x 512, 64 samples, seed 0, OpenImageDenoise. Runs with Blender's Python
module:
    .venv/bin/python -m sbv.appearance.render SCENE.xml CAMERAS.json OUT_DIR
CAMERAS.json: {"bounds": [[x0,y0,z0],[x1,y1,z1]], "resolution": 512, "samples": 64, "cameras": [{"name", "position",
"look_at", "hfov_deg"}]}; "architecture_only": true draws the walls, floor and ceiling without the objects (the
appearance floor, sbv/appearance/views.py floor()). Writes OUT_DIR/<name>.png and OUT_DIR/render.json (device, Blender
version, timings).
"""

import json, os, sys, time
import numpy as np, mujoco, bpy
from mathutils import Vector
from sbv.geometry.g2_shape import geom_mesh

TEX_RGB = 1  # mjTEXROLE_RGB
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # MuJoCo's internal default geom colour
LIGHT_W_PER_M2 = float(os.environ.get("SBV_LIGHT", 8.0))  # ceiling light power per square metre of room


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def texture_files(xml):
    """Compiled texture id -> image file on disk (textures keep their order when compiled; by id, because an unnamed
    texture's compiled name comes from its file), resolved against texturedir and the model folder."""
    spec = mujoco.MjSpec.from_file(xml)
    base = os.path.dirname(os.path.abspath(xml))
    folder = spec.texturedir or ""
    return {i: os.path.normpath(os.path.join(base, folder, t.file)) for i, t in enumerate(spec.textures) if t.file}


def texture_flips(xml):
    """Compiled texture id -> (hflip, vflip): MuJoCo flips such an image when it loads it, Blender reads the file as it
    is, so the UVs flip instead."""
    spec = mujoco.MjSpec.from_file(xml)
    return {i: (bool(t.hflip), bool(t.vflip)) for i, t in enumerate(spec.textures)}


class Builder:
    def __init__(self, scene, xml):
        self.s, self.m, self.d = scene, scene.m, scene.d
        self.mats = {}
        self.files = texture_files(xml)
        self.flips = texture_flips(xml)

    def material(self, g, uvless=False):
        m = self.m
        mat_id = int(m.geom_matid[g])
        tex = int(m.mat_texid[mat_id, TEX_RGB]) if mat_id >= 0 else -1
        own = not np.allclose(m.geom_rgba[g], DEFAULT_RGBA)  # MuJoCo: a geom rgba other than the default
        rgba = m.geom_rgba[g] if mat_id < 0 or own else m.mat_rgba[mat_id]  # takes precedence over the material colour
        if tex >= 0 and tex not in self.files:
            tex = -1  # built-in or missing texture: flat colour
        rough = (
            float(m.mat_roughness[mat_id]) if mat_id >= 0 and m.mat_roughness[mat_id] >= 0 else 0.6
        )  # MJCF roughness and metallic
        metal = (
            float(m.mat_metallic[mat_id]) if mat_id >= 0 and m.mat_metallic[mat_id] >= 0 else 0.0
        )  # when the material sets them (-1: unset)
        uvless = uvless and tex >= 0
        key = (tuple(np.round(rgba, 4)), tex, round(rough, 3), round(metal, 3), uvless)
        if key in self.mats:
            return self.mats[key]
        mat = bpy.data.materials.new(f"mat{len(self.mats)}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(rgba[:3]), 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        if rgba[3] < 1:
            bsdf.inputs["Alpha"].default_value = float(rgba[3])
        if tex >= 0:
            try:
                image = bpy.data.images.load(self.files[tex], check_existing=True)
            except Exception:
                image = None  # a picture Blender cannot read: its flat colour, and the render goes on
            if image is not None:
                nodes, links = mat.node_tree.nodes, mat.node_tree.links
                img = nodes.new("ShaderNodeTexImage")
                img.image = image
                colour = img.outputs["Color"]
                if uvless:  # a shape without UVs: a box projection in room coordinates,
                    coord, mp = (
                        nodes.new("ShaderNodeTexCoord"),
                        nodes.new("ShaderNodeMapping"),
                    )  # texrepeat tiles per metre: MuJoCo's
                    # texuniform="true"; m.mat_texuniform is not read, so texuniform="false" (per face in MuJoCo) tiles
                    # per metre too
                    r = float(m.mat_texrepeat[mat_id][0]) or 1.0
                    mp.inputs["Scale"].default_value = (r, r, r)
                    links.new(coord.outputs["Object"], mp.inputs["Vector"])
                    links.new(mp.outputs["Vector"], img.inputs["Vector"])
                    img.projection, img.projection_blend = "BOX", 0.2  # (was: every face took one texel, at UV 0,0)
                if not np.allclose(rgba[:3], 1.0):  # MuJoCo multiplies the texture by the material colour
                    mix = nodes.new("ShaderNodeMixRGB")
                    mix.blend_type = "MULTIPLY"
                    mix.inputs["Fac"].default_value = 1.0
                    links.new(colour, mix.inputs["Color1"])
                    mix.inputs["Color2"].default_value = (*srgb_to_linear(rgba[:3]), 1.0)
                    colour = mix.outputs["Color"]
                links.new(colour, bsdf.inputs["Base Color"])
        self.mats[key] = mat
        return mat

    def add(self, g, bounds):
        m, d = self.m, self.d
        uv = None
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            (x0, y0, _), (x1, y1, _) = bounds
            R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            f = np.array([[0, 1, 2], [0, 2, 3]])
            if R[2, 2] > 0.99:  # a floor: cover the room bounds
                e = 5.0  # the bounds come from the objects; walls can stand metres outside them
                v = np.array(
                    [[x0 - e, y0 - e, p[2]], [x1 + e, y0 - e, p[2]], [x1 + e, y1 + e, p[2]], [x0 - e, y1 + e, p[2]]],
                    float,
                )
            else:  # any other plane: its own size, or 5 m
                sx, sy = [
                    x if x > 0 else 100.0 for x in m.geom_size[g][:2]
                ]  # an infinite plane (size 0) as a 200 m square: 10 m left floor out of 8 m rooms
                v = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]]) @ R.T + p
        else:
            r = geom_mesh(self.s, g)
            if r is None:
                return
            v, f = r
            i = m.geom_dataid[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh_texcoordadr[i] >= 0:
                tc = m.mesh_texcoord[m.mesh_texcoordadr[i] : m.mesh_texcoordadr[i] + m.mesh_texcoordnum[i]]
                ft = m.mesh_facetexcoord[m.mesh_faceadr[i] : m.mesh_faceadr[i] + m.mesh_facenum[i]]
                uv = tc[ft]
        me = bpy.data.meshes.new(f"g{g}")
        me.from_pydata(v.tolist(), [], f.tolist())
        me.update()
        if uv is not None:
            layer = me.uv_layers.new(name="uv")
            mat_id = int(m.geom_matid[g])
            tex = int(m.mat_texid[mat_id, TEX_RGB]) if mat_id >= 0 else -1
            hf, vf = self.flips.get(tex, (False, False))
            for p in me.polygons:
                for k, li in enumerate(p.loop_indices):
                    u, w = uv[p.index, k, 0], uv[p.index, k, 1]
                    layer.data[li].uv = (1.0 - u if hf else u, w if vf else 1.0 - w)
        me.materials.append(self.material(g, uvless=uv is None))
        ob = bpy.data.objects.new(f"g{g}", me)
        bpy.context.scene.collection.objects.link(ob)


def render(xml, spec, out, phys=None):
    from sbv.common.scene import Scene

    t0 = time.time()
    scene = Scene(xml, phys)
    scene.reset(scene.q0)
    m = scene.m
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    b = Builder(scene, xml)
    arch = [
        g
        for g in range(m.ngeom)
        if scene.name(scene.owner[m.geom_bodyid[g]]) not in scene.objects
        and not scene.name(scene.owner[m.geom_bodyid[g]]).startswith("probe_")
    ]
    visible = scene.visible(arch) + (
        [] if spec.get("architecture_only") else [g for n in scene.objects for g in scene.visible(scene.ogeoms[n])]
    )  # same rule as G2 shapes
    for g in visible:
        b.add(g, spec["bounds"])
    (x0, y0, _), (x1, y1, z1) = spec["bounds"]
    light = bpy.data.lights.new("ceiling", "AREA")
    light.shape = "RECTANGLE"
    light.size, light.size_y = x1 - x0, y1 - y0
    light.energy = LIGHT_W_PER_M2 * (x1 - x0) * (y1 - y0)
    lo = bpy.data.objects.new("ceiling", light)
    lo.location = (
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        spec.get("light_z", min(z1, 2.6)),
    )  # the case's: just under the reference room's ceiling
    s.collection.objects.link(lo)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.4
    s.world = world
    s.render.engine = "CYCLES"
    s.cycles.samples = spec.get("samples", 64)
    s.cycles.seed = 0
    s.cycles.use_denoising = True
    s.cycles.denoiser = "OPENIMAGEDENOISE"
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    s.view_settings.exposure = 0
    s.render.resolution_x = s.render.resolution_y = spec.get("resolution", 512)
    s.render.image_settings.color_mode = "RGB"
    device = "CPU"
    if os.environ.get("SBV_RENDER_DEVICE", "CPU").upper() == "GPU":  # GPU renders are faster but not bit-repeatable
        s.cycles.device = "CPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        backends = (
            (os.environ["SBV_CYCLES_BACKEND"],)
            if os.environ.get("SBV_CYCLES_BACKEND")
            else ("OPTIX", "CUDA", "HIP", "METAL")
        )
        for kind in backends:  # the first backend with a GPU on this machine
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [dv for dv in prefs.devices if dv.type == kind]
                if not gpus:
                    continue
                for dv in prefs.devices:
                    dv.use = dv.type == kind
                s.cycles.device = "GPU"
                device = kind
                break
            except Exception:
                continue
    else:
        s.cycles.device = "CPU"
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    s.collection.objects.link(cam)
    s.camera = cam
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start, cam_data.clip_end = 0.05, 50.0
    os.makedirs(out, exist_ok=True)
    times = {}
    for c in spec["cameras"]:
        cam_data.angle_x = np.radians(c["hfov_deg"])
        cam.location = Vector(c["position"])
        cam.rotation_euler = (Vector(c["look_at"]) - Vector(c["position"])).to_track_quat("-Z", "Y").to_euler()
        t = time.time()
        s.render.filepath = os.path.join(out, c["name"] + ".png")
        bpy.ops.render.render(write_still=True)
        times[c["name"]] = round(time.time() - t, 2)
    info = dict(
        blender=bpy.app.version_string,
        device=device,
        geoms=len(visible),
        materials=len(b.mats),
        seconds=times,
        total=round(time.time() - t0, 1),
    )
    with open(os.path.join(out, "render.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    xml, cams, out = sys.argv[1:4]
    with open(cams) as f:
        spec = json.load(f)
    print(json.dumps(render(xml, spec, out, spec.get("physics"))))
