"""The physical view of a scene: the key under which physical results are cached. Nothing runs on it."""

import os
from pathlib import Path
import xml.etree.ElementTree as ET


def physical(room):
    from .calibration_jobs import asset_dependencies
    from .dependencies import memo

    deps = asset_dependencies(room)
    return memo(("physical-view", tuple(deps.items())), lambda: _physical(room, deps))


def _physical(room, deps):
    """Canonical MJCF plus non-image assets. Paths are relative to the scene package.

    Geometry, mass, joints, contacts, visibility alpha and material links remain;
    only image pixels, RGB and rendering-only settings are removed from the key.
    """
    from .case import scene_xml
    from .dependencies import sha, xml_tree

    xml = Path(scene_xml(str(room))).resolve()
    tree = xml_tree(xml)
    # Preserve authored numeric precision; MjSpec.to_xml rounds floating values.
    for texture in tree.iter("texture"):
        for key in ("file", "fileup", "filedown", "filefront", "fileback", "fileleft", "fileright", "content_type"):
            texture.attrib.pop(key, None)
        texture.set("builtin", "flat")
        texture.set("width", "1")
        texture.set("height", "1")
    for parent in tree.iter():
        for child in list(parent):
            # <statistic> stays: its meaninertia scales the solver's tolerances, so it is physics
            if child.tag in ("visual", "camera", "light"):
                parent.remove(child)
        for key in (
            "emission",
            "specular",
            "shininess",
            "reflectance",
            "metallic",
            "roughness",
            "texrepeat",
            "texuniform",
        ):
            parent.attrib.pop(key, None)
        if "rgba" in parent.attrib:
            rgba = parent.attrib["rgba"].split()
            parent.set("rgba", "0 0 0 " + rgba[-1])
        if parent.tag == "compiler":
            parent.attrib.pop("texturedir", None)
        if parent.tag == "texture":
            parent.attrib.pop("rgb1", None)
            parent.attrib.pop("rgb2", None)
    for element in tree.iter():
        element.text = element.text.strip() or None if element.text else None
        element.tail = None
    # The flattened XML no longer reads its includes. Keep only assets it names.
    assets = {}
    dirs = {xml.parent, *(Path(p).parent for p in deps if Path(p).suffix == ".xml")}
    comp = tree.find("compiler")
    if comp is not None:
        dirs |= {d / comp.get(k) for d in list(dirs) for k in ("assetdir", "meshdir", "texturedir") if comp.get(k)}
    for element in tree.iter():
        file = element.get("file")
        if not file:
            continue
        sub = (
            (comp.get("meshdir") if element.tag in ("mesh", "skin") else comp.get("assetdir"))
            if comp is not None
            else ""
        )
        candidates = [xml.parent / (sub or "") / file] + [d / file for d in sorted(dirs)]
        matches = [p for p in candidates if p.is_file()]
        if not matches:
            raise ValueError("unresolved physical asset: " + file)
        p = matches[0]
        name = os.path.relpath(p.resolve(), xml.parent)
        assets[name] = sha(p)
        element.set("file", name)
    compiler = tree.find("compiler")
    if compiler is not None:
        for key in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(key, None)
    return dict(xml=ET.tostring(tree, encoding="unicode"), assets=dict(sorted(assets.items())))
