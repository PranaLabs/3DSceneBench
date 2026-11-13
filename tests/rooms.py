"""A small reference room built from primitives, for testing the verifiers without downloads.
4 m x 3 m: a table with a hollow cup and a book on it, a chair, a crate carrying a vase, and a cabinet fixed against the
east wall with a drawer (slide, opens toward -x) and a door (hinge). Objects follow the submission convention:
top-level bodies named <category>__<n>, free joints on movable roots, walls and floor as world geoms.
"""

import os


def box(name, pos, half, **kw):
    extra = "".join(f' {k}="{v}"' for k, v in kw.items())
    return (
        f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" size="{half[0]} {half[1]} {half[2]}"{extra}/>'
    )


def legs(prefix, hx, hy, h, t=0.02):
    return "".join(
        box(f"{prefix}_leg{i}", (sx * (hx - t), sy * (hy - t), h / 2), (t, t, h / 2))
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    )


def room_xml():
    walls = "".join(
        [
            box("wall_w", (-0.05, 1.5, 1.25), (0.05, 1.6, 1.25)),
            box("wall_e", (4.05, 1.5, 1.25), (0.05, 1.6, 1.25)),
            box("wall_s", (2.0, -0.05, 1.25), (2.1, 0.05, 1.25)),
            box("wall_n", (2.0, 3.05, 1.25), (2.1, 0.05, 1.25)),
        ]
    )
    table = (
        f'<body name="table__0" pos="1.5 1.5 0"><freejoint/>{legs("table", 0.6, 0.4, 0.70)}'
        f"{box('table_top', (0, 0, 0.72), (0.6, 0.4, 0.02))}</body>"
    )
    cup = (
        '<body name="cup__0" pos="1.3 1.35 0.74"><freejoint/>'
        + box("cup_base", (0, 0, 0.0025), (0.04, 0.04, 0.0025), density=1500)
        + "".join(
            box(f"cup_w{i}", p, h, density=1500)
            for i, (p, h) in enumerate(
                [
                    ((0.0375, 0, 0.05), (0.0025, 0.04, 0.05)),
                    ((-0.0375, 0, 0.05), (0.0025, 0.04, 0.05)),
                    ((0, 0.0375, 0.05), (0.04, 0.0025, 0.05)),
                    ((0, -0.0375, 0.05), (0.04, 0.0025, 0.05)),
                ]
            )
        )
        + "</body>"
    )
    book = f'<body name="book__0" pos="1.75 1.6 0.755"><freejoint/>{box("book_g", (0, 0, 0), (0.075, 0.11, 0.015), density=700)}</body>'
    chair = (
        f'<body name="chair__0" pos="1.5 0.62 0"><freejoint/>{legs("chair", 0.225, 0.225, 0.45)}'
        f"{box('chair_seat', (0, 0, 0.47), (0.225, 0.225, 0.02))}{box('chair_back', (0, -0.205, 0.71), (0.225, 0.02, 0.22))}</body>"
    )
    crate = f'<body name="crate__0" pos="0.6 2.4 0.2"><freejoint/>{box("crate_g", (0, 0, 0), (0.2, 0.2, 0.2), density=80)}</body>'
    vase = (
        '<body name="vase__0" pos="0.6 2.4 0.5"><freejoint/>'
        '<geom name="vase_g" type="cylinder" size="0.05 0.1" density="900"/></body>'
    )
    # cabinet: carcass fixed to the room, open toward -x; drawer in the top half, door on the bottom half
    cab = (
        '<body name="cabinet__0" pos="3.7 1.4 0">'
        + box("cab_bottom", (0, 0, 0.02), (0.25, 0.4, 0.02))
        + box("cab_top", (0, 0, 0.78), (0.25, 0.4, 0.02))
        + box("cab_left", (0, -0.38, 0.4), (0.25, 0.02, 0.36))
        + box("cab_right", (0, 0.38, 0.4), (0.25, 0.02, 0.36))
        + box("cab_back", (0.23, 0, 0.4), (0.02, 0.36, 0.36))
        + box("cab_mid", (0, 0, 0.42), (0.23, 0.36, 0.01))
        + '<body name="cabinet_drawer" pos="0 0 0.43">'
        + '<joint name="drawer_slide" type="slide" axis="1 0 0" range="-0.4 0" damping="5" frictionloss="0.3"/>'
        + box("drawer_bottom", (0, 0, 0.012), (0.2, 0.34, 0.01), density=600)
        + box("drawer_front", (-0.235, 0, 0.17), (0.01, 0.35, 0.155), density=600)
        + box("drawer_back", (0.19, 0, 0.12), (0.01, 0.32, 0.1), density=600)
        + box("drawer_l", (0, -0.33, 0.12), (0.2, 0.01, 0.1), density=600)
        + box("drawer_r", (0, 0.33, 0.12), (0.2, 0.01, 0.1), density=600)
        + '</body><body name="cabinet_door" pos="-0.262 -0.36 0.22">'
        + '<joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.6" damping="0.5" frictionloss="0.2"/>'
        + box("door_panel", (0, 0.36, 0), (0.01, 0.36, 0.18), density=600)
        + "</body></body>"
    )
    return f"""<mujoco model="sbv_test_room">
  <compiler angle="radian" autolimits="true"/>
  <default><geom friction="0.6 0.005 0.0001" density="600" rgba="0.7 0.6 0.5 1"/></default>
  <worldbody>
    <light pos="2 1.5 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.8 0.8 1"/>
    {walls}
    {table}{cup}{book}{chair}{crate}{vase}{cab}
  </worldbody>
</mujoco>
"""


def robot_room_xml():
    """5 m x 4 m with open floor: a table with a cup near its east edge, and a low cabinet 1.6 m away as the goal."""
    walls = "".join(
        [
            box("wall_w", (-0.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            box("wall_e", (5.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            box("wall_s", (2.5, -0.05, 1.25), (2.6, 0.05, 1.25)),
            box("wall_n", (2.5, 4.05, 1.25), (2.6, 0.05, 1.25)),
        ]
    )
    table = (
        f'<body name="table__0" pos="1.5 2.0 0"><freejoint/>{legs("table", 0.5, 0.4, 0.70)}'
        f"{box('table_top', (0, 0, 0.72), (0.5, 0.4, 0.02))}</body>"
    )
    cup = (
        '<body name="cup__0" pos="1.85 2.0 0.74"><freejoint/>'
        + box("cup_base", (0, 0, 0.0025), (0.03, 0.03, 0.0025), density=1500)
        + "".join(
            box(f"cup_w{i}", p, h, density=1500)
            for i, (p, h) in enumerate(
                [
                    ((0.0275, 0, 0.04), (0.0025, 0.03, 0.04)),
                    ((-0.0275, 0, 0.04), (0.0025, 0.03, 0.04)),
                    ((0, 0.0275, 0.04), (0.03, 0.0025, 0.04)),
                    ((0, -0.0275, 0.04), (0.03, 0.0025, 0.04)),
                ]
            )
        )
        + "</body>"
    )
    cab = f'<body name="cabinet__0" pos="3.6 2.0 0">{box("cab_g", (0, 0, 0.4), (0.3, 0.4, 0.4))}</body>'
    return f"""<mujoco model="sbv_robot_room">
  <compiler angle="radian" autolimits="true"/>
  <default><geom friction="0.8 0.005 0.0001" density="600" rgba="0.7 0.6 0.5 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.8 0.8 1"/>
    {walls}
    {table}{cup}{cab}
  </worldbody>
</mujoco>
"""


def cabinet_room_xml(knob=True):
    """The robot room (its table and cup) with a tall cabinet fixed against the east wall, open toward -x: a drawer in
    the top half with a bar handle, a door below with a knob (knob=False: a flush door, which no two-finger hand can
    pull)."""
    walls = "".join(
        [
            box("wall_w", (-0.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            box("wall_e", (5.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            box("wall_s", (2.5, -0.05, 1.25), (2.6, 0.05, 1.25)),
            box("wall_n", (2.5, 4.05, 1.25), (2.6, 0.05, 1.25)),
        ]
    )
    table = (
        f'<body name="table__0" pos="1.5 2.0 0"><freejoint/>{legs("table", 0.5, 0.4, 0.70)}'
        f"{box('table_top', (0, 0, 0.72), (0.5, 0.4, 0.02))}</body>"
    )
    cup = (
        '<body name="cup__0" pos="1.85 2.0 0.74"><freejoint/>'
        + box("cup_base", (0, 0, 0.0025), (0.03, 0.03, 0.0025), density=1500)
        + "".join(
            box(f"cup_w{i}", p, h, density=1500)
            for i, (p, h) in enumerate(
                [
                    ((0.0275, 0, 0.04), (0.0025, 0.03, 0.04)),
                    ((-0.0275, 0, 0.04), (0.0025, 0.03, 0.04)),
                    ((0, 0.0275, 0.04), (0.03, 0.0025, 0.04)),
                    ((0, -0.0275, 0.04), (0.03, 0.0025, 0.04)),
                ]
            )
        )
        + "</body>"
    )
    cab = (
        '<body name="cabinet__0" pos="4.7 2.0 0">'
        + box("cab_bottom", (0, 0, 0.02), (0.25, 0.4, 0.02))
        + box("cab_top", (0, 0, 1.18), (0.25, 0.4, 0.02))
        + box("cab_left", (0, -0.38, 0.6), (0.25, 0.02, 0.56))
        + box("cab_right", (0, 0.38, 0.6), (0.25, 0.02, 0.56))
        + box("cab_back", (0.23, 0, 0.6), (0.02, 0.36, 0.56))
        + box("cab_mid", (0, 0, 0.62), (0.23, 0.36, 0.01))
        + '<body name="cabinet_drawer" pos="0 0 0.63">'
        + '<joint name="drawer_slide" type="slide" axis="1 0 0" range="-0.4 0" damping="5" frictionloss="0.3"/>'
        + box("drawer_bottom", (0, 0, 0.012), (0.2, 0.34, 0.01), density=600)
        + box("drawer_front", (-0.235, 0, 0.27), (0.01, 0.35, 0.255), density=600)
        + box("drawer_back", (0.19, 0, 0.27), (0.01, 0.35, 0.255), density=600)
        + box("drawer_l", (0, -0.33, 0.12), (0.2, 0.01, 0.1), density=600)
        + box("drawer_r", (0, 0.33, 0.12), (0.2, 0.01, 0.1), density=600)
        + box(
            "drawer_handle", (-0.265, 0, 0.27), (0.01, 0.06, 0.008), density=2000
        )  # a bar 12 cm long, 2 cm proud of the front
        + '</body><body name="cabinet_door" pos="-0.262 -0.36 0.32">'
        + '<joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.6" damping="0.5" frictionloss="0.2"/>'
        + box("door_panel", (0, 0.36, 0), (0.01, 0.36, 0.28), density=600)
        + (
            '<geom name="door_knob_neck" type="cylinder" pos="-0.02 0.64 0.1" zaxis="1 0 0" size="0.006 0.01" density="2000"/>'
            '<geom name="door_knob" type="sphere" pos="-0.045 0.64 0.1" size="0.015" density="2000"/>'
            if knob
            else ""
        )
        + "</body></body>"
    )
    return f"""<mujoco model="sbv_cabinet_room">
  <compiler angle="radian" autolimits="true"/>
  <default><geom friction="0.8 0.005 0.0001" density="600" rgba="0.7 0.6 0.5 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.8 0.8 1"/>
    {walls}
    {table}{cup}{cab}
  </worldbody>
</mujoco>
"""


def write(directory, xml=None, functions=True):
    """The room and its functions.json: for the default room the cup's opening, so F3.2 checks it; for any other room
    an empty one (no container tested), since calibration stops for a room with a cup, crate or box and no file.
    functions=False writes no file, as a room that forgot it."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "scene.xml")
    with open(path, "w") as f:
        f.write(xml or room_xml())
    if functions:
        with open(os.path.join(directory, "functions.json"), "w") as f:
            f.write(
                '{"containers": {"cup__0": {"openings": [{"pos": [0, 0, 0.1], "dir": [0, 0, -1], "radius": 0.035}]}}}'
                if xml is None
                else '{"containers": {}}'
            )
    return path


if __name__ == "__main__":
    import sys

    print(write(sys.argv[1] if len(sys.argv) > 1 else "test_room"))
