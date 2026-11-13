"""Incremental authoring self-checks. `pipeline.score` and Harbor grading stay live."""

import copy
import json
import os
from pathlib import Path

from sbv.common import case as K
from sbv.common.dependencies import model_files, sha, runtime, environment, PHYSICS_ENV, SCOPES
from sbv.common.scene_inputs import physical
from sbv.common.code_dependencies import native
from sbv.common.phase_cache import cached
from sbv.appearance import views as A


def appearance_identity(case):
    from sbv.appearance.cache_identity import identity
    from sbv.common.dependencies import render_identity

    weights = identity(A.TORCH_PY)
    if weights is None:
        return None
    return dict(
        renderer=render_identity(),
        metrics=weights,
        code=native("appearance"),
        model=model_files(K.reference_path(case)),
        appearance={k: v for k, v in case["appearance"].items() if k != "floor"},
        bounds=case["room"]["bounds"],
        physics=case["physics"],
        metric_device=os.environ.get("SBV_METRIC_DEVICE", "cpu"),
    )


def calibration(room, out, options, procs, log, resume=True):
    from sbv import calibrate
    from sbv.common.calibration_inputs import rigid_identity

    def identity():
        defaults = dict(
            draws=10,
            need=9,
            jitter=3,
            nav_start=None,
            robot_tasks=4,
            push_tasks=2,
            isolate_robot=True,
            tasks=None,
            require_tasks=True,
            mounts=None,
            functions=None,
            camera_seed=None,
            local_m=calibrate.LOCAL_M,
            small_models=True,
            isolate=True,
        )
        defaults.update(
            {k: v for k, v in options.items() if k not in ("checkpoint", "job_timeout", "resume", "tolerances")}
        )
        return dict(
            physical=rigid_identity(room, defaults),
            code=native("calibration_case"),
            cameras={
                n: sha(Path(room) / n)
                for n in ("input_cameras.json", "heldout_cameras.json", "extra_cameras.json")
                if (Path(room) / n).is_file()
            },
        )

    def compute():
        return calibrate.build(room, procs, log=log, **options)

    result = cached(out, "calibration", identity, compute, resume=resume)
    calibrate.resources(room, options.get("checkpoint"), resume)
    # Measurement semantics are unchanged; bind the current staging path and raw resources.
    result = copy.deepcopy(result)
    xml = K.scene_xml(os.path.abspath(room))
    result["reference"] = xml
    result["reference_sha256"] = sha(xml)
    result["reference_files"] = model_files(xml)
    return result


def score(case_path, room, out, procs, log, resume=True):
    from sbv import evaluate, pipeline as P
    from sbv.nonrigid import verify as Nonrigid
    from sbv.package_tasks import verify as Package

    case = K.load(case_path)
    out = Path(out)

    def rigid_key():
        c = {
            k: v
            for k, v in case.items()
            if k not in ("case_hash", "_dir", "reference", "reference_sha256", "reference_files", "appearance")
        }
        c = copy.deepcopy(c)
        c.get("calibration", {}).pop("seconds", None)
        return dict(
            model=physical(room),
            case=c,
            code=native("rigid_score"),
            runtime=runtime(packages=SCOPES["rigid"]),
            environment=environment(PHYSICS_ENV),
        )

    def rigid():
        r = evaluate.evaluate(case, room, procs, log=log)
        if r.get("status") == "grader_failed":
            raise RuntimeError(r.get("failure"))
        return r

    r = cached(out, "self-rigid", rigid_key, rigid, resume=resume)
    r = copy.deepcopy(r)
    r["case_hash"] = case["case_hash"]
    r["submission"] = K.scene_xml(os.path.abspath(room))
    if r.get("geometry") is not None:

        def appearance():
            from sbv.common.scene import Scene

            scene = Scene(r["submission"], case["physics"], textures=False)
            scene.reset(scene.q0)
            near = A.near_cameras(
                case["appearance"]["cameras"], scene, true=A.ray_depths(case["appearance"]["cameras"], scene)
            )
            return dict(A.evaluate(case, r["submission"], str(out / "self_renders"), scene=scene), near_camera=near)

        app = cached(
            out,
            "self-appearance",
            lambda: appearance_identity(case),
            appearance,
            ["self_renders"],
            accept=lambda a: all((a.get("mean") or {}).get(k) is not None for k in A.METRICS),
            resume=resume,
        )
        r["appearance"] = dict(app, floor=case["appearance"].get("floor"))
    P.write(str(out / "self.json"), r)
    nonrigid = None
    package = None
    side = out / "nonrigid_case.json"
    if side.is_file():

        def nonrigid_key():
            return dict(
                case=json.loads(side.read_text()),
                model=physical(room),
                code=native("nonrigid"),
                inputs=Nonrigid.reference_inputs(
                    room, [u["truth_object"] for u in json.loads(side.read_text())["units"]]
                ),
                runtime=runtime(Nonrigid.PYTHON, packages=SCOPES["nonrigid"]),
                environment=environment(PHYSICS_ENV),
            )

        (out / "self_nonrigid").mkdir(exist_ok=True)
        nonrigid = cached(
            out,
            "self-nonrigid",
            nonrigid_key,
            lambda: Nonrigid.evaluate(
                str(side), room, str(out / "self_nonrigid/nonrigid_report.json"), max(1, procs // 4), qualification=True
            ),
            ["self_nonrigid"],
            resume=resume,
        )
    if (out / "package_case.json").is_file():
        (out / "self_package").mkdir(exist_ok=True)
        package = Package.evaluate(
            str(out / "package_case.json"),
            room,
            str(out / "self.json"),
            str(out / "self_package/package_report.json"),
            max(1, procs // 2),
            qualification=resume,
        )
    return P.scores(r, nonrigid, package)


def floor(case, out, log, resume=True):
    return cached(out, "appearance-floor", lambda: appearance_identity(case), lambda: A.floor(case, log), resume=resume)


def nonrigid_calibration(room, out, procs, resume=True):
    from sbv.nonrigid import verify as Nonrigid

    def key():
        objs, bad = Nonrigid.load(Nonrigid.side_file(room))
        if bad:
            raise ValueError(bad)
        return dict(
            inputs=Nonrigid.reference_inputs(room, objs),
            code=native("nonrigid_calibration"),
            runtime=runtime(Nonrigid.PYTHON, packages=SCOPES["nonrigid"]),
            environment=environment(PHYSICS_ENV),
            threads=Nonrigid.THREADS,
            video=Nonrigid.VIDEO,
        )

    def compute():
        Nonrigid.calibrate(room, str(Path(out) / "nonrigid_case.json"), procs)
        return json.loads((Path(out) / "nonrigid_case.json").read_text())

    result = cached(out, "nonrigid-calibration", key, compute, ["nonrigid_case.json", "nonrigid_cal"], resume=resume)
    result["room"] = os.path.relpath(room, out)
    (Path(out) / "nonrigid_case.json").write_text(json.dumps(result, indent=1))
    return result


def authoring(fn):
    """Scope private caches to this check; --no-resume bypasses all child reuse."""
    from functools import wraps
    import inspect
    from sbv.common.dependencies import session, current_session

    @wraps(fn)
    def run(*args, **kwargs):
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        out = Path(bound.arguments["out"])
        resume = bound.arguments["resume"]
        old = {name: os.environ.get(name) for name in ("SBV_PACKAGE_CACHE", "SBV_NONRIGID_CACHE")}
        for name, sub in (("SBV_PACKAGE_CACHE", "package"), ("SBV_NONRIGID_CACHE", "nonrigid")):
            if old[name] is None:
                os.environ[name] = str(out.resolve() / ".trial-cache" / sub)
        try:
            with session():
                state = current_session()
                previous = state["reuse"]
                state["reuse"] = resume and previous
                try:
                    return fn(*args, **kwargs)
                finally:
                    state["reuse"] = previous
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return run
