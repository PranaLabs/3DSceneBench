"""Speed harness for a room's robot verification.

- profile: where a room's robot calibration time goes, cheaply (a sampled whole-room settle, one nominal trial per
  task under the engine's speed modes, each task's zone settle, collision complexity, a modelled calibration cost).
- layer1: the engine's speed modes (the settle cache, SBV_ROBOT_SETTLE_CACHE and SBV_GRAVITY_CACHE, and frozen robot
  trials, SBV_ROBOT_FREEZE / params.freeze) against the plain engine on the room and its copies; a mode with any
  mismatch is switched off for that room, with the reason.

Hard rules: timestep, solver and integrator are never changed; no verdict changes; nothing here edits a room.
CLI: python -m sbv.speed {profile,layer1} ... (sbv/speed/__main__.py).
"""
