"""Two fixes to Genesis's PBD cloth: friction with rigid bodies, and bending about its starting shape (see _BEND_OLD).

Genesis 1.4.3 resolves a cloth particle that touches a rigid body by pushing it out along the contact normal, and
nothing else: its own code leaves friction out ("we don't consider friction for now"). Cloth then slides off anything
that is not level: a towel dropped over a beaker slid off onto the bench, and a folded cloth slowly opened, with nothing
to hold its lower layer. cloth_rigid_friction() adds Coulomb friction to that contact, at velocity level: the particle's
velocity along the surface, relative to the body's own motion there, is cancelled while it is under mu_s times the
normal speed the contact removes (the approach speed), else reduced by mu_k times that. (Measured by push-out depth
alone, as Genesis does between cloth particles, the friction was about a hundred times too weak here: a cloth slid down
a 20 degree slope with mu 0.6.) The coefficients are the cloth material's own static and kinetic friction.

Run it before the scene is built (Genesis compiles its kernels at build). It checks that Genesis's code is the code it
was written against and refuses otherwise.
"""

import inspect
import linecache
import textwrap

_DONE = False
_OLD = """            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt
"""
_NEW = """            depth = self.pbd_solver.particle_size / 2 - signed_dist
            new_pos = pos_world + stiffness * contact_normal * depth
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            dt_sub = self.pbd_solver._substep_dt
            new_vel = (new_pos - prev_pos) / dt_sub
            vel_rigid = self.rigid_solver._func_vel_at_point(
                pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx], i_b=batch_idx, links_state=links_state
            )
            # Coulomb friction at velocity level: the normal impulse is the approach speed the contact removes (for a
            # cloth at rest, what gravity gave it this substep); the tangential velocity relative to the body loses up
            # to mu times that. The push-out speed is not counted: a particle resting a hair inside the contact band
            # would otherwise hold on any slope.
            dv_n = qd.max(-(vel - vel_rigid).dot(contact_normal), 0.0)
            rel = new_vel - vel_rigid
            rel_t = rel - contact_normal * rel.dot(contact_normal)
            rel_t_n = rel_t.norm(gs.EPS)
            mu_s = self.pbd_solver.particles_info_reordered[i, batch_idx].mu_s
            mu_k = self.pbd_solver.particles_info_reordered[i, batch_idx].mu_k
            if rel_t_n <= mu_s * dv_n:
                new_vel = new_vel - rel_t
            else:
                new_vel = new_vel - rel_t * qd.min(1.0, mu_k * dv_n / rel_t_n)
            new_pos = prev_pos + new_vel * dt_sub
"""


def _replace(module, cls, name, old, new, label, kernel=False):
    """Recompile cls.name from Genesis's own source with old replaced by new; refuse if old is not there. A kernel is
    compiled inside a class statement: that is how quadrants tells a method's self from a kernel argument."""
    f = getattr(cls, name)
    src = inspect.getsource(f.fn)  # as indented in Genesis's file, as old is
    if old not in src:
        raise RuntimeError(f"Genesis's {label} changed; re-check sbv/nonrigid/genesis_patch.py")
    new = textwrap.dedent(src.replace(old, new))
    if kernel:
        new = "class _Patched:\n" + textwrap.indent(new, "    ")
    # Genesis's kernel compiler reads a function's source back by file name
    fname = f"<sbv-genesis-{label.replace(' ', '-')}>"
    linecache.cache[fname] = (len(new), None, new.splitlines(True), fname)
    ns = {}
    exec(compile(new, fname, "exec"), module.__dict__, ns)
    setattr(cls, name, ns["_Patched"].__dict__[name] if kernel else ns[name])


# Bending. Genesis 1.4.3 bends every cloth toward flat: its constraint is the angle between two neighbouring triangles
# minus 180 degrees, whatever the shape the cloth starts in. The paper its code follows (Mueller et al., Position Based
# Dynamics, AGEIA, VRIPHYS 2006) takes "the initial dihedral angle between the two triangles" instead. A garment is
# curved, so Genesis's sprang open at its start. A flat strip starts flat, so the bending lengths measured on one
# (schema.FABRIC) are the same either way.
_BEND_OLD = """                    constraint = qd.acos(d) - qd.acos(-1.0)
"""
_BEND_NEW = """                    r2 = self.particles_info[v2].pos_rest - self.particles_info[v1].pos_rest
                    r3 = self.particles_info[v3].pos_rest - self.particles_info[v1].pos_rest
                    r4 = self.particles_info[v4].pos_rest - self.particles_info[v1].pos_rest
                    d0 = qd.math.clamp((r2.cross(r3) / r2.cross(r3).norm()).dot(r2.cross(r4) / r2.cross(r4).norm()), -1.0, 1.0)   # no EPS: it is larger than these squared norms
                    constraint = qd.acos(d) - qd.acos(d0)
"""


def cloth_rigid_friction():
    """Coulomb friction between cloth and rigid bodies, and bending about the cloth's starting shape."""
    global _DONE
    if _DONE:
        return
    from genesis.engine.couplers import legacy_coupler as LC
    from genesis.engine.solvers import pbd_solver as PS

    _replace(LC, LC.LegacyCoupler, "_func_pbd_collide_with_rigid_geom", _OLD, _NEW, "cloth-rigid contact")
    _replace(PS, PS.PBDSolver, "_kernel_solve_bending", _BEND_OLD, _BEND_NEW, "cloth bending", kernel=True)
    _DONE = True
