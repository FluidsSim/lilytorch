"""Throwaway: FluidExtension that records the fluid kinetic-energy trace.

TwoPhaseSolver.finalize_step does not call the base diagnostics, so we compute
KE ourselves from the solver's current velocity after each fluid step.  Kept
import-light so the FARMS subprocess can load it by full package path.
"""
import os

from lilytorch.integration.extensions import FluidExtension

_OUT = "/data/andreaferrario/lilytorch/_submerged_diag"


class KEFluidExtension(FluidExtension):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ke = []
        self._frc = []          # (it, Fx, Fy, Fz) net hydro force (force_scaling=0)
        self._it = 0

    def initialize_episode(self, task, physics):
        super().initialize_episode(task, physics)
        # KEFLOW_GLIDE: SpawnMode.TRANSVERSE builds explicit slide/hinge ROOT
        # joints (a "special base"), so the spawn.velocity in the animat yaml is
        # ignored (mjcf.py only applies it to a free base).  Inject a constant
        # surge velocity directly on the root slide-x DOF (axis [1,0,0], the
        # FIRST root joint) here, AFTER FARMS has reset to the keyframe.  With
        # damping=0, force_scaling=0 (no fluid force on the body) and no vertical
        # DOF in TRANSVERSE, the body then COASTS at constant speed -> a steady
        # glide that exercises the convective flux but NOT the unsteady undulation.
        if os.environ.get("KEFLOW_GLIDE", "0") == "1":
            vx = float(os.environ.get("KEFLOW_VX", -0.13))
            try:
                names = [n for n in physics.named.data.qvel.axes.row.names
                         if "root0_j" in n]
                for n in names:
                    physics.named.data.qvel[n] = vx
                print(f"[KEFluidExtension] GLIDE: set qvel={vx} on {names}")
            except Exception as exc:
                print(f"[KEFluidExtension] GLIDE qvel inject failed: {exc}")

    def _net_force(self, fs):
        """Net hydro force (viscous + pressure) summed over bodies, per axis.
        Computed by the solver even at force_scaling=0 (only the APPLY is
        scaled), so this is the raw readout thrust."""
        import torch
        out = []
        for v, p in (("friction_force_lin_x", "pressure_force_x"),
                     ("friction_force_lin_y", "pressure_force_y"),
                     ("friction_force_lin_z", "pressure_force_z")):
            fv = getattr(fs, v, None); fp = getattr(fs, p, None)
            if fv is None or fp is None:
                out.append(float("nan")); continue
            out.append(float((fv + fp).sum()))
        return out

    def before_step(self, task, action, physics):
        super().before_step(task, action, physics)      # advances the fluid step
        try:
            fs = getattr(self.BDIMhandler, "fluid_solver", None)
            if fs is not None and getattr(fs, "u0", None) is not None:
                u, v = fs.u0, fs.v0
                w = getattr(fs, "w0", None)
                hd = getattr(fs, "h3", None) or getattr(fs, "h2", None)
                ke = (u * u + v * v)
                if w is not None:
                    ke = ke + w * w
                ek_tot = 0.5 * float(hd) * float(ke.sum())
                # FLUID-only KE: exclude the body interior + band (sdf<=eps),
                # so the imposed body-region velocity (identical across cases)
                # doesn't mask the fluid comparison.
                ek_fl = ek_tot
                comp = getattr(fs, "composite_body", None)
                sdf = getattr(comp, "sdf_val", None)
                if sdf is not None and sdf.shape == ke.shape:
                    eps = float(comp.bodies[0].eps)
                    fluid = (sdf > eps)
                    ek_fl = 0.5 * float(hd) * float((ke * fluid).sum())
                self._ke.append((self._it, ek_tot, ek_fl))
                try:
                    fx, fy, fz = self._net_force(fs)
                    self._frc.append((self._it, fx, fy, fz))
                except Exception:
                    pass
        except Exception as exc:
            if self._it == 0:
                print(f"[KEFluidExtension] KE compute failed: {exc}")
        self._it += 1

    def end_episode(self, task, physics):
        try:
            tag = os.environ.get("KEFLOW_TAG", "ke")
            path = os.path.join(_OUT, f"keflow_{tag}.csv")
            with open(path, "w") as fh:
                fh.write("it,ke_tot,ke_fluid\n")
                for i, kt, kf in self._ke:
                    fh.write(f"{i},{kt},{kf}\n")
            print(f"[KEFluidExtension] wrote {path} ({len(self._ke)} rows)")
            fpath = os.path.join(_OUT, f"force_{tag}.csv")
            with open(fpath, "w") as fh:
                fh.write("it,Fx,Fy,Fz\n")
                for i, fx, fy, fz in self._frc:
                    fh.write(f"{i},{fx},{fy},{fz}\n")
            print(f"[KEFluidExtension] wrote {fpath} ({len(self._frc)} rows)")
            # KEFLOW_DUMP_P=1: dump the mid-y xz pressure slice (+ SDF + z coords)
            # so we can subtract the hydrostatic reference and inspect the DYNAMIC
            # pressure structure around the body (is the around-body signal present
            # in two-phase, or only hidden under the hydrostatic colour scale?).
            if os.environ.get("KEFLOW_DUMP_P", "0") == "1":
                import numpy as np, torch
                fs = getattr(self.BDIMhandler, "fluid_solver", None)
                p = fs.p0; cb = fs.composite_body
                # pick the y-plane that cuts deepest through the body (min SDF),
                # not the geometric mid-plane (the eel spawns off-centre at y≈0.1).
                sdf_full = cb.sdf_val
                ny2 = int(torch.argmin(sdf_full.amin(dim=(0, 2))))
                sl = (slice(None), ny2, slice(None))
                np.savez(
                    os.path.join(_OUT, f"pdump_{tag}.npz"),
                    p=p[sl].detach().cpu().numpy(),
                    sdf=cb.sdf_val[sl].detach().cpu().numpy(),
                    h=float(fs.h), zmin=float(getattr(fs, "zmin", 0.0)),
                    rho=float(fs.rho),
                )
                print(f"[KEFluidExtension] wrote pdump_{tag}.npz")
        except Exception as exc:
            print(f"[KEFluidExtension] KE dump failed: {exc}")
        super().end_episode(task, physics)
