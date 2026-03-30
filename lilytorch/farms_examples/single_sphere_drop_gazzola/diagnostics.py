"""Diagnostic tools for Gazzola sphere sedimentation instability.

Logs key variables to identify causes of instability near bottom boundary.
"""

import numpy as np
import torch
import h5py
from pathlib import Path


class GazzolaDiagnostics:
    """Track and log critical variables during simulation."""

    def __init__(self, handler, log_path="gazzola_diagnostics.h5", log_every=10):
        """
        Args:
            handler: BDIMhandler instance
            log_path: where to save HDF5 diagnostics
            log_every: log every N iterations
        """
        self.handler = handler
        self.log_every = log_every
        self.iterations = []

        # Pressure field stats
        self.p_min = []
        self.p_max = []
        self.p_mean = []
        self.p_std = []
        self.p_rms = []

        # Velocity field stats
        self.u_min = []
        self.u_max = []
        self.u_rms = []
        self.v_min = []
        self.v_max = []
        self.v_rms = []
        self.max_vel = []

        # Divergence and CFL
        self.max_div = []
        self.cfl = []

        # Force magnitudes
        self.force_visc_x = []
        self.force_visc_y = []
        self.force_pres_x = []
        self.force_pres_y = []
        self.force_visc_total = []
        self.force_pres_total = []
        self.force_total = []

        # Body position and boundaries
        self.body_y_pos = []
        self.dist_to_bottom = []
        self.dist_to_top = []
        self.body_vel_z = []

        # SDF diagnostics
        self.sdf_min = []      # minimum SDF on domain
        self.sdf_near_body = []  # SDF value at body surface region
        self.sdf_body_count = []  # cells where sdf < eps (inside body)

        # Pressure near body
        self.p_at_body_mean = []
        self.p_at_body_max = []
        self.p_at_body_min = []

        # Stress tensor diagnostics
        self.stress_max = []
        self.pforce_max = []

        # Reynolds number
        self.reynolds = []

        # Stability metrics
        self.energy_kin = []
        self.enstrophy = []

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, iteration, u, v, p):
        """Log diagnostics at current iteration."""
        if iteration % self.log_every != 0:
            return

        fs = self.handler.fluid_solver
        comp = fs.composite_body
        dtype = fs.dtype
        device = fs.device
        h = fs.h.cpu().numpy()

        self.iterations.append(iteration)

        # === Pressure field ===
        p_np = p.cpu().numpy() if hasattr(p, 'cpu') else p
        self.p_min.append(float(np.nanmin(p_np)))
        self.p_max.append(float(np.nanmax(p_np)))
        self.p_mean.append(float(np.nanmean(p_np)))
        self.p_std.append(float(np.nanstd(p_np)))
        self.p_rms.append(float(np.sqrt(np.nanmean(p_np**2))))

        # === Velocity field ===
        u_np = u.cpu().numpy() if hasattr(u, 'cpu') else u
        v_np = v.cpu().numpy() if hasattr(v, 'cpu') else v
        self.u_min.append(float(np.nanmin(u_np)))
        self.u_max.append(float(np.nanmax(u_np)))
        self.u_rms.append(float(np.sqrt(np.nanmean(u_np**2))))
        self.v_min.append(float(np.nanmin(v_np)))
        self.v_max.append(float(np.nanmax(v_np)))
        self.v_rms.append(float(np.sqrt(np.nanmean(v_np**2))))
        max_vel = float(np.max(np.abs([u_np, v_np])))
        self.max_vel.append(max_vel)

        # === Divergence ===
        div = fs.divergence(u, v)
        max_div = float(torch.max(torch.abs(div)).cpu().numpy() if hasattr(div, 'cpu') else np.max(np.abs(div)))
        self.max_div.append(max_div)

        # === CFL ===
        cfl = float(max_vel * fs.dt.cpu().numpy() / h)
        self.cfl.append(cfl)

        # === Forces ===
        fv_x = float(fs.friction_force_lin_x[0].cpu().numpy() if hasattr(fs.friction_force_lin_x[0], 'cpu') else fs.friction_force_lin_x[0])
        fv_y = float(fs.friction_force_lin_y[0].cpu().numpy() if hasattr(fs.friction_force_lin_y[0], 'cpu') else fs.friction_force_lin_y[0])
        fp_x = float(fs.pressure_force_x[0].cpu().numpy() if hasattr(fs.pressure_force_x[0], 'cpu') else fs.pressure_force_x[0])
        fp_y = float(fs.pressure_force_y[0].cpu().numpy() if hasattr(fs.pressure_force_y[0], 'cpu') else fs.pressure_force_y[0])

        self.force_visc_x.append(fv_x)
        self.force_visc_y.append(fv_y)
        self.force_pres_x.append(fp_x)
        self.force_pres_y.append(fp_y)
        self.force_visc_total.append(float(np.sqrt(fv_x**2 + fv_y**2)))
        self.force_pres_total.append(float(np.sqrt(fp_x**2 + fp_y**2)))
        self.force_total.append(float(np.sqrt((fv_x+fp_x)**2 + (fv_y+fp_y)**2)))

        # === Body position ===
        com_y = float(comp.com_pos[0, 1].cpu().numpy() if hasattr(comp.com_pos[0, 1], 'cpu') else comp.com_pos[0, 1])
        ymin = fs.y[0]
        ymax = fs.y[-1]
        self.body_y_pos.append(com_y)
        self.dist_to_bottom.append(float(com_y - ymin))
        self.dist_to_top.append(float(ymax - com_y))

        # === Body velocity ===
        try:
            vel_z = float(self.handler.data[0].sensors.links.com_lin_velocities()[iteration, 0][2])
            self.body_vel_z.append(vel_z)
        except:
            self.body_vel_z.append(np.nan)

        # === SDF diagnostics ===
        sdf_np = comp.sdf_val.cpu().numpy() if hasattr(comp.sdf_val, 'cpu') else comp.sdf_val
        self.sdf_min.append(float(np.nanmin(sdf_np)))

        eps_body = comp.bodies[0].eps.cpu().numpy()
        sdf_near_body_mask = (sdf_np >= -2*eps_body) & (sdf_np <= 2*eps_body)
        if np.any(sdf_near_body_mask):
            self.sdf_near_body.append(float(np.nanmean(np.abs(sdf_np[sdf_near_body_mask]))))
        else:
            self.sdf_near_body.append(np.nan)

        sdf_inside = np.sum(sdf_np < 0)
        self.sdf_body_count.append(int(sdf_inside))

        # === Pressure at body ===
        p_at_body = p_np.copy()
        p_at_body[sdf_np > 0] = np.nan
        self.p_at_body_min.append(float(np.nanmin(p_at_body)))
        self.p_at_body_max.append(float(np.nanmax(p_at_body)))
        self.p_at_body_mean.append(float(np.nanmean(p_at_body)))

        # === Stress diagnostics ===
        if hasattr(fs, 'xstress_tensor'):
            stress_np = fs.xstress_tensor.cpu().numpy() if hasattr(fs.xstress_tensor, 'cpu') else fs.xstress_tensor
            self.stress_max.append(float(np.max(np.abs(stress_np))))
        else:
            self.stress_max.append(np.nan)

        if hasattr(fs, 'pforce_x'):
            pf = np.sqrt(fs.pforce_x.cpu().numpy()**2 + fs.pforce_y.cpu().numpy()**2) if hasattr(fs.pforce_x, 'cpu') else np.sqrt(fs.pforce_x**2 + fs.pforce_y**2)
            self.pforce_max.append(float(np.max(np.abs(pf))))
        else:
            self.pforce_max.append(np.nan)

        # === Reynolds number ===
        try:
            if len(self.body_vel_z) > 0 and not np.isnan(self.body_vel_z[-1]):
                vel_z = abs(self.body_vel_z[-1])
                radius = 0.0025
                nu = self.handler.pars['solver']['nu']
                re = vel_z * 2 * radius / nu
                self.reynolds.append(re)
            else:
                self.reynolds.append(np.nan)
        except:
            self.reynolds.append(np.nan)

        # === Kinetic energy and enstrophy ===
        u_no_nan = np.nan_to_num(u_np, 0)
        v_no_nan = np.nan_to_num(v_np, 0)
        ke = 0.5 * float(np.mean(u_no_nan**2 + v_no_nan**2))
        self.energy_kin.append(ke)

        # Simple enstrophy: |curl(u)|^2
        dvdx = np.gradient(v_no_nan, axis=0) / h
        dudy = np.gradient(u_no_nan, axis=1) / h
        omega = dvdx - dudy
        enstr = float(np.mean(omega**2))
        self.enstrophy.append(enstr)

    def save(self):
        """Save all diagnostics to HDF5."""
        with h5py.File(self.log_path, 'w') as f:
            f.create_dataset('iterations', data=np.array(self.iterations))

            f.create_group('pressure')
            f['pressure/min'] = np.array(self.p_min)
            f['pressure/max'] = np.array(self.p_max)
            f['pressure/mean'] = np.array(self.p_mean)
            f['pressure/std'] = np.array(self.p_std)
            f['pressure/rms'] = np.array(self.p_rms)

            f.create_group('velocity')
            f['velocity/u_min'] = np.array(self.u_min)
            f['velocity/u_max'] = np.array(self.u_max)
            f['velocity/u_rms'] = np.array(self.u_rms)
            f['velocity/v_min'] = np.array(self.v_min)
            f['velocity/v_max'] = np.array(self.v_max)
            f['velocity/v_rms'] = np.array(self.v_rms)
            f['velocity/max_vel'] = np.array(self.max_vel)

            f.create_group('solvers')
            f['solvers/max_divergence'] = np.array(self.max_div)
            f['solvers/cfl'] = np.array(self.cfl)

            f.create_group('forces')
            f['forces/visc_x'] = np.array(self.force_visc_x)
            f['forces/visc_y'] = np.array(self.force_visc_y)
            f['forces/pres_x'] = np.array(self.force_pres_x)
            f['forces/pres_y'] = np.array(self.force_pres_y)
            f['forces/visc_total'] = np.array(self.force_visc_total)
            f['forces/pres_total'] = np.array(self.force_pres_total)
            f['forces/total'] = np.array(self.force_total)

            f.create_group('body')
            f['body/y_position'] = np.array(self.body_y_pos)
            f['body/dist_to_bottom'] = np.array(self.dist_to_bottom)
            f['body/dist_to_top'] = np.array(self.dist_to_top)
            f['body/velocity_z'] = np.array(self.body_vel_z)
            f['body/reynolds'] = np.array(self.reynolds)

            f.create_group('sdf')
            f['sdf/min'] = np.array(self.sdf_min)
            f['sdf/near_body'] = np.array(self.sdf_near_body)
            f['sdf/body_cell_count'] = np.array(self.sdf_body_count)

            f.create_group('pressure_at_body')
            f['pressure_at_body/min'] = np.array(self.p_at_body_min)
            f['pressure_at_body/max'] = np.array(self.p_at_body_max)
            f['pressure_at_body/mean'] = np.array(self.p_at_body_mean)

            f.create_group('stress')
            f['stress/max'] = np.array(self.stress_max)
            f['stress/pforce_max'] = np.array(self.pforce_max)

            f.create_group('stability')
            f['stability/kinetic_energy'] = np.array(self.energy_kin)
            f['stability/enstrophy'] = np.array(self.enstrophy)

        print(f"[DIAG] Saved diagnostics to {self.log_path}")

    def print_summary(self, iteration):
        """Print diagnostic summary for current iteration."""
        if iteration % (self.log_every * 10) != 0:
            return

        idx = len(self.iterations) - 1
        if idx < 0:
            return

        print(f"\n[DIAG @it={iteration}]")
        print(f"  Pressure:     min={self.p_min[idx]:.3e}  max={self.p_max[idx]:.3e}  rms={self.p_rms[idx]:.3e}")
        print(f"  Velocity:     max={self.max_vel[idx]:.3e}  CFL={self.cfl[idx]:.3f}  div_max={self.max_div[idx]:.3e}")
        print(f"  Forces:       visc={self.force_visc_total[idx]:.3e}  pres={self.force_pres_total[idx]:.3e}  total={self.force_total[idx]:.3e}")
        print(f"  Body:         y={self.body_y_pos[idx]:.4f}  dist_bottom={self.dist_to_bottom[idx]:.5f}  v_z={self.body_vel_z[idx]:.3e}")
        print(f"  SDF:          min={self.sdf_min[idx]:.3e}  near_body={self.sdf_near_body[idx]:.3e}  cells={self.sdf_body_count[idx]}")
        print(f"  Stability:    KE={self.energy_kin[idx]:.3e}  enstrophy={self.enstrophy[idx]:.3e}  Re={self.reynolds[idx]:.1f}")
