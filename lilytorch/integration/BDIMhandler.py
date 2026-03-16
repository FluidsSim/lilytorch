"""
Unified BDIM handler for FARMS <-> lilytorch coupling.

Supports both 2-D and 3-D simulations.  Dimensionality is auto-detected
from the presence of ``Nz`` in ``pars['solver']``.

All simulation-specific hyperparameters are read from the ``bdim_yaml``
config dict, so a single class covers every animat (1guilla, pleurodeles,
zebrafish, salamander, ...).

New config keys (add to ``bdim_yaml``):
    solver.dtype                 : "float32" | "float64"   (default "float32")
    solver.rho_body              : float                   (default 800.0)
    solver.zero_pressure_inside  : bool                    (default False)
    solver.force_method          : "method1" | "method2"   (default "method2";
                                   3-D always uses method2_3d)
    body.force_scaling           : "auto" | float          (default "auto")
    body.contour_mask            : bool                    (default False)
    physics.solref               : [float, float] | null   (default null)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from lilytorch.src.solver import FluidSolver
from lilytorch.src.body import (rotate_grid_2d, rotate_grid_3d,
                                _rotate_grid_3d_compiled,
                                _stagger_sdf_3d, _stagger_sdf_3d_compiled)
import torch


class BDIMhandler:
    """Unified boundary-data immersion method handler (2-D and 3-D)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, yaml_file, data, physics, dtype=None):

        self.pars = yaml_file

        # ---- auto-detect dimensionality ----
        self.ndim = 3 if "Nz" in self.pars["solver"] else 2

        # ---- dtype ----
        if dtype is not None:
            self.dtype = dtype
        else:
            dtype_str = self.pars["solver"].get("dtype", "float32")
            self.dtype = torch.float32 if dtype_str == "float32" else torch.float64
        self.dtype_np = np.float32 if self.dtype == torch.float32 else np.float64

        # ---- bookkeeping ----
        self.data = data          # list[AnimatData] from FARMS
        self.iteration = 0
        self.terminate = False

        # ---- create fluid solver ----
        self.fluid_solver = FluidSolver(
            self.pars,
            dtype=self.dtype,
            custom_update=True,
            compute_forces=True,
        )
        self.device = self.fluid_solver.device

        # override composite-body update with our FARMS-driven version
        self.fluid_solver.composite_body.update = self.update

        # ---- initialise u0 from inlet BC ----
        u_inlet = self.fluid_solver.adv_diff_solver.BC_values_u[1]
        if self.ndim == 3:
            gs = self.fluid_solver.grid_shape
            self.fluid_solver.u0 = u_inlet * torch.ones(
                gs, device=self.device, dtype=self.dtype
            )
        else:
            self.fluid_solver.u0 = u_inlet * torch.ones(
                (self.fluid_solver.nx, self.fluid_solver.ny),
                device=self.device, dtype=self.dtype,
            )

        # ---- axes (derived from ndim) ----
        if self.ndim == 3:
            self.lin_axes = [0, 1, 2]
            self.ang_axes = [0, 1, 2]
        else:
            self.lin_axes = [0, 1]
            self.ang_axes = [2]

        # ---- force scaling (config or auto) ----
        fs_cfg = self.pars.get("body", {}).get("force_scaling", "auto")
        if fs_cfg == "auto":
            if self.ndim == 2:
                comp = self.fluid_solver.composite_body
                try:
                    self.force_scaling = torch.tensor(
                        [np.diff(body.bb[2])[0] for body in comp.bodies],
                        device=self.device, dtype=self.dtype
                    )
                except Exception:
                    self.force_scaling = 1.0
            else:
                self.force_scaling = 1.0
        else:
            self.force_scaling = float(fs_cfg)

        # ---- densities ----
        self.rho_fluid = self.pars["solver"]["rho"]
        self.rho_body  = self.pars["solver"].get("rho_body", 800.0)

        # ---- toggles ----
        self.zero_pressure_inside = self.pars["solver"].get(
            "zero_pressure_inside", False
        )
        self.contour_mask = self.pars.get("body", {}).get("contour_mask", False)
        self.force_method = self.pars["solver"].get("force_method", "method2")

        # ---- FARMS-style buoyancy parameters ----
        # With all-Neumann BCs the BDIM pressure field is purely dynamic;
        # no hydrostatic gradient builds up, so buoyancy must be added
        # explicitly.  We replicate FARMS' compute_buoyancy() formula from
        # drag.pyx which uses MuJoCo body mass, bounding-sphere half-height,
        # and a linear submersion fraction.
        self.gravity_z = float(physics.model.opt.gravity[2])  # e.g. -9.81
        self.water_surface = float(self.pars["solver"]["zmax"]) if "zmax" in self.pars["solver"] else 0.0
        self._buoyancy_initialized = False

        # ---- optional physics solref tweak ----
        solref = self.pars.get("physics", {}).get("solref", None)
        if solref is not None:
            physics.model.geom_solref[:, 0] = solref[0]
            physics.model.geom_solref[:, 1] = solref[1]

        # ---- allocate per-body SDF / velocity arrays if missing ----
        comp = self.fluid_solver.composite_body
        gs = self.fluid_solver.grid_shape
        if self.ndim == 3:
            comp.sdf_vals = torch.zeros(
                (comp.nbodies, *gs), device=self.device, dtype=self.dtype,
            )
        else:
            # 2-D update uses per-body stacks for argmin-based union
            if not hasattr(comp, 'sdf_vals'):
                comp.sdf_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'sdf_vals_u'):
                comp.sdf_vals_u = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'sdf_vals_v'):
                comp.sdf_vals_v = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'u_vals'):
                comp.u_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)
            if not hasattr(comp, 'v_vals'):
                comp.v_vals = torch.zeros(
                    (comp.nbodies, *gs), device=self.device, dtype=self.dtype)

            # Pre-build batched SDF tensor for grid_sample (2-D only)
            self._init_batched_sdf_2d()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _init_buoyancy_params(self, task, physics):
        """Precompute per-body mass & half-height for FARMS-style buoyancy.

        Called lazily on the first ``_apply_forces_3d`` invocation because
        ``task`` (needed for the body-index mapping) is not available at
        ``__init__`` time.

        FARMS formula (from drag.pyx ``compute_buoyancy``):
            if pos_z - height < surface:
                F_z = -rho_water * mass * gravity / density
                      * min((surface + height - pos_z) / (2*height), 1)
        where
            mass   = MuJoCo body_mass            [kg]
            density = link density from config    [kg/m³]
            height  = 0.5 * geom bounding radius  [m]
            surface = water surface height (zmax)  [m]
            gravity = -9.81                        [m/s²]
        """
        comp = self.fluid_solver.composite_body
        n = comp.nbodies

        self._buoy_mass   = np.zeros(n)
        self._buoy_height = np.zeros(n)

        for body_i in range(n):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            self._buoy_mass[body_i] = float(physics.model.body_mass[ind])

            # Find the maximum bounding-sphere radius among geoms attached
            # to this MuJoCo body (same logic as FARMS SwimmingHandler).
            max_rbound = 0.0
            for gi in range(physics.model.ngeom):
                if int(physics.model.geom_bodyid[gi]) == ind:
                    rb = float(physics.model.geom_rbound[gi])
                    if rb > max_rbound:
                        max_rbound = rb
            self._buoy_height[body_i] = 0.5 * max_rbound

        self._buoyancy_initialized = True

    def cython2numpy(self, array):
        return torch.from_numpy(
            np.array(array).astype(self.dtype_np)
        ).to(self.device)

    # ==================================================================
    #  update: FARMS kinematics  ->  SDF fields + body velocities
    # ==================================================================
    def update(self, t, iteration, dt=1):
        if self.ndim == 3:
            self._update_3d(t, iteration, dt)
        else:
            self._update_2d(t, iteration, dt)

    # ---- 2-D update --------------------------------------------------
    def _update_2d(self, t, iteration, dt=1):
        fs   = self.fluid_solver
        comp = fs.composite_body
        B    = comp.nbodies

        _FAR = 1e4   # far-field SDF sentinel (same as 3-D path)

        # gather per-animat kinematics
        com_poses  = []
        urdf_poses = []
        Rs         = []
        lin_vels   = []
        ang_vels   = []
        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses.append(
                self.cython2numpy(sen.com_positions()[iteration, :])[:, self.lin_axes]
            )
            urdf_poses.append(
                self.cython2numpy(sen.urdf_positions()[iteration, :])[:, self.lin_axes]
            )
            Rs.append(
                self.cython2numpy(
                    Rotation.from_quat(
                        sen.urdf_orientations()[iteration, :]
                    ).as_matrix().astype(self.dtype_np)
                )[:, self.lin_axes, :][:, :, self.lin_axes]
            )
            lin_vels.append(
                self.cython2numpy(sen.com_lin_velocities()[iteration, :])[:, self.lin_axes]
            )
            ang_vels.append(
                self.cython2numpy(
                    [sen.com_ang_velocity(iteration, lk)[2]
                     for lk in range(len(sen.names))]
                )
            )

        # ── Stack per-body kinematics into batched GPU tensors ────
        urdf_t = torch.stack([urdf_poses[aid][lid] for aid, lid in comp.body_ids])  # (B, 2)
        com_t  = torch.stack([com_poses[aid][lid]  for aid, lid in comp.body_ids])   # (B, 2)
        R_t    = torch.stack([Rs[aid][lid]         for aid, lid in comp.body_ids])    # (B, 2, 2)
        lv_t   = torch.stack([lin_vels[aid][lid]   for aid, lid in comp.body_ids])   # (B, 2)
        av_t   = torch.stack([ang_vels[aid][lid]   for aid, lid in comp.body_ids])   # (B,)
        R_T_t  = R_t.transpose(1, 2)  # (B, 2, 2)

        # ── Batched rotation + grid_sample for CC grid ────────────
        dx = comp.X.unsqueeze(0) - urdf_t[:, 0, None, None]   # (B, nx, ny)
        dy = comp.Y.unsqueeze(0) - urdf_t[:, 1, None, None]   # (B, nx, ny)
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        # grid_sample grid: [..., 0] → W (y axis), [..., 1] → H (x axis)
        sdf_cc_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched rotation + grid_sample for u-staggered grid ───
        dx = comp.Xu_stag.unsqueeze(0) - urdf_t[:, 0, None, None]
        dy = comp.Yu_stag.unsqueeze(0) - urdf_t[:, 1, None, None]
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        sdf_u_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched rotation + grid_sample for v-staggered grid ───
        dx = comp.Xv_stag.unsqueeze(0) - urdf_t[:, 0, None, None]
        dy = comp.Yv_stag.unsqueeze(0) - urdf_t[:, 1, None, None]
        px = R_T_t[:, 0, 0, None, None] * dx + R_T_t[:, 0, 1, None, None] * dy
        py = R_T_t[:, 1, 0, None, None] * dx + R_T_t[:, 1, 1, None, None] * dy
        x_n = px * self._sdf_x_scale[:, None, None] + self._sdf_x_offset[:, None, None]
        y_n = py * self._sdf_y_scale[:, None, None] + self._sdf_y_offset[:, None, None]
        sdf_v_all = torch.nn.functional.grid_sample(
            self._sdf_batch, torch.stack([y_n, x_n], dim=-1),
            mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(1)                                              # (B, nx, ny)

        # ── Batched body velocities ───────────────────────────────
        vel_u_all = lv_t[:, 0, None, None] - av_t[:, None, None] * (
            comp.Yu_stag.unsqueeze(0) - com_t[:, 1, None, None])  # (B, nx, ny)
        vel_v_all = lv_t[:, 1, None, None] + av_t[:, None, None] * (
            comp.Xv_stag.unsqueeze(0) - com_t[:, 0, None, None])  # (B, nx, ny)

        # ── Batched union-min ─────────────────────────────────────
        comp.sdf_vals[:]   = sdf_cc_all
        comp.sdf_vals_u[:] = sdf_u_all
        comp.sdf_vals_v[:] = sdf_v_all
        comp.sdf_val       = sdf_cc_all.min(dim=0).values

        min_u = sdf_u_all.min(dim=0)
        comp.sdf_val_u = min_u.values
        comp.body_u    = vel_u_all.gather(0, min_u.indices.unsqueeze(0)).squeeze(0)

        min_v = sdf_v_all.min(dim=0)
        comp.sdf_val_v = min_v.values
        comp.body_v    = vel_v_all.gather(0, min_v.indices.unsqueeze(0)).squeeze(0)

        # ── Per-body: contour update + contour mask ───────────────
        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            comp.com_pos[body_i] = com_t[body_i]

            # contour update
            body.cnt_update = R_t[body_i] @ body.cnt + urdf_t[body_i, :, None]

            # optional contour mask for overlapping links
            if self.contour_mask:
                x_cnt = body.cnt_update[0]
                y_cnt = body.cnt_update[1]
                if link_id == 0:
                    body_p = comp.bodies[body_i + 1]
                    pt = Rs[animat_id][link_id + 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id + 1][:, None]
                    )
                    mask = (body_p.sdf(pt[0], pt[1]) - body.h) >= 0
                elif link_id == urdf_poses[animat_id].shape[0] - 1:
                    body_m = comp.bodies[body_i - 1]
                    pt = Rs[animat_id][link_id - 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id - 1][:, None]
                    )
                    mask = (body_m.sdf(pt[0], pt[1]) - body.h) >= 0
                else:
                    body_m = comp.bodies[body_i - 1]
                    pt_m = Rs[animat_id][link_id - 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id - 1][:, None]
                    )
                    body_p = comp.bodies[body_i + 1]
                    pt_p = Rs[animat_id][link_id + 1].T @ (
                        torch.stack((x_cnt, y_cnt))
                        - urdf_poses[animat_id][link_id + 1][:, None]
                    )
                    sdf_m = body_m.sdf(pt_m[0], pt_m[1]) - body.h
                    sdf_p = body_p.sdf(pt_p[0], pt_p[1]) - body.h
                    mask = (sdf_m >= 0) & (sdf_p >= 0)
                body.mask = mask

            body.r_com   = body.cnt_update - com_t[body_i, :, None]
            body.com_pos = com_t[body_i]

    @staticmethod
    def _body_sdf_grid(body):
        """Return ``(F, x, y)`` for any 2-D body type.

        * **Mesh / interpolated bodies** store the SDF on a regular
          grid inside a ``RegularGridInterpolator``-like object →
          return ``(sdf.F, sdf.x, sdf.y)`` directly.
        * **Analytical bodies** (circle, box, segment …) only expose
          a callable ``sdf(X, Y)`` with no pre-sampled grid → evaluate
          the SDF on a local grid covering the body contour plus a
          margin of ``4 h`` (enough for the BDIM transition layer +
          bilinear stencil) and return the sampled tensor.
        """
        if hasattr(body.sdf, 'F'):
            # Mesh / interpolated body – grid already available
            return body.sdf.F, body.sdf.x, body.sdf.y

        # ── Analytical body: pre-sample onto a regular grid ──────
        h = body.h
        margin = 4.0 * h

        # Derive local extent from the contour (computed during init
        # in _initialize_2d via skimage.measure.find_contours).
        x_lo = float(body.cnt[0].min()) - margin
        x_hi = float(body.cnt[0].max()) + margin
        y_lo = float(body.cnt[1].min()) - margin
        y_hi = float(body.cnt[1].max()) + margin

        nx = max(int((x_hi - x_lo) / h) + 1, 4)
        ny = max(int((y_hi - y_lo) / h) + 1, 4)

        x_coords = torch.linspace(x_lo, x_hi, nx,
                                  device=body.device, dtype=body.dtype)
        y_coords = torch.linspace(y_lo, y_hi, ny,
                                  device=body.device, dtype=body.dtype)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        F = body.sdf(X, Y)
        return F, x_coords, y_coords

    def _init_batched_sdf_2d(self):
        """Pre-build batched SDF tensor and normalization constants for grid_sample.

        Called lazily on the first ``_update_2d`` invocation.  The SDF
        fields are in body-local coordinates and never change, so this
        only runs once.

        Body SDF grids may have different resolutions (e.g. legs vs
        torso links).  We pad every SDF to the maximum (H, W) with a
        far-field sentinel so they can be stacked into a single tensor
        for ``grid_sample``.  The coordinate normalization maps each
        body's physical extent to the *unpadded* sub-region of the
        padded tensor.
        """
        comp = self.fluid_solver.composite_body
        B = comp.nbodies
        _FAR = 1e4  # sentinel: "far from any body surface"

        # --- extract (F, x, y) for every body (mesh or analytical) ---
        grids = [self._body_sdf_grid(body) for body in comp.bodies]

        # --- gather per-body shapes and find max dims ---
        shapes = [(F.shape[0], F.shape[1]) for F, _x, _y in grids]
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)

        # --- pad each SDF to (max_h, max_w) ---
        padded = []
        for (F, _x, _y), (h_i, w_i) in zip(grids, shapes):
            if h_i < max_h or w_i < max_w:
                # F.pad order: (W_left, W_right, H_top, H_bottom)
                F = torch.nn.functional.pad(
                    F, (0, max_w - w_i, 0, max_h - h_i), value=_FAR)
            padded.append(F)

        self._sdf_batch = torch.stack(padded).unsqueeze(1).contiguous()
        # shape: (B, 1, max_h, max_w)

        # --- axis ranges for coordinate normalization ---
        x_min = torch.tensor([g[1][0].item()  for g in grids],
                             device=self.device, dtype=self.dtype)
        x_max = torch.tensor([g[1][-1].item() for g in grids],
                             device=self.device, dtype=self.dtype)
        y_min = torch.tensor([g[2][0].item()  for g in grids],
                             device=self.device, dtype=self.dtype)
        y_max = torch.tensor([g[2][-1].item() for g in grids],
                             device=self.device, dtype=self.dtype)

        # Original per-body pixel counts (float for division)
        h_orig = torch.tensor([s[0] for s in shapes],
                              device=self.device, dtype=self.dtype)
        w_orig = torch.tensor([s[1] for s in shapes],
                              device=self.device, dtype=self.dtype)

        # --- affine map: physical coord → grid_sample [-1, 1] ---
        # With align_corners=True the grid maps [-1,+1] onto
        # [0, max_dim-1].  The *original* data lives in [0, orig-1],
        # so physical x_min→ grid=-1 and x_max→ grid=-1+2*(orig-1)/(max-1).
        #   grid_x = x * scale_x + offset_x
        # where scale_x = 2*(h_orig-1) / ((x_max-x_min) * (max_h-1))
        self._sdf_x_scale  = 2.0 * (h_orig - 1) / ((x_max - x_min) * (max_h - 1))
        self._sdf_x_offset = -1.0 - x_min * self._sdf_x_scale
        self._sdf_y_scale  = 2.0 * (w_orig - 1) / ((y_max - y_min) * (max_w - 1))
        self._sdf_y_offset = -1.0 - y_min * self._sdf_y_scale

        n_unique = len(set(shapes))
        print(f"  [batched-SDF] built ({B}, 1, {max_h}, {max_w}) "
              f"grid_sample tensor  "
              f"({self._sdf_batch.nelement()*self._sdf_batch.element_size()/1e6:.0f} MB)  "
              f"({n_unique} unique SDF sizes, padded to max)")

    # ---- 3-D update --------------------------------------------------
    # ------------------------------------------------------------------
    #  AABB narrow-band helpers (3-D)
    # ------------------------------------------------------------------
    @staticmethod
    def _body_aabb_indices(body, R, urdf_pos, comp_x, comp_y, comp_z,
                           h, gs, pad=3):
        """Compute fluid-grid index ranges for a body's SDF domain.

        Transforms the body-local SDF grid bounding box into world
        coordinates (accounting for rotation), then finds the axis-
        aligned sub-block of the fluid grid that covers this region
        plus ``pad`` cells of margin on each side.

        Returns (i0, i1, j0, j1, k0, k1) — Python-style half-open
        indices suitable for slicing ``comp.X[i0:i1, j0:j1, k0:k1]``.
        If the sub-block covers >90 % of the grid, returns ``None``
        to signal that full-grid evaluation is cheaper (avoids the
        fill + scatter overhead).
        """
        # Body SDF grid bounds in local coordinates
        sdf_lo = torch.stack([body.sdf.x[0],  body.sdf.y[0],  body.sdf.z[0]])
        sdf_hi = torch.stack([body.sdf.x[-1], body.sdf.y[-1], body.sdf.z[-1]])

        local_center = 0.5 * (sdf_lo + sdf_hi)
        local_half   = 0.5 * (sdf_hi - sdf_lo)

        # AABB of the oriented bounding box in world space
        #   world_half[i] = Σ_j |R[i,j]| · local_half[j]
        world_half   = R.abs() @ local_half
        world_center = R @ local_center + urdf_pos

        w_min = world_center - world_half
        w_max = world_center + world_half

        # Convert to grid indices (uniform grid: idx = (coord - x0) / h)
        x0, y0, z0 = comp_x[0], comp_y[0], comp_z[0]
        inv_h = 1.0 / h

        i0 = max(0,     int(((w_min[0] - x0) * inv_h).item()) - pad)
        i1 = min(gs[0], int(((w_max[0] - x0) * inv_h).item()) + 1 + pad)
        j0 = max(0,     int(((w_min[1] - y0) * inv_h).item()) - pad)
        j1 = min(gs[1], int(((w_max[1] - y0) * inv_h).item()) + 1 + pad)
        k0 = max(0,     int(((w_min[2] - z0) * inv_h).item()) - pad)
        k1 = min(gs[2], int(((w_max[2] - z0) * inv_h).item()) + 1 + pad)

        sub_vol = (i1 - i0) * (j1 - j0) * (k1 - k0)
        full_vol = gs[0] * gs[1] * gs[2]

        if sub_vol > 0.9 * full_vol:
            return None                       # fall back to full grid

        return (i0, i1, j0, j1, k0, k1)


    # ---- 3-D update --------------------------------------------------
    def _update_3d(self, t, iteration, dt=1):
        fs   = self.fluid_solver
        comp = fs.composite_body
        gs   = fs.grid_shape

        # Far-field SDF value (>> eps) so mu0 = 1 (pure fluid) and
        # union-min always prefers the closest real body value.
        _FAR = 1e4

        com_poses  = []
        urdf_poses = []
        Rs         = []
        lin_vels   = []
        ang_vels   = []

        for exp_data in self.data:
            sen = exp_data.sensors.links
            com_poses.append(self.cython2numpy(sen.com_positions()[iteration, :]))
            urdf_poses.append(self.cython2numpy(sen.urdf_positions()[iteration, :]))
            Rs.append(
                self.cython2numpy(
                    Rotation.from_quat(
                        sen.urdf_orientations()[iteration, :]
                    ).as_matrix().astype(self.dtype_np)
                )
            )
            lin_vels.append(self.cython2numpy(sen.com_lin_velocities()[iteration, :]))
            nlinks = len(sen.names)
            ang_vels.append(
                self.cython2numpy(
                    np.stack([sen.com_ang_velocity(iteration, lk)
                              for lk in range(nlinks)])
                )
            )

        h_grid = comp.h          # uniform grid spacing

        # Per-body SDF storage for forces (reset sparse list each step)
        comp._sdf_sparse = [None] * len(comp.bodies)

        # Initialise union fields to _FAR / zero (once per step).
        # Reuse existing tensors in-place to avoid re-allocation.
        comp.sdf_val.fill_(_FAR)
        comp.sdf_val_u.fill_(_FAR)
        comp.sdf_val_v.fill_(_FAR)
        comp.sdf_val_w.fill_(_FAR)
        comp.body_u.zero_()
        comp.body_v.zero_()
        comp.body_w.zero_()

        # Cache per-body AABBs for downstream use (e.g. narrow-band forces)
        if not hasattr(comp, '_body_aabbs'):
            comp._body_aabbs = [None] * len(comp.bodies)

        for body_i, body in enumerate(comp.bodies):
            (animat_id, link_id) = comp.body_ids[body_i]

            com_pos  = com_poses[animat_id][link_id]
            urdf_pos = urdf_poses[animat_id][link_id]
            R        = Rs[animat_id][link_id]
            lin_vel  = lin_vels[animat_id][link_id]
            ang_vel  = ang_vels[animat_id][link_id]

            R_T = R.T  # (3, 3) — transposed rotation

            # ── AABB-clipped SDF evaluation ─────────────────────────
            aabb = None
            if hasattr(body, 'sdf') and hasattr(body.sdf, 'z'):
                aabb = self._body_aabb_indices(
                    body, R, urdf_pos,
                    comp.x, comp.y, comp.z,
                    h_grid, gs, pad=3,
                )
            comp._body_aabbs[body_i] = aabb   # cache for force integration

            if aabb is not None:
                # ── Sub-block path (main saving) ────────────────────
                # Only rotate + interpolate the AABB sub-block of the
                # grid, then stagger / velocity / union-min ALL on the
                # sub-block with contiguous intermediates.  Strided
                # reads of union fields are small (~3-5 MB per body)
                # and fit in L2 cache.
                (i0, i1, j0, j1, k0, k1) = aabb
                sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

                px, py, pz = rotate_grid_3d(
                    comp.X[sl], comp.Y[sl], comp.Z_grid[sl],
                    R_T, urdf_pos,
                )
                sdf_sub = body.sdf(px, py, pz)       # contiguous

                # Per-body SDF (sparse: store sub-block + AABB for forces)
                comp._sdf_sparse[body_i] = (aabb, sdf_sub)

                # Sub-block stagger (contiguous in → contiguous out)
                sdf_sub_u, sdf_sub_v, sdf_sub_w = _stagger_sdf_3d(
                    sdf_sub)

                # Sub-block body velocity (strided coord reads →
                # contiguous output from element-wise arithmetic)
                vel_sub_u = (
                    lin_vel[0]
                    + ang_vel[1] * (comp.Zu_stag[sl] - com_pos[2])
                    - ang_vel[2] * (comp.Yu_stag[sl] - com_pos[1]))
                vel_sub_v = (
                    lin_vel[1]
                    + ang_vel[2] * (comp.Xv_stag[sl] - com_pos[0])
                    - ang_vel[0] * (comp.Zv_stag[sl] - com_pos[2]))
                vel_sub_w = (
                    lin_vel[2]
                    + ang_vel[0] * (comp.Yw_stag[sl] - com_pos[1])
                    - ang_vel[1] * (comp.Xw_stag[sl] - com_pos[0]))

                # Sub-block union min  ─  read old union values
                # (contiguous copy avoids slow strided reads inside
                # torch.where), then write result back (strided write).
                old_cc = comp.sdf_val[sl].contiguous()
                mask_cc = sdf_sub < old_cc
                comp.sdf_val[sl] = torch.where(
                    mask_cc, sdf_sub, old_cc)

                old_u = comp.sdf_val_u[sl].contiguous()
                mask_u = sdf_sub_u < old_u
                comp.sdf_val_u[sl] = torch.where(
                    mask_u, sdf_sub_u, old_u)
                comp.body_u[sl] = torch.where(
                    mask_u, vel_sub_u,
                    comp.body_u[sl].contiguous())

                old_v = comp.sdf_val_v[sl].contiguous()
                mask_v = sdf_sub_v < old_v
                comp.sdf_val_v[sl] = torch.where(
                    mask_v, sdf_sub_v, old_v)
                comp.body_v[sl] = torch.where(
                    mask_v, vel_sub_v,
                    comp.body_v[sl].contiguous())

                old_w = comp.sdf_val_w[sl].contiguous()
                mask_w = sdf_sub_w < old_w
                comp.sdf_val_w[sl] = torch.where(
                    mask_w, sdf_sub_w, old_w)
                comp.body_w[sl] = torch.where(
                    mask_w, vel_sub_w,
                    comp.body_w[sl].contiguous())

            else:
                # ── Full-grid path (body covers >90 % of grid) ─────
                _rotate = (_rotate_grid_3d_compiled
                           if fs._compile_sdf else rotate_grid_3d)
                px, py, pz = _rotate(comp.X, comp.Y, comp.Z_grid,
                                     R_T, urdf_pos)
                sdf_cc = body.sdf(px, py, pz)
                comp._sdf_sparse[body_i] = (None, sdf_cc)  # None = full grid

                sdf_u, sdf_v, sdf_w = _stagger_sdf_3d(sdf_cc)

                vel_u = (lin_vel[0]
                         + ang_vel[1] * (comp.Zu_stag - com_pos[2])
                         - ang_vel[2] * (comp.Yu_stag - com_pos[1]))
                vel_v = (lin_vel[1]
                         + ang_vel[2] * (comp.Xv_stag - com_pos[0])
                         - ang_vel[0] * (comp.Zv_stag - com_pos[2]))
                vel_w = (lin_vel[2]
                         + ang_vel[0] * (comp.Yw_stag - com_pos[1])
                         - ang_vel[1] * (comp.Xw_stag - com_pos[0]))

                mask_cc = sdf_cc < comp.sdf_val
                comp.sdf_val = torch.where(mask_cc, sdf_cc, comp.sdf_val)

                mask_u = sdf_u < comp.sdf_val_u
                comp.sdf_val_u = torch.where(mask_u, sdf_u, comp.sdf_val_u)
                comp.body_u    = torch.where(mask_u, vel_u, comp.body_u)

                mask_v = sdf_v < comp.sdf_val_v
                comp.sdf_val_v = torch.where(mask_v, sdf_v, comp.sdf_val_v)
                comp.body_v    = torch.where(mask_v, vel_v, comp.body_v)

                mask_w = sdf_w < comp.sdf_val_w
                comp.sdf_val_w = torch.where(mask_w, sdf_w, comp.sdf_val_w)
                comp.body_w    = torch.where(mask_w, vel_w, comp.body_w)

            comp.com_pos[body_i] = com_pos
            body.com_pos = com_pos

    # ==================================================================
    #  fluid_step: one BDIM time-step (advection-diffusion + projection)
    # ==================================================================
    def _compute_variable_density_coefficients(self, timestep):
        """Compute variable-density Poisson coefficients.

        Returns (ch, cv) for 2-D or (ch, cv, cw) for 3-D, using
        rho = rho_body + (rho_fluid - rho_body) * mu0  at each
        staggered grid location.
        """
        fs = self.fluid_solver
        _drho = self.rho_fluid - self.rho_body

        ch = timestep / (self.rho_body + _drho * fs.mu0_all_u)
        cv = timestep / (self.rho_body + _drho * fs.mu0_all_v)
        if self.ndim == 3:
            cw = timestep / (self.rho_body + _drho * fs.mu0_all_w)
            return ch, cv, cw
        return ch, cv

    def fluid_step(self, *args):
        if self.ndim == 3:
            return self._fluid_step_3d(*args)
        return self._fluid_step_2d(*args)

    # ---- 2-D fluid step ---------------------------------------------
    def _fluid_step_2d(self, u, v, p, timestep):
        fs = self.fluid_solver
        _bdim = fs._bdim_meta_compiled
        _h = fs.h

        nu_t = fs._compute_nu_t(u, v)
        (uprime, vprime) = fs.adv_diff_solver.solve(u, v, nu_t=nu_t)
        # Clone CUDA-graph outputs so they can safely be passed to
        # another compiled kernel (_bdim) and modified by set_BCs.
        uprime = uprime.clone()
        vprime = vprime.clone()
        fs.adv_diff_solver.set_BCs(uprime, vprime)

        # BDIM2 meta-equation  (reuses FluidSolver's compiled kernel)
        uprime = _bdim(
            uprime, fs.mu0_all_u,
            fs.composite_body.body_u, fs.mu1_all_u,
            fs.normal_x_u, fs.normal_y_u, _h, 2,
        ).clone()
        vprime = _bdim(
            vprime, fs.mu0_all_v,
            fs.composite_body.body_v, fs.mu1_all_v,
            fs.normal_x_v, fs.normal_y_v, _h, 2,
        ).clone()

        # Variable-density Poisson coefficients
        poisson_method = getattr(fs, "poisson_method", "multigrid")
        if poisson_method == "fft":
            coeff = timestep / self.rho_fluid
            (u, v, p) = fs.project(uprime, vprime, p, ch=coeff, cv=coeff)
        else:
            ch, cv = self._compute_variable_density_coefficients(timestep)
            (u, v, p) = fs.project(uprime, vprime, p, ch=ch, cv=cv)

        # Sponge damping (2-D)
        if fs.use_sponge:
            (u, v) = fs.apply_sponge_damping(u, v)

        fs.adv_diff_solver.set_BCs(u, v)
        return (u, v, p)

    # ---- 3-D fluid step ---------------------------------------------
    def _fluid_step_3d(self, u, v, w, p, timestep):
        fs = self.fluid_solver
        _bdim = fs._bdim_meta_compiled
        _h = fs.h

        nu_t = fs._compute_nu_t(u, v, w)
        (uprime, vprime, wprime) = fs.adv_diff_solver.solve(u, v, w, nu_t=nu_t)
        # Clone CUDA-graph outputs so they can safely be passed to
        # another compiled kernel (_bdim) and modified by set_BCs.
        uprime = uprime.clone()
        vprime = vprime.clone()
        wprime = wprime.clone()
        fs.adv_diff_solver.set_BCs(uprime, vprime, wprime)

        # BDIM2 meta-equation  (reuses FluidSolver's compiled kernel)
        uprime = _bdim(
            uprime, fs.mu0_all_u,
            fs.composite_body.body_u, fs.mu1_all_u,
            fs.normal_x_u, fs.normal_y_u, fs.normal_z_u, _h, 3,
        ).clone()
        vprime = _bdim(
            vprime, fs.mu0_all_v,
            fs.composite_body.body_v, fs.mu1_all_v,
            fs.normal_x_v, fs.normal_y_v, fs.normal_z_v, _h, 3,
        ).clone()
        wprime = _bdim(
            wprime, fs.mu0_all_w,
            fs.composite_body.body_w, fs.mu1_all_w,
            fs.normal_x_w, fs.normal_y_w, fs.normal_z_w, _h, 3,
        ).clone()

        # ── Free mu1 + staggered normals right after BDIM ────────────
        # project() only uses mu0_{u,v,w}.  forces_method2_3d recomputes
        # CC normals on-the-fly.  Freeing these 12 grid-sized arrays
        # (~12 × 131 MB ≈ 1.6 GB) reduces peak memory during the
        # Poisson solve that follows.
        for _attr in ('mu1_all_u', 'mu1_all_v', 'mu1_all_w',
                      'normal_x_u', 'normal_y_u', 'normal_z_u',
                      'normal_x_v', 'normal_y_v', 'normal_z_v',
                      'normal_x_w', 'normal_y_w', 'normal_z_w',
                      # CC mu1 / m_m0 are also not needed by project
                      'mu1_all', 'm_m0_all'):
            if hasattr(fs, _attr):
                setattr(fs, _attr, None)

        # Variable-density Poisson coefficients
        poisson_method = getattr(fs, "poisson_method", "multigrid")
        if poisson_method == "fft":
            coeff = timestep / self.rho_fluid
            # Free mu0 before Poisson solve (not needed by FFT path)
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w', 'mu0_all'):
                if hasattr(fs, _attr):
                    setattr(fs, _attr, None)
            (u, v, w, p) = fs.project(uprime, vprime, p,
                                      w_vel=wprime, ch=coeff, cv=coeff, cw=coeff)
        else:
            ch, cv, cw = self._compute_variable_density_coefficients(timestep)

            # ── Free mu0 now — ch/cv/cw are independent tensors ──────
            for _attr in ('mu0_all_u', 'mu0_all_v', 'mu0_all_w', 'mu0_all'):
                if hasattr(fs, _attr):
                    setattr(fs, _attr, None)

            (u, v, w, p) = fs.project(uprime, vprime, p,
                                      w_vel=wprime, ch=ch, cv=cv, cw=cw)

        # Sponge damping: damp velocity near domain boundaries
        if fs.use_sponge:
            (u, v, w) = fs.apply_sponge_damping(u, v, w)

        fs.adv_diff_solver.set_BCs(u, v, w)
        return (u, v, w, p)

    # ==================================================================
    #  apply_forces: fluid -> body forces via MuJoCo xfrc_applied
    # ==================================================================
    def apply_forces(self, task, physics):
        if self.ndim == 3:
            self._apply_forces_3d(task, physics)
        else:
            self._apply_forces_2d(task, physics)

    def _apply_forces_2d(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling

        # Single GPU→CPU transfer instead of 6 separate .cpu().numpy() calls
        forces_gpu = torch.stack([
            fs.friction_force_lin_x,
            fs.friction_force_lin_y,
            fs.friction_force_ang_z,
            fs.pressure_force_x,
            fs.pressure_force_y,
            fs.pressure_force_ang_z,
        ])                                          # (6, B)
        forces_cpu = (s * forces_gpu).cpu().numpy() # single sync
        friction_force_lin_x = forces_cpu[0]
        friction_force_lin_y = forces_cpu[1]
        friction_force_ang_z = forces_cpu[2]
        pressure_force_x     = forces_cpu[3]
        pressure_force_y     = forces_cpu[4]
        pressure_force_ang_z = forces_cpu[5]

        for body_i in range(len(fs.composite_body.bodies)):
            (animat_id, link_id) = fs.composite_body.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            physics.data.xfrc_applied[ind, 0] = (
                friction_force_lin_x[body_i] + pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                friction_force_lin_y[body_i] + pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 5] = (
                friction_force_ang_z[body_i] + pressure_force_ang_z[body_i]
            ) * task.units.newtons

    def _apply_forces_3d(self, task, physics):
        fs = self.fluid_solver
        s  = self.force_scaling

        friction_force_lin_x = s * fs.friction_force_lin_x.cpu().numpy()
        friction_force_lin_y = s * fs.friction_force_lin_y.cpu().numpy()
        friction_force_lin_z = s * fs.friction_force_lin_z.cpu().numpy()
        friction_force_ang_x = s * fs.friction_force_ang_x.cpu().numpy()
        friction_force_ang_y = s * fs.friction_force_ang_y.cpu().numpy()
        friction_force_ang_z = s * fs.friction_force_ang_z.cpu().numpy()

        pressure_force_x     = s * fs.pressure_force_x.cpu().numpy()
        pressure_force_y     = s * fs.pressure_force_y.cpu().numpy()
        pressure_force_z     = s * fs.pressure_force_z.cpu().numpy()
        pressure_force_ang_x = s * fs.pressure_force_ang_x.cpu().numpy()
        pressure_force_ang_y = s * fs.pressure_force_ang_y.cpu().numpy()
        pressure_force_ang_z = s * fs.pressure_force_ang_z.cpu().numpy()

        # ---- FARMS-identical buoyancy (drag.pyx  compute_buoyancy) ----
        # Lazy-init per-body mass & half-height on first call
        if not self._buoyancy_initialized:
            self._init_buoyancy_params(task, physics)

        comp    = fs.composite_body
        surface = self.water_surface
        g_z     = self.gravity_z          # e.g. -9.81

        for body_i in range(len(comp.bodies)):
            (animat_id, link_id) = comp.body_ids[body_i]
            ind = task.maps[animat_id]["sensors"]["data2xfrc"][link_id]

            # FARMS buoyancy per link
            mass   = self._buoy_mass[body_i]
            height = self._buoy_height[body_i]
            pos_z  = float(comp.com_pos[body_i][2])

            buoyancy_z = 0.0
            if mass > 0 and height > 0 and pos_z - height < surface:
                frac = min((surface + height - pos_z) / (2.0 * height), 1.0)
                # -rho_water * (mass/density) * gravity * frac
                # = -rho_water * V_link * gravity * frac  (upward when g<0)
                buoyancy_z = (
                    -self.rho_fluid * mass * g_z / self.rho_body * frac
                )

            physics.data.xfrc_applied[ind, 0] = (
                friction_force_lin_x[body_i] + pressure_force_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 1] = (
                friction_force_lin_y[body_i] + pressure_force_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 2] = (
                friction_force_lin_z[body_i] + pressure_force_z[body_i]
                + buoyancy_z
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 3] = (
                friction_force_ang_x[body_i] + pressure_force_ang_x[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 4] = (
                friction_force_ang_y[body_i] + pressure_force_ang_y[body_i]
            ) * task.units.newtons
            physics.data.xfrc_applied[ind, 5] = (
                friction_force_ang_z[body_i] + pressure_force_ang_z[body_i]
            ) * task.units.newtons

    # ==================================================================
    #  step: one full coupled step (called by FluidExtension.before_step)
    # ==================================================================
    def step(self, task, physics):

        iteration = self.iteration
        timestep  = self.pars["solver"]["dt"]
        if iteration >= self.pars["solver"]["nt"]:
            return

        t  = iteration * timestep
        fs = self.fluid_solver

        if not self.terminate:

            # 1. update SDF + body velocities from FARMS kinematics
            self.update(t, iteration, dt=timestep)

            # 2. recompute mu / normal fields
            if self.ndim == 3:
                fs._recompute_mu_normals_3d()
            else:
                fs._recompute_mu_normals_2d()

            # 3. BDIM fluid step
            if self.ndim == 3:
                (u, v, w, p) = self.fluid_step(
                    fs.u0, fs.v0, fs.w0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.w0, fs.p0) = (u, v, w, p)
            else:
                (u, v, p) = self.fluid_step(
                    fs.u0, fs.v0, fs.p0, timestep
                )
                if self.zero_pressure_inside:
                    p = torch.where(fs.composite_body.sdf_val < 0, 0, p)
                (fs.u0, fs.v0, fs.p0) = (u, v, p)

            # 4. compute fluid forces on each body
            if self.ndim == 3:
                fs.forces_method2_3d(fs.u0, fs.v0, fs.w0, fs.p0, iteration)
            elif self.force_method == "method1":
                fs.forces_method1(fs.u0, fs.v0, fs.p0, iteration)
            else:
                fs.forces_method2(fs.u0, fs.v0, fs.p0, iteration)

            # ── Free cached force-density tensors (6 × grid_shape) ───
            # These are only needed for the per-body integration above;
            # plotting_and_saving does not use them.
            for _attr in ('xstress_tensor', 'ystress_tensor', 'zstress_tensor',
                          'pforce_x', 'pforce_y', 'pforce_z'):
                if hasattr(fs, _attr):
                    setattr(fs, _attr, None)

            # 5. plotting / saving
            if self.ndim == 3:
                self.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, w_vel=fs.w0, check_termination=False
                )
            else:
                self.terminate = fs.plotting_and_saving(
                    fs.u0, fs.v0, fs.p0, iteration, check_termination=False
                )

            # 6. apply forces to MuJoCo bodies
            self.apply_forces(task, physics)

            # 7. free BDIM intermediates to reclaim GPU memory
            fs._release_bdim_fields()

        self.iteration += 1
