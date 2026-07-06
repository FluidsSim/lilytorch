"""Post-process a pleurodeles 3-D swim run.

Loads:
* ``<data_dir>/parameters.yaml``           (solver parameters: rho, dt, nu, grid)
* ``<data_dir>/output/simulation.hdf5``    (FARMS sensor data + link names)
* ``<data_dir>/output/drags.h5``           (lilytorch per-link forces/torques)
* ``<data_dir>/interp_data_3d/``           (per-link cached SDFs for geometry)

Produces:
* ``com_velocity_plot.png``            whole-animat COM velocity
* ``link_drag_coefficients.png``       per-link ``C_{D,alpha}`` (linear)
* ``link_torque_coefficients.png``     per-link ``C_{T,alpha}`` (angular)
* ``link_drag_coefficients.npz``       arrays for downstream analysis

Drag-coefficient conventions (documented here, applied in-code):

Non-dimensionalisation per link *i* and axis :math:`\\alpha \\in \\{x,y,z\\}`:

.. math::

    C_{D,i,\\alpha}(t) = \\frac{F_{i,\\alpha}(t)}
                              {\\tfrac{1}{2}\\rho\\, A_{w,i}\\,
                               U_{i,\\alpha}(t)\\,|U_{i,\\alpha}(t)|}

    C_{T,i,\\alpha}(t) = \\frac{T_{i,\\alpha}(t)}
                              {\\tfrac{1}{2}\\rho\\, A_{w,i}\\, L_i^{2}\\,
                               \\Omega_{i,\\alpha}(t)\\,|\\Omega_{i,\\alpha}(t)|}

Choices / assumptions:
  * ``F`` = viscous + pressure force from ``drags.h5``.
  * ``T`` = viscous + pressure torque about each link's COM.
  * ``U``, ``Omega`` = per-link linear / angular velocity from FARMS sensors
    (``link_com_velocity_{lin,ang}_{x,y,z}``) in the MuJoCo world frame —
    the same frame in which lilytorch records forces, so no rotation is
    applied.
  * ``rho`` is read from ``parameters.yaml`` (solver.rho).
  * Reference area ``A_{w,i}`` is reconstructed from the cached SDF
    ``interp_data_3d/sdf_val_<link>_collision.npy`` by integrating a
    cosine-kernel regularised delta against the (true) signed distance:
    :math:`A \\approx \\sum \\delta_\\varepsilon(\\phi)\\,dx\\,dy\\,dz`
    with :math:`\\varepsilon = 2\\max(dx,dy,dz)`. This works because the
    cached field is a full SDF (``|\\nabla\\phi|=1``).
  * Reference length ``L_i`` = bounding-box extent along x of the SDF
    0-isosurface (computed from ``sdf_val`` and ``xnp_<link>.npy``).
  * Signed product ``U_alpha|U_alpha|`` is used so that :math:`C_D > 0`
    when the force opposes motion along that axis.
  * Below a velocity threshold ``|U| < max(1e-3, 0.01 max|U|)`` per axis
    the coefficient is masked to ``NaN``.
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from farms_core.sensors.sensor_convention import sc


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores unknown (e.g. ``!!python/object:...``) tags."""


def _construct_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("", _construct_unknown)
_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_unknown)


def _scrape_solver_rho(path: str) -> float | None:
    """Fallback: extract ``solver.rho`` by scanning lines of ``parameters.yaml``.

    The solver's ``parameters.yaml`` contains Python-object tags with
    self-referential anchors that PyYAML cannot construct. We only need
    ``solver.rho``, so scrape it textually.
    """
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return None
    in_solver = False
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("solver:"):
            in_solver = True
            continue
        if in_solver:
            # leaving the solver block once we hit another top-level key
            if stripped and not stripped.startswith((" ", "\t")):
                break
            s = stripped.strip()
            if s.startswith("rho:"):
                try:
                    return float(s.split(":", 1)[1].strip())
                except ValueError:
                    return None
    return None


def _load_parameters_yaml(path: str) -> dict:
    """Return a minimal dict with ``{'solver': {'rho': ...}}`` if available.

    The solver writes ``parameters.yaml`` with ``!!python/object:...`` tags
    and recursive anchors that neither ``safe_load`` nor a tolerant loader
    can reconstruct. Since only ``solver.rho`` is needed downstream, try a
    tolerant YAML load first and fall back to text scraping.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.load(f, Loader=_TolerantLoader) or {}
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass
    rho = _scrape_solver_rho(path)
    return {"solver": {"rho": rho}} if rho is not None else {}


def _load_link_sensors(sim_hdf5_path: str):
    """Return (times, link_vel_lin, link_vel_ang, link_names, timestep)."""
    with h5py.File(sim_hdf5_path, "r") as f:
        link_array = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"]
        )
        timestep = float(np.array(f["timestep"]))
        link_names = None
        links_grp = f["FARMSLISTanimats"]["0"]["sensors"]["links"]
        if "names" in links_grp:
            raw = np.array(links_grp["names"])
            link_names = [
                v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                for v in raw.tolist()
            ]

    nt = link_array.shape[0]
    times = timestep * np.arange(nt)
    vel_lin = link_array[
        :, :, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1
    ]  # (nt, n_links, 3)
    vel_ang = link_array[
        :, :, sc.link_com_velocity_ang_x : sc.link_com_velocity_ang_z + 1
    ]  # (nt, n_links, 3)
    urdf_quat = link_array[
        :, :, sc.link_urdf_orientation_x : sc.link_urdf_orientation_w + 1
    ]  # (nt, n_links, 4)  [x, y, z, w], body->world
    return times, vel_lin, vel_ang, urdf_quat, link_names, timestep


def _load_drags(drags_h5_path: str):
    """Return dict with per-link force/torque histories (no metadata)."""
    with h5py.File(drags_h5_path, "r") as f:
        return {
            "viscous_drags":    np.array(f["viscous_drags"]),
            "pressure_drags":   np.array(f["pressure_drags"]),
            "viscous_torques":  np.array(f["viscous_torques"]),
            "pressure_torques": np.array(f["pressure_torques"]),
        }


# ---------------------------------------------------------------------------
# Per-link geometry from cached SDFs
# ---------------------------------------------------------------------------
def _sdf_tag_for_link(link_name: str, interp_dir: str) -> str | None:
    """Map a FARMS link name to the mesh tag used in the cache filenames.

    FARMS stores ``link_body_00`` while the cache is named
    ``sdf_val_link_body_00_collision.npy``.  We try the direct name, then
    append ``_collision``.
    """
    for candidate in (link_name, f"{link_name}_collision"):
        if os.path.exists(os.path.join(interp_dir, f"sdf_val_{candidate}.npy")):
            return candidate
    return None


def _wetted_area_and_length_from_sdf(interp_dir: str, mesh_tag: str
                                     ) -> tuple[float, float]:
    """Compute (wetted_area [m^2], x-bbox length [m]) from the cached SDF.

    Uses a cosine-kernel regularised delta with
    ``eps = 2 * max(dx, dy, dz)``.
    """
    phi = np.load(os.path.join(interp_dir, f"sdf_val_{mesh_tag}.npy"))
    xnp = np.load(os.path.join(interp_dir, f"xnp_{mesh_tag}.npy"))
    ynp = np.load(os.path.join(interp_dir, f"ynp_{mesh_tag}.npy"))
    znp = np.load(os.path.join(interp_dir, f"znp_{mesh_tag}.npy"))

    dx = float(xnp[1] - xnp[0])
    dy = float(ynp[1] - ynp[0])
    dz = float(znp[1] - znp[0])
    eps = 2.0 * max(dx, dy, dz)

    mask = np.abs(phi) <= eps
    delta = np.zeros_like(phi, dtype=np.float64)
    phi_m = phi[mask].astype(np.float64)
    delta[mask] = (1.0 / (2.0 * eps)) * (1.0 + np.cos(np.pi * phi_m / eps))
    area = float(delta.sum()) * dx * dy * dz

    # x-bbox length of the body = extent of SDF<=0 along x
    inside = phi <= 0
    if inside.any():
        x_any = inside.any(axis=(1, 2))
        length = float(xnp[x_any][-1] - xnp[x_any][0])
    else:
        length = float(xnp[-1] - xnp[0])
    return area, length


def _load_link_geometry(link_names: list[str], interp_dir: str
                        ) -> tuple[np.ndarray, np.ndarray]:
    """For each link in ``link_names`` return (A_w, L) arrays (NaN if missing)."""
    n = len(link_names)
    A_w = np.full(n, np.nan)
    L   = np.full(n, np.nan)
    if not os.path.isdir(interp_dir):
        print(f"[warn] interp_data dir not found: {interp_dir}")
        return A_w, L
    for i, name in enumerate(link_names):
        tag = _sdf_tag_for_link(name, interp_dir)
        if tag is None:
            print(f"[warn] no cached SDF for link '{name}'")
            continue
        try:
            A_w[i], L[i] = _wetted_area_and_length_from_sdf(interp_dir, tag)
        except Exception as e:
            print(f"[warn] failed reading SDF for '{name}': {e}")
    return A_w, L


# ---------------------------------------------------------------------------
# Coefficient computation
# ---------------------------------------------------------------------------
def _signed_square(x: np.ndarray) -> np.ndarray:
    """Return ``x * |x|`` (keeps sign of *x*)."""
    return x * np.abs(x)


def _mask_small(denom: np.ndarray, ref: np.ndarray, rel: float = 0.01,
                abs_floor: float = 1e-3) -> np.ndarray:
    """NaN-mask wherever ``|ref|`` is below ``max(abs_floor, rel*max|ref|)``."""
    out = denom.copy()
    thr = max(abs_floor, rel * float(np.nanmax(np.abs(ref)) or 0.0))
    out[np.abs(ref) < thr] = np.nan
    return out


def compute_drag_coefficients(
    F: np.ndarray,         # (n_b, 3, nt)
    T: np.ndarray,         # (n_b, 3, nt)
    U: np.ndarray,         # (n_b, 3, nt)
    Omega: np.ndarray,     # (n_b, 3, nt)
    A_w: np.ndarray,       # (n_b,)
    L: np.ndarray,         # (n_b,)
    rho: float,
):
    """Return (C_D, C_T) of shapes (n_b, 3, nt) with NaNs at low velocity."""
    n_b = F.shape[0]
    A_w_col = A_w.reshape(n_b, 1, 1)
    L_col   = L.reshape(n_b, 1, 1)

    denom_lin = 0.5 * rho * A_w_col * _signed_square(U)
    denom_ang = 0.5 * rho * A_w_col * (L_col ** 2) * _signed_square(Omega)

    denom_lin_masked = np.empty_like(denom_lin)
    denom_ang_masked = np.empty_like(denom_ang)
    for i in range(n_b):
        for a in range(3):
            denom_lin_masked[i, a] = _mask_small(denom_lin[i, a], U[i, a])
            denom_ang_masked[i, a] = _mask_small(denom_ang[i, a], Omega[i, a])

    with np.errstate(divide="ignore", invalid="ignore"):
        C_D = F / denom_lin_masked
        C_T = T / denom_ang_masked
    return C_D, C_T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_com_velocity(times: np.ndarray, vel_lin: np.ndarray,
                       it_max: int, save_path: str,
                       xlim: tuple | None = None):
    v_com = np.mean(vel_lin, axis=1)  # (nt, 3)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for a, lab in enumerate("xyz"):
        ax.plot(times[:it_max], v_com[:it_max, a], label=f"v_{lab}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("COM Velocity (mean over links)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_per_link_grid(times: np.ndarray, coeff: np.ndarray, it_max: int,
                        link_names: list, title: str, ylabel: str,
                        save_path: str, ylim: tuple | None = None,
                        xlim: tuple | None = None):
    """One figure with a 3x1 grid (x/y/z axes) showing all links."""
    n_b = coeff.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    cmap = plt.get_cmap("viridis", max(n_b, 2))
    for a, ax in enumerate(axes):
        for i in range(n_b):
            lab = link_names[i] if i < len(link_names) and link_names[i] else f"link{i}"
            ax.plot(times[:it_max], coeff[i, a, :it_max], color=cmap(i),
                    lw=0.9, label=lab)
        ax.set_ylabel(f"{ylabel} ({'xyz'[a]})")
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlim is not None:
            ax.set_xlim(*xlim)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if n_b <= 12:
        axes[0].legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    else:
        fig.legend(handles, labels, loc="center right", fontsize=6,
                   bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FARMS drag-coefficient calibration (URDF-frame axis-wise regression)
# ---------------------------------------------------------------------------
def _rest_quat(q: np.ndarray, method: str = "svd") -> np.ndarray:
    """Return a unit quaternion representing the 'rest' orientation of ``q``.

    ``q`` has shape (N, 4) with scipy [x, y, z, w] convention.

    Two estimators are supported:

    - ``"svd"``: the chordal mean on SO(3). Averages the stack of rotation
       matrices and projects back to SO(3) via singular-value decomposition
       (Moakher 2002). Robust when the link undergoes *large* rotational
       excursions about a slowly-varying mean axis, which is typical of a
       swimming tail link that sweeps +/- tens of degrees at each stroke.

    - ``"first"``: use ``q[0]``. This is only correct if the first sample
       is guaranteed to be at rest (neutral URDF pose with zero joint
       deflection); otherwise the estimator is biased by whatever
       deformation is present at t = fit_t_start.

    The SVD/chordal mean is the default because it is a true Frechet-like
    mean: the 3x3 matrix that minimises the Frobenius distance to every
    sample. Antipodal samples are aligned first so the averaging is
    well-posed.
    """
    q = np.asarray(q)
    ref = q[0]
    signs = np.sign(q @ ref)
    signs[signs == 0] = 1.0
    q_aligned = q * signs[:, None]
    if method == "first":
        q_bar = q_aligned[0]
    else:
        # Chordal / Frobenius mean on SO(3): mean R, then SVD project.
        R_mat = Rotation.from_quat(q_aligned).as_matrix()  # (N, 3, 3)
        R_mean = R_mat.mean(axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R_proj = U @ D @ Vt
        q_bar = Rotation.from_matrix(R_proj).as_quat()
    return q_bar / np.linalg.norm(q_bar)


def _rotate_world_to_urdf(
    vec_world: np.ndarray,  # (n_b, 3, nt) in world axes
    quat_u2g: np.ndarray,   # (nt, n_b, 4) body->world quaternion [x,y,z,w]
    detilt: bool = False,
    rest_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``vec_world`` re-expressed in each link's body basis.

    ``quat_u2g[t, i]`` is the body->world rotation; the world->body rotation
    is its inverse, applied per (link, time) sample.

    If ``detilt=True`` the reference frame of each link is *re-centred* on
    its mean orientation over the samples selected by ``rest_mask`` (or all
    samples if ``rest_mask`` is None). Concretely, let R(t)=R_w2b(t) and
    R_bar be the mean body->world rotation; the de-tilted body axes B'
    satisfy "B' = world when link is at its rest pose", i.e. R_bar maps B'
    to world. A world vector is then expressed in B' via

        v_B'(t) = R_bar^T . R_w2b(t)^{-1}? -- equivalently,
        v_B'(t) = R_bar.apply( R_w2b(t).apply( v_world(t) ) ),

    where the first rotation takes the vector to the instantaneous URDF
    frame B(t), and the second rotates B(t) axes to B'(t) by removing the
    static offset R_bar. This is the *generally valid* fix:

      * Planar mode: R_bar contains the accumulated static rest-posture
        pitch of each spine segment; removing it cancels the spurious
        v_x^{world} -> v_z^{URDF} projection, so v_z^{B'} ~ 0 for every
        body link -- as physics dictates.
      * Full 3D: B' is still a body-fixed frame (it follows the link as it
        rotates); only the average orientation is zeroed. All dynamic
        rotational motion about the mean is preserved, so the fitted
        coefficients still represent the link's intrinsic anisotropy.
    """
    n_b, _, nt = vec_world.shape
    out = np.empty_like(vec_world)
    for i in range(n_b):
        q_i = quat_u2g[:nt, i, :]
        rots = Rotation.from_quat(q_i)
        v_w = vec_world[i].T                         # (nt, 3)
        v_b = rots.inv().apply(v_w)                  # URDF frame
        if detilt:
            sel = rest_mask if rest_mask is not None else slice(None)
            q_bar = _rest_quat(q_i[sel])
            rot_bar = Rotation.from_quat(q_bar)
            # The de-tilted body frame B' is defined so that B' = world
            # when the link is at its rest orientation R_bar. Then the
            # B'(t)->world rotation is R(t) * R_bar^{-1}, so
            # world->B'(t)  =  R_bar * R(t)^{-1}  =  rot_bar . rot_w2b.
            # v_B'(t) = rot_bar.apply( rot_w2b.apply(v_world(t)) ).
            v_b = rot_bar.apply(v_b)
        out[i] = v_b.T                               # (3, nt)
    return out


def fit_farms_drag_coefficients(
    F_urdf: np.ndarray,      # (n_b, 3, nt)
    T_urdf: np.ndarray,      # (n_b, 3, nt)
    V_urdf: np.ndarray,      # (n_b, 3, nt) link linear vel in URDF
    Omega_urdf: np.ndarray,  # (n_b, 3, nt) link angular vel in URDF
    viscosity: float,
    excitation_ratio: float = 1e-8,
    r2_min: float = 0.0,
):
    """Least-squares fit of FARMS per-axis quadratic drag coefficients.

    FARMS drag model (farms_mujoco/swimming/drag.pyx), per link *i* and per
    URDF-frame axis :math:`\\alpha \\in \\{x, y, z\\}`:

        F_alpha(t)   = mu * c^F_alpha * v_alpha(t)   * |v_alpha(t)|
        tau_alpha(t) =      c^T_alpha * omega_alpha(t) * |omega_alpha(t)|

    The three axes are treated *independently* (diagonal drag tensor in the
    URDF basis). For each (link, axis) we therefore solve a 1-D ordinary
    least-squares problem

        minimise_k  sum_t  ( F_alpha(t) - k * s_v(t) )**2,
        with s_v(t) = v_alpha(t) * |v_alpha(t)|,

    whose closed form is

                 sum_t  F_alpha(t) * s_v(t)
        k*  =   ----------------------------           (= mu * c^F_alpha)
                 sum_t  s_v(t)**2

    c^F_alpha is then obtained by dividing by the FARMS ``viscosity`` scalar
    (the dimensionless ``water.viscosity`` multiplier used at simulation
    time, *not* the kinematic viscosity of water). The torque coefficients
    use the same formula with s_w(t) = omega_alpha(t) |omega_alpha(t)| and
    no viscosity factor, since FARMS's ``compute_torque`` does not multiply
    by ``water.viscosity``.

    Goodness-of-fit per (link, axis) is the coefficient of determination

        R^2 = 1 - sum_t (F - F_hat)^2 / sum_t (F - mean(F))^2,

    with the same definition for torques.

    All three components (c^F_x, c^F_y, c^F_z and c^T_x, c^T_y, c^T_z) are
    computed by the *same* formula; they differ only in which axis of the
    URDF-frame velocity / angular velocity history drives the fit. For a
    slender undulating swimmer with near-zero vertical motion, the z
    denominator ``sum s_v_z**2`` becomes tiny, so c^F_z is numerically
    ill-conditioned and R^2_z is typically negative or meaningless.

    Returns dict with arrays of shape (n_b, 3):
        ``cF``, ``cT``, ``R2_F``, ``R2_T``, plus ``F_hat``/``T_hat`` and the
        input time histories for downstream diagnostics.
    """
    sv = V_urdf * np.abs(V_urdf)         # (n_b, 3, nt)
    sw = Omega_urdf * np.abs(Omega_urdf)

    # Closed-form 1-D least squares: k = <F, s> / <s, s>
    num_F = np.sum(F_urdf * sv, axis=-1)
    den_F = np.sum(sv * sv,     axis=-1)
    num_T = np.sum(T_urdf * sw, axis=-1)
    den_T = np.sum(sw * sw,     axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mu_cF = np.where(den_F > 0, num_F / den_F, np.nan)
        cT    = np.where(den_T > 0, num_T / den_T, np.nan)
    cF = mu_cF / float(viscosity)

    # R^2 per (link, axis)
    def _r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        ss_res = np.sum((y - yhat) ** 2, axis=-1)
        ss_tot = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    F_hat = mu_cF[..., None] * sv
    T_hat = cT[..., None]    * sw
    R2_F = _r2(F_urdf, F_hat)
    R2_T = _r2(T_urdf, T_hat)

    # ------------------------------------------------------------------
    # Conditioning guards (per link, per axis).
    #
    # The 1-D LS solution k* = <F, s> / <s, s> is ill-conditioned whenever
    # the regressor s_alpha(t) = v_alpha(t) |v_alpha(t)| is much smaller
    # than on the most excited axis of the same link. Two complementary
    # failure modes have to be detected:
    #
    #   * Amplitude-starved axes -- e.g. the vertical axis of a planar
    #     swimmer where v_z is quasi-zero. Guarded by
    #         r_amp_alpha = max_t |v_alpha(t)| / max_beta max_t |v_beta(t)|.
    #
    #   * DC-only / projection-leakage axes -- e.g. a tail link whose
    #     small roll / pitch couple lateral motion into body-z, producing
    #     a near-constant v_z of order 1e-3 m/s on top of zero oscillation.
    #     Both the amplitude guard *and* the 1-D LS see this as a valid
    #     signal (because v|v| has a nonzero mean) and would return a huge
    #     spurious c^F with moderate R^2. Guarded by an oscillation ratio
    #         r_osc_alpha = std_t(v_alpha(t)) / max_beta std_t(v_beta(t))
    #     which discounts the DC component and only counts genuine
    #     alternating motion (the physical source of quadratic drag).
    #
    # An axis is accepted only if *both* ratios exceed the threshold.
    # Amplitude threshold (``excitation_ratio``) defaults to 1e-2 and
    # oscillation threshold is derived as 5x that -- the AC signal needs
    # to be a more substantial fraction of the dominant axis because it
    # drives most of the drag-model variance.
    # ------------------------------------------------------------------
    vmax  = np.max(np.abs(V_urdf),     axis=-1)  # (n_b, 3)
    wmax  = np.max(np.abs(Omega_urdf), axis=-1)
    vstd  = np.std(V_urdf,     axis=-1)
    wstd  = np.std(Omega_urdf, axis=-1)
    vmax_link = np.max(vmax, axis=-1, keepdims=True)
    wmax_link = np.max(wmax, axis=-1, keepdims=True)
    vstd_link = np.max(vstd, axis=-1, keepdims=True)
    wstd_link = np.max(wstd, axis=-1, keepdims=True)
    osc_ratio = 5.0 * excitation_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        exc_F_amp = np.where(vmax_link > 0, vmax / vmax_link, 0.0)
        exc_T_amp = np.where(wmax_link > 0, wmax / wmax_link, 0.0)
        exc_F_osc = np.where(vstd_link > 0, vstd / vstd_link, 0.0)
        exc_T_osc = np.where(wstd_link > 0, wstd / wstd_link, 0.0)

    mask_F = (
        (exc_F_amp < excitation_ratio)
        | (exc_F_osc < osc_ratio)
        | ~np.isfinite(cF)
        | (R2_F < r2_min)
    )
    mask_T = (
        (exc_T_amp < excitation_ratio)
        | (exc_T_osc < osc_ratio)
        | ~np.isfinite(cT)
        | (R2_T < r2_min)
    )

    cF_clean = np.where(mask_F, 0.0, cF)
    cT_clean = np.where(mask_T, 0.0, cT)

    # Recompute reconstructions with the cleaned coefficients so the
    # diagnostic plots reflect the values actually written to YAML.
    mu_cF_clean = cF_clean * float(viscosity)
    F_hat = mu_cF_clean[..., None] * sv
    T_hat = cT_clean[..., None]    * sw

    return {
        "cF":   cF_clean,
        "cT":   cT_clean,
        "cF_raw": cF,
        "cT_raw": cT,
        "R2_F": R2_F,
        "R2_T": R2_T,
        "mask_F": mask_F,
        "mask_T": mask_T,
        "excitation_F": exc_F_amp,
        "excitation_T": exc_T_amp,
        "excitation_F_osc": exc_F_osc,
        "excitation_T_osc": exc_T_osc,
        "F_hat": F_hat,
        "T_hat": T_hat,
        "V_urdf": V_urdf,
        "Omega_urdf": Omega_urdf,
        "F_urdf": F_urdf,
        "T_urdf": T_urdf,
    }


def _emit_farms_yaml_fragment(
    link_names: list,
    cF: np.ndarray,
    cT: np.ndarray,
    save_path: str,
    viscosity: float = 1.0,
    R2_F: np.ndarray | None = None,
    R2_T: np.ndarray | None = None,
    fit_window: tuple | None = None,
) -> None:
    """Write a YAML fragment with per-link ``drag_coefficients``.

    A header comment documents the model and the closed-form least-squares
    solution used to obtain every number in the file, so the YAML is
    self-contained.
    """
    header = [
        "# FARMS drag coefficients fitted from CFD link forces / torques.",
        "# Model (URDF-frame, per link and per axis alpha in {x,y,z}):",
        "#     F_alpha(t)   = mu * c^F_alpha * v_alpha(t)   * |v_alpha(t)|",
        "#     tau_alpha(t) =      c^T_alpha * w_alpha(t)   * |w_alpha(t)|",
        "# Closed-form 1-D least squares per (link, axis):",
        "#     k*       = sum_t F*s_v  /  sum_t s_v**2,   s_v = v|v|",
        "#     c^F      = k* / mu                       (mu = water.viscosity)",
        "#     c^T      = sum_t tau*s_w / sum_t s_w**2, s_w = w|w|",
        f"# mu (water.viscosity) used in the fit: {viscosity:g}",
        (f"# Fit window: t in [{fit_window[0]:g}, {fit_window[1]:g}] s "
         "(initial transient discarded)."
         if fit_window is not None else
         "# Fit window: full record."),
        "# R^2 per (link, axis) shown as inline comments (fx/fy/fz | tx/ty/tz).",
        "",
    ]
    lines = list(header)
    for i, name in enumerate(link_names):
        cFi = [float(x) if np.isfinite(x) else 0.0 for x in cF[i]]
        cTi = [float(x) if np.isfinite(x) else 0.0 for x in cT[i]]
        r2f = R2_F[i] if R2_F is not None else [np.nan] * 3
        r2t = R2_T[i] if R2_T is not None else [np.nan] * 3
        r2_str = (
            f"  # R2_F=[{r2f[0]:+.2f},{r2f[1]:+.2f},{r2f[2]:+.2f}]  "
            f"R2_T=[{r2t[0]:+.2f},{r2t[1]:+.2f},{r2t[2]:+.2f}]"
        )
        lines.append(f"- name: {name}")
        lines.append(
            f"  drag_coefficients: [[{cFi[0]:.6g}, {cFi[1]:.6g}, {cFi[2]:.6g}], "
            f"[{cTi[0]:.6g}, {cTi[1]:.6g}, {cTi[2]:.6g}]]" + r2_str
        )
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot_fit_reconstruction(
    times: np.ndarray,
    truth: np.ndarray,     # (n_b, 3, nt)
    recon: np.ndarray,     # (n_b, 3, nt)
    link_names: list,
    title: str,
    ylabel: str,
    save_path: str,
    ncols: int = 6,
    detrend: bool = True,
) -> None:
    """One subplot per link, with x/y/z overlaid (solid=truth, dashed=fit).

    If ``detrend=True`` (default) the time-mean of each (link, axis)
    channel is subtracted from *both* truth and fit before plotting. This
    is the correct view of the fit quality because the FARMS quadratic
    model F = mu c^F v|v| has zero time-average for any symmetric gait
    (positive and negative excursions of v cancel in the integrand v|v|):
    the DC component of F is a hydrostatic / pressure-residual / wake
    contribution that the model *cannot* represent and *should not* be
    asked to match. Plotting the raw signal creates a visual mismatch on
    axes where the truth has a nonzero mean but the fit is zero, which is
    not a fit failure -- it is a property of the model space.
    """
    n_b = truth.shape[0]
    nrows = int(np.ceil(n_b / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.4 * ncols, 1.8 * nrows),
        sharex=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    axis_colors = ["#1f77b4", "#d62728", "#2ca02c"]  # x, y, z
    truth_mean = truth.mean(axis=-1, keepdims=True) if detrend else np.zeros_like(truth[..., :1])
    recon_mean = recon.mean(axis=-1, keepdims=True) if detrend else np.zeros_like(recon[..., :1])
    truth_plot = truth - truth_mean
    recon_plot = recon - recon_mean
    for i in range(n_b):
        ax = axes_flat[i]
        for a, col in enumerate(axis_colors):
            ax.plot(times, truth_plot[i, a], color=col, lw=0.6, alpha=0.7)
            ax.plot(times, recon_plot[i, a], color=col, lw=1.0, ls="--", alpha=0.9)
        name = link_names[i] if i < len(link_names) else f"link{i}"
        # Annotate any sizeable DC offset on each axis so the reader sees
        # how much 'truth' was subtracted (model cannot explain this).
        if detrend:
            dc_labels = []
            ampl = np.max(np.abs(truth_plot[i]))
            for a in range(3):
                dc = float(truth_mean[i, a, 0])
                # Show if DC is at least 10% of the AC amplitude -- this is
                # where the eye would otherwise see a 'mismatch'.
                if ampl > 0 and abs(dc) > 0.1 * ampl:
                    dc_labels.append(f"{'xyz'[a]}={dc:+.1e}")
            if dc_labels:
                ax.text(
                    0.02, 0.02, "<F>: " + ", ".join(dc_labels),
                    transform=ax.transAxes, fontsize=5,
                    va="bottom", ha="left", color="#555555",
                )
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    for j in range(n_b, len(axes_flat)):
        axes_flat[j].axis("off")
    # Shared legend: one entry per (axis, style)
    import matplotlib.lines as mlines
    legend_handles = []
    for a, col in enumerate(axis_colors):
        legend_handles.append(mlines.Line2D(
            [], [], color=col, lw=1.2, label=f"truth ({'xyz'[a]})"))
        legend_handles.append(mlines.Line2D(
            [], [], color=col, lw=1.2, ls="--", label=f"fit   ({'xyz'[a]})"))
    full_title = title
    if detrend:
        full_title = (title + "  -- time-mean subtracted per (link, axis); "
                      "DC components are hydrostatic / wake / projection bias "
                      "that a zero-mean quadratic-drag model cannot represent.")
    fig.suptitle(full_title, fontsize=10)
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=6, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.supxlabel("Time [s]", fontsize=9)
    fig.supylabel(ylabel, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-process pleurodeles 3-D swim: COM velocity and "
                    "per-link drag coefficients.",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/data/andreaferrario/ns_data/pleurodeles_3d/2026-04-22T21:44:50.482912",
        help="Run directory containing parameters.yaml, interp_data_3d/, "
             "and output/{simulation.hdf5,drags.h5}.",
    )
    parser.add_argument(
        "--interp_data_subfolder", type=str, default="interp_data_3d",
        help="Sub-folder of --data_dir containing the cached SDFs.",
    )
    parser.add_argument(
        "--interp_data_dir", type=str, default=None,
        help="Absolute path to the cached SDFs. Overrides "
             "--interp_data_subfolder. Defaults to the pleurodeles example "
             "folder alongside this script if set to 'auto'.",
    )
    parser.add_argument(
        "--it_max", type=int, default=None,
        help="Clip plotting to this number of iterations (default: all).",
    )
    parser.add_argument(
        "--cd_clip", type=float, default=None,
        help="y-axis clip for drag coefficient plots.",
    )
    parser.add_argument(
        "--fit_farms_coefficients", action="store_true",
        help="Fit axis-wise FARMS quadratic-drag coefficients (per link) in "
             "the URDF frame and emit a YAML fragment + diagnostic plots.",
    )
    parser.add_argument(
        "--farms_viscosity", type=float, default=1.0,
        help="Dimensionless viscosity scale used in the FARMS drag model "
             "(``arena.water.viscosity``; all stock FARMS configs use 1.0). "
             "Do NOT pass the real kinematic viscosity of water here.",
    )
    parser.add_argument(
        "--excitation_ratio", type=float, default=1e-2,
        help="Per-link axis *amplitude* threshold for the FARMS fit: an axis "
             "is zeroed if its peak |v_alpha| (resp. |omega_alpha|) is below "
             "this fraction of the link's most excited axis. Prevents huge "
             "spurious c^F_z values when the link barely moves vertically. "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--r2_min", type=float, default=0.0,
        help="Zero out fitted coefficients whose R^2 is below this threshold. "
             "Negative R^2 means the quadratic model is worse than a constant "
             "so the fitted scalar is not meaningful.",
    )
    parser.add_argument(
        "--fit_t_start", type=float, default=1.0,
        help="Start of the fitting window in seconds. Samples before this "
             "time are discarded as initial transient (ramp-up of the body "
             "from rest, dominated by added-mass / acceleration terms that "
             "the quadratic-drag model cannot capture).",
    )
    parser.add_argument(
        "--fit_t_end", type=float, default=None,
        help="End of the fitting window in seconds (default: full record). "
             "Use to restrict the fit to a specific swimming regime.",
    )
    parser.add_argument(
        "--no_detilt", action="store_true",
        help="Disable rest-pose de-tilting. By default, each link's body "
             "frame is re-centred on its mean orientation over the fit "
             "window, which removes the static rest-posture tilt that leaks "
             "e.g. axial motion into the 'vertical' URDF axis for a planar "
             "swimmer. Works in full 3D as well: only the average "
             "orientation is zeroed; dynamic rotation is preserved.",
    )
    args = parser.parse_args()

    out_dir    = os.path.join(args.data_dir, "output")
    params     = _load_parameters_yaml(os.path.join(args.data_dir,
                                                    "parameters.yaml"))
    # Prefer the run-local SDF cache; fall back to the source example folder
    # (``lilytorch/examples/pleurodeles/interp_data_3d``) where the
    # SimConfig actually writes it via ``data_folder``.
    if args.interp_data_dir is not None:
        interp_dir = args.interp_data_dir
    else:
        run_local = os.path.join(args.data_dir, args.interp_data_subfolder)
        source_local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    args.interp_data_subfolder)
        interp_dir = run_local if os.path.isdir(run_local) else source_local
    sim_h5     = os.path.join(out_dir, "simulation.hdf5")
    # ``drags.h5`` is written by ``FluidSolver.save_drags_h5`` into the run
    # directory itself (``self.save_path``), not into ``output/``.
    drags_h5_candidates = [
        os.path.join(args.data_dir, "drags.h5"),
        os.path.join(out_dir, "drags.h5"),
    ]
    drags_h5 = next((p for p in drags_h5_candidates if os.path.exists(p)), None)

    times, vel_lin, vel_ang, urdf_quat, link_names_sim, _ = _load_link_sensors(sim_h5)
    drags = _load_drags(drags_h5) if drags_h5 is not None else None
    if drags is None:
        print(f"[warn] drags.h5 not found in {drags_h5_candidates} -- "
              "skipping drag-coefficient plots.")

    # rho from parameters.yaml (written by the solver alongside drags.h5).
    rho = float(params.get("solver", {}).get("rho", 1000.0))

    it_max = args.it_max if args.it_max is not None else times.shape[0]
    it_max = min(it_max, times.shape[0])
    if drags is not None:
        it_max = min(it_max, drags["viscous_drags"].shape[-1])

    # Time window used for *all* time-history plots and for the FARMS fit.
    # Defaults: start at ``--fit_t_start`` (discard initial transient),
    # end at the record end unless ``--fit_t_end`` is given.
    t_end_plot = (args.fit_t_end if args.fit_t_end is not None
                  else float(times[it_max - 1]))
    plot_xlim = (float(args.fit_t_start), float(t_end_plot))
    print(f"Time-history plots restricted to t in "
          f"[{plot_xlim[0]:g}, {plot_xlim[1]:g}] s")

    _plot_com_velocity(times, vel_lin, it_max,
                       os.path.join(args.data_dir, "com_velocity_plot.png"),
                       xlim=plot_xlim)

    if drags is None:
        print(f"Wrote COM velocity plot under {args.data_dir}")
        return

    # Line up axes: solver records (n_bodies, 3, nt); FARMS sensors are
    # (nt, n_links, 3). Transpose sensors to match and clip to common length.
    n_b_drags = drags["viscous_drags"].shape[0]
    n_links   = vel_lin.shape[1]
    if n_b_drags != n_links:
        print(
            f"[warn] drags.h5 has {n_b_drags} bodies but simulation.hdf5 "
            f"has {n_links} links -- taking the leading "
            f"min({n_b_drags},{n_links})."
        )
    n_b = min(n_b_drags, n_links)
    link_names = (link_names_sim[:n_b] if link_names_sim
                  else [f"link{i}" for i in range(n_b)])

    U     = vel_lin[:it_max, :n_b, :].transpose(1, 2, 0).copy()
    Omega = vel_ang[:it_max, :n_b, :].transpose(1, 2, 0).copy()

    F_visc = drags["viscous_drags"][:n_b, :, :it_max]
    F_pres = drags["pressure_drags"][:n_b, :, :it_max]
    T_visc = drags["viscous_torques"][:n_b, :, :it_max]
    T_pres = drags["pressure_torques"][:n_b, :, :it_max]
    F = F_visc + F_pres
    T = T_visc + T_pres

    # Per-link force time histories (viscous / pressure / total)
    for arr, tag, ylabel, title in [
        (F_visc, "viscous",  r"$F_{\mathrm{visc}}$ [N]",
         "Per-link viscous (friction) force"),
        (F_pres, "pressure", r"$F_{\mathrm{pres}}$ [N]",
         "Per-link pressure force"),
        (F,      "total",    r"$F$ [N]",
         "Per-link total drag force (viscous + pressure)"),
    ]:
        _plot_per_link_grid(
            times, arr, it_max, link_names,
            title=title, ylabel=ylabel,
            save_path=os.path.join(args.data_dir,
                                   f"link_force_{tag}.png"),
            ylim=None,
            xlim=plot_xlim,
        )

    A_w, L = _load_link_geometry(link_names, interp_dir)
    # Guard against missing / zero entries
    A_w = np.where(np.isfinite(A_w) & (A_w > 0), A_w, np.nan)
    L   = np.where(np.isfinite(L)   & (L   > 0), L,   np.nan)

    C_D, C_T = compute_drag_coefficients(F, T, U, Omega, A_w, L, rho)

    _plot_per_link_grid(
        times, C_D, it_max, link_names,
        title=r"Per-link linear drag coefficients $C_{D,\alpha}=F_\alpha/"
              r"(\frac{1}{2} \rho\, A_w\, U_\alpha|U_\alpha|)$",
        ylabel=r"$C_D$",
        save_path=os.path.join(args.data_dir, "link_drag_coefficients.png"),
        ylim=(-args.cd_clip, args.cd_clip) if args.cd_clip is not None else None,
        xlim=plot_xlim,
    )
    _plot_per_link_grid(
        times, C_T, it_max, link_names,
        title=r"Per-link angular drag coefficients $C_{T,\alpha}=T_\alpha/"
              r"(\frac{1}{2} \rho\, A_w L^2\, \Omega_\alpha|\Omega_\alpha|)$",
        ylabel=r"$C_T$",
        save_path=os.path.join(args.data_dir, "link_torque_coefficients.png"),
        ylim=(-args.cd_clip, args.cd_clip) if args.cd_clip is not None else None,
        xlim=plot_xlim,
    )

    # ------------------------------------------------------------------
    # Time-averaged drag forces/torques over the window [plot_xlim].
    # Reports both <F> (signed mean) and <|F|> (magnitude mean). Also
    # aggregates across all body links.
    # ------------------------------------------------------------------
    t_win_mask = (times[:it_max] >= plot_xlim[0]) & (times[:it_max] <= plot_xlim[1])
    F_win = F[:, :, t_win_mask]
    T_win = T[:, :, t_win_mask]
    F_mean     = F_win.mean(axis=-1)          # (n_b, 3)  signed mean
    F_abs_mean = np.abs(F_win).mean(axis=-1)  # (n_b, 3)  magnitude mean
    T_mean     = T_win.mean(axis=-1)
    T_abs_mean = np.abs(T_win).mean(axis=-1)

    print(
        f"\nTime-averaged drag in window "
        f"[{plot_xlim[0]:g}, {plot_xlim[1]:g}] s  "
        f"({int(t_win_mask.sum())} samples):"
    )
    print(f"  {'link':24s}  "
          f"{'<Fx>':>9s} {'<Fy>':>9s} {'<Fz>':>9s}   "
          f"{'<|Fx|>':>9s} {'<|Fy|>':>9s} {'<|Fz|>':>9s}")
    for i, name in enumerate(link_names):
        print(
            f"  {name:24s}  "
            f"{F_mean[i,0]:9.3g} {F_mean[i,1]:9.3g} {F_mean[i,2]:9.3g}   "
            f"{F_abs_mean[i,0]:9.3g} {F_abs_mean[i,1]:9.3g} {F_abs_mean[i,2]:9.3g}"
        )
    F_tot      = F_mean.sum(axis=0)
    F_abs_tot  = F_abs_mean.sum(axis=0)
    print(
        f"  {'ALL LINKS (sum)':24s}  "
        f"{F_tot[0]:9.3g} {F_tot[1]:9.3g} {F_tot[2]:9.3g}   "
        f"{F_abs_tot[0]:9.3g} {F_abs_tot[1]:9.3g} {F_abs_tot[2]:9.3g}"
    )

    np.savez(
        os.path.join(args.data_dir, "link_drag_time_averages.npz"),
        window=np.array(plot_xlim),
        link_names=np.array(link_names, dtype=object),
        F_mean=F_mean, F_abs_mean=F_abs_mean,
        T_mean=T_mean, T_abs_mean=T_abs_mean,
    )
    print(f"Wrote link_drag_time_averages.npz under {args.data_dir}")

    np.savez(
        os.path.join(args.data_dir, "link_drag_coefficients.npz"),
        times=times[:it_max],
        F=F, T=T, U=U, Omega=Omega,
        C_D=C_D, C_T=C_T,
        wetted_area=A_w, body_length=L, rho=rho,
        link_names=np.array(link_names, dtype=object),
    )
    print(f"Wrote plots and arrays under {args.data_dir}")

    if args.fit_farms_coefficients:
        # Steady-state window: drop initial transient before the fit, and
        # use the same samples to estimate each link's *rest orientation*
        # for the de-tilting described in ``_rotate_world_to_urdf``.
        t_fit = times[:it_max]
        t_end = args.fit_t_end if args.fit_t_end is not None else t_fit[-1]
        fit_mask = (t_fit >= args.fit_t_start) & (t_fit <= t_end)
        if fit_mask.sum() < 10:
            raise RuntimeError(
                f"Fit window [{args.fit_t_start}, {t_end}] s contains "
                f"{fit_mask.sum()} samples; need at least 10. "
                "Lower --fit_t_start or raise --fit_t_end."
            )
        print(
            f"\nFit window: t in [{args.fit_t_start:g}, {t_end:g}] s "
            f"({fit_mask.sum()} / {it_max} samples, discarding "
            f"{(~fit_mask).sum()} transient samples)"
        )

        # Rotate world-frame quantities into each link's body basis. By
        # default the frame is *de-tilted* using the mean orientation over
        # the fit window so static rest-posture offsets do not contaminate
        # the per-axis quadratic fit (critical for planar-constrained
        # kinematics; also physically correct in full 3D).
        # ``urdf_quat`` is ``[x, y, z, w]``, body->world (matches MuJoCo xquat
        # after FARMS re-ordering in physicslinks2data).
        q_u2g = urdf_quat[:it_max, :n_b, :]  # (nt, n_b, 4)
        detilt = not args.no_detilt
        if detilt:
            print("Rest-pose de-tilting: ON (rest orientation = mean body->world "
                  "quaternion over the fit window).")
        else:
            print("Rest-pose de-tilting: OFF (using raw URDF frames).")
        F_urdf     = _rotate_world_to_urdf(F,     q_u2g, detilt=detilt, rest_mask=fit_mask)
        T_urdf     = _rotate_world_to_urdf(T,     q_u2g, detilt=detilt, rest_mask=fit_mask)
        V_urdf     = _rotate_world_to_urdf(U,     q_u2g, detilt=detilt, rest_mask=fit_mask)
        Omega_urdf = _rotate_world_to_urdf(Omega, q_u2g, detilt=detilt, rest_mask=fit_mask)

        F_urdf_fit     = F_urdf[:,     :, fit_mask]
        T_urdf_fit     = T_urdf[:,     :, fit_mask]
        V_urdf_fit     = V_urdf[:,     :, fit_mask]
        Omega_urdf_fit = Omega_urdf[:, :, fit_mask]

        fit = fit_farms_drag_coefficients(
            F_urdf_fit, T_urdf_fit, V_urdf_fit, Omega_urdf_fit,
            viscosity=args.farms_viscosity,
            excitation_ratio=args.excitation_ratio,
            r2_min=args.r2_min,
        )

        # YAML fragment
        yaml_path = os.path.join(args.data_dir, "farms_drag_coefficients.yaml")
        _emit_farms_yaml_fragment(
            link_names, fit["cF"], fit["cT"], yaml_path,
            viscosity=args.farms_viscosity,
            R2_F=fit["R2_F"], R2_T=fit["R2_T"],
            fit_window=(float(args.fit_t_start), float(t_end)),
        )

        # Per-link R^2 printout (compact)
        print(
            "\nFARMS drag-coefficient fit:\n"
            "  Model (URDF frame, per link, per axis alpha):\n"
            "    F_alpha(t)   = mu * c^F_alpha * v_alpha(t) |v_alpha(t)|\n"
            "    tau_alpha(t) =      c^T_alpha * w_alpha(t) |w_alpha(t)|\n"
            "  Closed-form LS per (link, axis):\n"
            "    c^F_alpha = (sum_t F_alpha * v_alpha|v_alpha|)\n"
            "              / (mu * sum_t (v_alpha|v_alpha|)^2)\n"
            "    c^T_alpha = (sum_t tau_alpha * w_alpha|w_alpha|)\n"
            "              / (sum_t (w_alpha|w_alpha|)^2)"
        )
        print(f"  mu (water.viscosity) = {args.farms_viscosity:g}, "
              f"n_t = {int(fit_mask.sum())}  "
              f"(window [{args.fit_t_start:g}, {t_end:g}] s)")
        n_masked_F = int(fit["mask_F"].sum())
        n_masked_T = int(fit["mask_T"].sum())
        print(
            f"  Guards: excitation_ratio={args.excitation_ratio:g}, "
            f"r2_min={args.r2_min:g}  ->  "
            f"zeroed {n_masked_F} c^F and {n_masked_T} c^T entries"
        )
        print(f"  {'link':24s}  "
              f"{'cF_x':>10s} {'cF_y':>10s} {'cF_z':>10s}   "
              f"{'R2_Fx':>6s} {'R2_Fy':>6s} {'R2_Fz':>6s}")
        for i, name in enumerate(link_names):
            print(
                f"  {name:24s}  "
                f"{fit['cF'][i, 0]:10.3g} {fit['cF'][i, 1]:10.3g} "
                f"{fit['cF'][i, 2]:10.3g}   "
                f"{fit['R2_F'][i, 0]:6.2f} {fit['R2_F'][i, 1]:6.2f} "
                f"{fit['R2_F'][i, 2]:6.2f}"
            )

        # Diagnostic plots: URDF-frame F and T, truth vs reconstruction.
        times_fit = t_fit[fit_mask]
        _plot_fit_reconstruction(
            times_fit, fit["F_urdf"], fit["F_hat"], link_names,
            title=(r"Per-link URDF-frame force: "
                   r"$F^{\mathrm{urdf}}_\alpha$ vs "
                   r"$\mu\,c^F_\alpha\,v_\alpha|v_\alpha|$"),
            ylabel=r"$F^{\mathrm{urdf}}$ [N]",
            save_path=os.path.join(args.data_dir, "farms_fit_force.png"),
        )
        _plot_fit_reconstruction(
            times_fit, fit["T_urdf"], fit["T_hat"], link_names,
            title=(r"Per-link URDF-frame torque: "
                   r"$\tau^{\mathrm{urdf}}_\alpha$ vs "
                   r"$c^T_\alpha\,\omega_\alpha|\omega_\alpha|$"),
            ylabel=r"$\tau^{\mathrm{urdf}}$ [N m]",
            save_path=os.path.join(args.data_dir, "farms_fit_torque.png"),
        )

        # Dump full fit arrays for downstream use.
        np.savez(
            os.path.join(args.data_dir, "farms_drag_fit.npz"),
            times=times_fit,
            fit_t_start=float(args.fit_t_start),
            fit_t_end=float(t_end),
            cF=fit["cF"], cT=fit["cT"],
            R2_F=fit["R2_F"], R2_T=fit["R2_T"],
            F_urdf=fit["F_urdf"], T_urdf=fit["T_urdf"],
            V_urdf=fit["V_urdf"], Omega_urdf=fit["Omega_urdf"],
            F_hat=fit["F_hat"], T_hat=fit["T_hat"],
            viscosity=args.farms_viscosity,
            link_names=np.array(link_names, dtype=object),
        )
        print(f"Wrote {yaml_path} and farms_fit_{{force,torque}}.png.")


if __name__ == "__main__":
    main()
