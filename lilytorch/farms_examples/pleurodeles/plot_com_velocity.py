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
    return times, vel_lin, vel_ang, link_names, timestep


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
                       it_max: int, save_path: str):
    v_com = np.mean(vel_lin, axis=1)  # (nt, 3)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for a, lab in enumerate("xyz"):
        ax.plot(times[:it_max], v_com[:it_max, a], label=f"v_{lab}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("COM Velocity (mean over links)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_per_link_grid(times: np.ndarray, coeff: np.ndarray, it_max: int,
                        link_names: list, title: str, ylabel: str,
                        save_path: str, ylim: tuple | None = None):
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
    args = parser.parse_args()

    out_dir    = os.path.join(args.data_dir, "output")
    params     = _load_parameters_yaml(os.path.join(args.data_dir,
                                                    "parameters.yaml"))
    # Prefer the run-local SDF cache; fall back to the source example folder
    # (``lilytorch/farms_examples/pleurodeles/interp_data_3d``) where the
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

    times, vel_lin, vel_ang, link_names_sim, _ = _load_link_sensors(sim_h5)
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

    _plot_com_velocity(times, vel_lin, it_max,
                       os.path.join(args.data_dir, "com_velocity_plot.png"))

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

    F = (drags["viscous_drags"][:n_b, :, :it_max]
         + drags["pressure_drags"][:n_b, :, :it_max])
    T = (drags["viscous_torques"][:n_b, :, :it_max]
         + drags["pressure_torques"][:n_b, :, :it_max])

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
    )
    _plot_per_link_grid(
        times, C_T, it_max, link_names,
        title=r"Per-link angular drag coefficients $C_{T,\alpha}=T_\alpha/"
              r"(\frac{1}{2} \rho\, A_w L^2\, \Omega_\alpha|\Omega_\alpha|)$",
        ylabel=r"$C_T$",
        save_path=os.path.join(args.data_dir, "link_torque_coefficients.png"),
        ylim=(-args.cd_clip, args.cd_clip) if args.cd_clip is not None else None,
    )

    np.savez(
        os.path.join(args.data_dir, "link_drag_coefficients.npz"),
        times=times[:it_max],
        F=F, T=T, U=U, Omega=Omega,
        C_D=C_D, C_T=C_T,
        wetted_area=A_w, body_length=L, rho=rho,
        link_names=np.array(link_names, dtype=object),
    )
    print(f"Wrote plots and arrays under {args.data_dir}")


if __name__ == "__main__":
    main()
