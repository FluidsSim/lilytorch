"""Generate the jellyfish mesh and inertial properties from the analytical SDF.

This script implements the analytical jellyfish SDF from WaterLily-jl's
``examples/ThreeD_Jelly.jl`` (https://github.com/WaterLily-jl/WaterLily-Examples):

    sphere(x) = abs(|x| - R) - t          # thin spherical shell, thickness 2*t
    plane(x)  = x[3] - h                  # half-space  z >= h
    body(x)   = max(sphere(x), -plane(x)) # sphere ∩ {z > h}   (CSG "A - B")

i.e. the body is the upper cap of a hollow spherical shell — the
characteristic jellyfish bell.

The script

1. evaluates the analytical SDF on a regular 3-D grid,
2. extracts a watertight triangle surface with marching cubes,
3. writes it as an STL mesh ready to be referenced from an SDF XML,
4. estimates mass and full inertia tensor of the *bell volume*
   (treated as a uniformly-dense rigid body) by Monte-Carlo integration
   on the same SDF.  Those values are the inputs to Newton's equations of
   motion that MuJoCo / FARMS will integrate for the single free body.

Physical parameters (defaults)
------------------------------
* ``R``     : outer-ish bell radius (metres, default 0.05)
* ``t``     : shell half-thickness relative to R, scaled from WaterLily
  where ``t_grid = 1`` and ``R_grid = 2L/3`` with ``L = 2^5 = 32`` cells,
  i.e. ``t/R = 3/(2*L) = 3/64``.  With ``R = 0.05 m`` this gives
  ``t ≈ 2.34e-3 m`` (shell thickness ≈ 4.7 mm).
* ``h``     : plane height relative to R.  In WaterLily, ``h = 4L - 2R``
  in grid units and the sphere centre is mapped to ``(0, 0, h)``;
  relative to the sphere centre the plane therefore sits at
  ``z_local = 0``.  We place the sphere centre at the origin and the
  plane at ``z = 0`` — this yields the canonical upper-hemispherical
  bell used in the reference.
* ``rho_body`` : uniform density (kg/m^3, default 1025, close to sea
  water so the bell is quasi-neutrally buoyant).

Run
---
    python -m lilytorch.farms_examples.sdfs.jellyfish.generate_jellyfish_mesh

which refreshes ``meshes/jellyfish.stl`` next to this file and prints the
inertial properties used in ``jellyfish.sdf``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from skimage import measure
from stl import mesh as stl_mesh


# ---------------------------------------------------------------------------
# Analytical SDF (direct port of WaterLily ThreeD_Jelly.jl)
# ---------------------------------------------------------------------------

def sdf_sphere_shell(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     R: float, t: float) -> np.ndarray:
    """Signed distance to a spherical shell of mean radius ``R`` and half
    thickness ``t`` (inside of the shell is negative)."""
    r = np.sqrt(x * x + y * y + z * z)
    return np.abs(r - R) - t


def sdf_plane_upper(z: np.ndarray, h: float) -> np.ndarray:
    """SDF of the half-space ``z >= h`` (inside is negative)."""
    return z - h


def sdf_jellyfish(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  R: float, t: float, h: float) -> np.ndarray:
    """Analytical jellyfish SDF: spherical shell ∩ {z > h}.

    Matches ``body = sphere - plane`` from ThreeD_Jelly.jl, with the plane
    defined as ``z - h`` (inside = ``z < h``) and the CSG "A minus B"
    implemented as ``max(sdf_A, -sdf_B)``.
    """
    return np.maximum(sdf_sphere_shell(x, y, z, R, t),
                      -sdf_plane_upper(z, h))


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

def sample_sdf_grid(R: float, t: float, h: float,
                    resolution: int = 128, padding: float = 1.25):
    """Sample the analytical SDF on a cube grid large enough to enclose
    the bell, returning ``(sdf_values, spacing, origin)``."""
    half = padding * (R + t)
    xs = np.linspace(-half, half, resolution)
    ys = np.linspace(-half, half, resolution)
    zs = np.linspace(-half, half, resolution)

    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    origin = np.array([xs[0], ys[0], zs[0]])

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    sdf = sdf_jellyfish(X, Y, Z, R, t, h)
    return sdf, spacing, origin


def extract_mesh(sdf: np.ndarray, spacing, origin: np.ndarray):
    """Run marching cubes on the SDF and return a watertight triangle
    mesh (vertices in world coordinates, faces as index triples)."""
    verts, faces, _normals, _values = measure.marching_cubes(
        sdf, level=0.0, spacing=spacing,
    )
    verts = verts + origin
    return verts, faces


def write_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype)
    tri["vectors"] = verts[faces]
    m = stl_mesh.Mesh(tri)
    m.save(path)


# ---------------------------------------------------------------------------
# Inertia from the analytical SDF (Monte-Carlo)
# ---------------------------------------------------------------------------

def inertial_properties(R: float, t: float, h: float,
                        rho_body: float,
                        n_samples: int = 400_000,
                        seed: int = 0):
    """Estimate (mass, com, inertia_tensor) of the jellyfish bell treated
    as a uniformly-dense solid shell, using rejection sampling of the
    analytical SDF.

    The bell is bounded by ``r in [R - t, R + t]`` and ``z >= h``, so we
    draw uniform samples in the bounding sphere (easy analytic sampling)
    and accept those with ``sdf <= 0``.
    """
    rng = np.random.default_rng(seed)
    r_box = R + t
    # Uniform samples inside the bounding sphere.
    u = rng.uniform(0.0, 1.0, n_samples)
    costh = rng.uniform(-1.0, 1.0, n_samples)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    rr = r_box * np.cbrt(u)
    sinth = np.sqrt(np.clip(1.0 - costh * costh, 0.0, 1.0))
    x = rr * sinth * np.cos(phi)
    y = rr * sinth * np.sin(phi)
    z = rr * costh

    sdf = sdf_jellyfish(x, y, z, R, t, h)
    inside = sdf <= 0.0
    n_in = int(inside.sum())
    if n_in == 0:
        raise RuntimeError("No MC samples fell inside the body — bad R/t/h?")
    v_box = (4.0 / 3.0) * np.pi * r_box ** 3
    volume = v_box * (n_in / n_samples)
    mass = rho_body * volume

    xi, yi, zi = x[inside], y[inside], z[inside]
    com = np.array([xi.mean(), yi.mean(), zi.mean()])

    dx = xi - com[0]
    dy = yi - com[1]
    dz = zi - com[2]
    dm = mass / n_in  # equal-mass elements

    Ixx = np.sum(dm * (dy * dy + dz * dz))
    Iyy = np.sum(dm * (dx * dx + dz * dz))
    Izz = np.sum(dm * (dx * dx + dy * dy))
    Ixy = -np.sum(dm * dx * dy)
    Ixz = -np.sum(dm * dx * dz)
    Iyz = -np.sum(dm * dy * dz)
    inertia = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])
    return mass, com, inertia, volume


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_R = 0.05                              # [m] bell mean radius
# t/R = 3/(2*L) from WaterLily (L=32, R=2L/3), kept proportional
DEFAULT_T = DEFAULT_R * 3.0 / (2.0 * 32.0)    # half-thickness ≈ 2.34e-3 m
DEFAULT_H = 0.0                               # plane at the sphere centre
DEFAULT_RHO = 1025.0                          # [kg/m^3] near sea water
DEFAULT_RESOLUTION = 128
DEFAULT_MC_SAMPLES = 400_000

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STL = os.path.join(HERE, "meshes", "jellyfish.stl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--R", type=float, default=DEFAULT_R)
    parser.add_argument("--t", type=float, default=DEFAULT_T)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--out", default=DEFAULT_STL)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[jellyfish] R={args.R} m  t={args.t:.5f} m  h={args.h} m  "
          f"rho={args.rho} kg/m^3")
    print(f"[jellyfish] sampling SDF on a {args.resolution}^3 grid...")
    sdf, spacing, origin = sample_sdf_grid(
        args.R, args.t, args.h, resolution=args.resolution,
    )
    verts, faces = extract_mesh(sdf, spacing, origin)
    print(f"[jellyfish] mesh: {len(verts)} vertices, {len(faces)} triangles")

    write_stl(args.out, verts, faces)
    print(f"[jellyfish] wrote STL to {args.out}")

    mass, com, inertia, volume = inertial_properties(
        args.R, args.t, args.h, args.rho, n_samples=args.samples,
    )
    print(f"[jellyfish] volume = {volume:.6e} m^3")
    print(f"[jellyfish] mass   = {mass:.6e} kg")
    print(f"[jellyfish] com    = {com}")
    print(f"[jellyfish] inertia (about COM, kg.m^2):\n{inertia}")

    return mass, com, inertia


if __name__ == "__main__":
    main()
