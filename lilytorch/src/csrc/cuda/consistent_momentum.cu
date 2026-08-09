// =====================================================================
//  consistent_momentum.cu — Nangia et al. (2019) consistent conservative
//  mass/momentum transport for the high-density-ratio two-phase step.
//
//  Op: ``lilytorch_kernels::consistent_momentum_3d`` (schema in ops.cpp).
//
//  WHY.  The default step advects VELOCITY (semi-Lagrangian or flux) and
//  rebuilds rho from an independently transported alpha.  The momentum a
//  face carries is then inconsistent with the mass that crossed it, which
//  is stable only to ~100:1 density ratio; water/air here is 816:1.  This
//  op transports rho*u with the SAME upwind mass flux that evolves rho and
//  recovers u = rho*u / rho_new, so the 816:1 jump cannot amplify.  It
//  replaces BOTH the velocity advection AND the Weymouth & Yue VOF sweep —
//  the interface is carried by the density itself.
//
//  Gravity is a rho*g BODY FORCE here, not a velocity pre-kick, so the mass
//  flux rides on the (nearly divergence-free) projected velocity.  The
//  caller MUST suppress the predictor-side dt*g.
//
//  Single forward-Euler pass (Nangia n_cycles=1): the advecting, advected
//  and momentum-carrying velocities are all u^n.  This is the default the
//  deleted python path used; the fixed-point cycles are not ported.
//
//  ``flux_scheme`` picks the reconstruction of the SHARED mass flux:
//    0 = first-order donor cell (the recovered reference).  Stable, but its
//        numerical diffusion (~|u|h/2) damps resolved surface waves.
//    1 = Weymouth & Yue Courant-corrected van-Leer donor for the density.
//        Algebraically the SAME face value ``cvof_sweep`` uses: rho is affine
//        in alpha and the van-Leer slope is linear in it, so limiting rho and
//        limiting alpha agree exactly.  The interface keeps the stock VOF's
//        sharpness while the flux stays shared with the momentum.
//    2 = as 1, plus a van-Leer MUSCL reconstruction of the ADVECTED velocity
//        (removes the momentum's own first-order diffusion).
//  Level >=1 is bounded for CFL <= 1 per direction; this update is unsplit, so
//  in 3-D keep the per-direction Courant number comfortably below 1/3.
//
//  MAC convention: q[k] along axis d is the face LEFT of cell k, so
//  F[d][0] is outside the domain and is defined to be 0 (matching the
//  reference's ``torch.zeros_like`` + ``[1:]`` assignment).
//
//  Semantics — reproduced index-for-index from the recovered reference
//  ``_consistent_advect`` (see _dbg/stillwater_rigs/
//  nangia_consistent_momentum_RECOVERED.py):
//
//    rho[c]      = alpha[c]*(rho_w - rho_a) + rho_a
//    F[d][k]     = u[d][k] * (u[d][k] >= 0 ? rho[k-1] : rho[k])   (k>=1)
//    rho_new[c]  = max(rho[c] - (dt/h) * sum_d (F[d][c+1] - F[d][c]), rho_a)
//
//    for component ci, at an interior face k (interior on EVERY axis):
//      rfi   = 0.5*(rho[k-1] + rho[k])            (face density, axis ci)
//      mu    = rfi * u[ci][k]
//      d==ci (self, cell centred):
//        Mc[m]  = 0.5*(F[ci][m] + F[ci][m+1])
//        ui[m]  = Mc[m] >= 0 ? u[ci][m] : u[ci][m+1]
//        dmu   += Mc[k]*ui[k] - Mc[k-1]*ui[k-1]
//      d!=ci (cross, on the (ci,d) edge):
//        Me[q]  = 0.5*(F[d][q-1] + F[d][q])       (average along axis ci)
//        Med[m] = Me at d-index m+1
//        vid[m] = Med[m] >= 0 ? u[ci][m] : u[ci][m+1]   (along axis d)
//        dmu   += Med[jd]*vid[jd] - Med[jd-1]*vid[jd-1]
//      mu    += -(dt/h)*dmu + dt*g[ci]*rfi
//      out    = mu / (0.5*(rho_new[k-1] + rho_new[k]))
//
//  Non-interior points copy u[ci] through, matching the reference's
//  ``vi = u_start[i].clone()`` followed by the interior-only assignment.
//
//  ``alpha_out`` must be DISTINCT from ``alpha``: the op schema declares
//  ``alpha`` an immutable input.  (The launch order would tolerate aliasing --
//  the alpha sync runs after the momentum kernels -- but relying on that would
//  break the dispatcher's aliasing contract.)
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace lilytorch_kernels {

static __host__ __device__ __forceinline__ int cdiv_cm(int a, int b) {
    return (a + b - 1) / b;
}

// Strided 3-D accessor (velocity components are row views of a stacked
// tensor, so nothing here may assume contiguity).
template <typename T>
struct Acc3 {
    const T* __restrict__ p;
    int64_t s0, s1, s2;
    __device__ __forceinline__ T operator()(int i, int j, int k) const {
        return p[(int64_t)i * s0 + (int64_t)j * s1 + (int64_t)k * s2];
    }
};

template <typename T>
struct Acc3W {
    T* __restrict__ p;
    int64_t s0, s1, s2;
    __device__ __forceinline__ T& operator()(int i, int j, int k) const {
        return p[(int64_t)i * s0 + (int64_t)j * s1 + (int64_t)k * s2];
    }
};

// Step one index along axis d.
__device__ __forceinline__ void step_axis(int d, int s, int& i, int& j, int& k) {
    if (d == 0)      i += s;
    else if (d == 1) j += s;
    else             k += s;
}
__device__ __forceinline__ int axis_idx(int d, int i, int j, int k) {
    return (d == 0) ? i : ((d == 1) ? j : k);
}

template <typename T>
__device__ __forceinline__ T rho_at(const Acc3<T>& al, int i, int j, int k,
                                    T drho, T rho_a) {
    return al(i, j, k) * drho + rho_a;
}

__device__ __forceinline__ int clampi(int v, int hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

// van-Leer (harmonic) limited slope; 0 at extrema / sign changes.  Identical
// to ``cv_vleer`` in cvof_sweep.cu so the two schemes agree exactly.
template <typename T>
__device__ __forceinline__ T cm_vleer(T db, T df) {
    T sum   = db + df;
    T denom = (sum == T(0)) ? T(1) : sum;
    T s     = T(2) * db * df / denom;
    return (db * df > T(0)) ? s : T(0);
}

// Read q at (i,j,k) shifted by `off` along axis d, EDGE-CLAMPED (the
// Neumann-consistent read cvof_sweep uses).
template <typename T>
__device__ __forceinline__ T at_off(const Acc3<T>& q, int i, int j, int k,
                                    int d, int off, int Nd) {
    int t = clampi(axis_idx(d, i, j, k) + off, Nd - 1);
    int ii = i, jj = j, kk = k;
    if (d == 0)      ii = t;
    else if (d == 1) jj = t;
    else             kk = t;
    return q(ii, jj, kk);
}

// van-Leer MUSCL face value at the interface between axis-d indices m and m+1,
// where (i,j,k) sits at index m.  `from_left` selects the donor side.
template <typename T>
__device__ __forceinline__ T muscl_between(const Acc3<T>& q, int i, int j, int k,
                                           int d, int Nd, bool from_left) {
    if (from_left) {
        T c  = at_off<T>(q, i, j, k, d,  0, Nd);
        T m1 = at_off<T>(q, i, j, k, d, -1, Nd);
        T p1 = at_off<T>(q, i, j, k, d,  1, Nd);
        return c + T(0.5) * cm_vleer<T>(c - m1, p1 - c);
    }
    T c  = at_off<T>(q, i, j, k, d,  1, Nd);
    T m1 = at_off<T>(q, i, j, k, d,  0, Nd);
    T p1 = at_off<T>(q, i, j, k, d,  2, Nd);
    return c - T(0.5) * cm_vleer<T>(c - m1, p1 - c);
}

// Mass flux through the d-face LEFT of cell (i,j,k).  Zero at index 0 on
// axis d (that face is outside the domain).
template <typename T>
__device__ __forceinline__ T Fface(
    int d, int i, int j, int k,
    const Acc3<T>& al, const Acc3<T>* uu, T drho, T rho_a,
    int scheme, T dtdh, int Nd)
{
    if (axis_idx(d, i, j, k) == 0) return T(0);
    T ud = uu[d](i, j, k);
    if (scheme == 0) {                                   // donor cell
        int im = i, jm = j, km = k;
        step_axis(d, -1, im, jm, km);
        T r_up = (ud >= T(0)) ? rho_at<T>(al, im, jm, km, drho, rho_a)
                              : rho_at<T>(al, i, j, k, drho, rho_a);
        return ud * r_up;
    }
    // Weymouth & Yue Courant-corrected van-Leer donor extrapolation, the same
    // face value cvof_sweep builds (see cv_face there).
    T C   = ud * dtdh;
    T ak  = at_off<T>(al, i, j, k, d,  0, Nd) * drho + rho_a;
    T am1 = at_off<T>(al, i, j, k, d, -1, Nd) * drho + rho_a;
    T am2 = at_off<T>(al, i, j, k, d, -2, Nd) * drho + rho_a;
    T ap1 = at_off<T>(al, i, j, k, d,  1, Nd) * drho + rho_a;
    T f_pos = am1 + T(0.5) * (T(1) - C) * cm_vleer<T>(am1 - am2, ak - am1);
    T f_neg = ak  - T(0.5) * (T(1) + C) * cm_vleer<T>(ak - am1, ap1 - ak);
    return ud * ((C >= T(0)) ? f_pos : f_neg);
}

// ── kernel 0: materialise the SHARED mass flux once ──────────────────
// The density update and all three momentum components read the same F, and
// each thread would otherwise re-evaluate it ~39 times per cell (the limited
// reconstruction costs 4 alpha loads apiece).  Computing it once into a
// (3, N0, N1, N2) scratch turns that into 3 stores + ~39 plain loads.
template <typename scalar_t>
__global__ void cm_flux_kernel(
    Acc3<scalar_t>  al,
    Acc3<scalar_t>  u0, Acc3<scalar_t> u1, Acc3<scalar_t> u2,
    Acc3W<scalar_t> f0, Acc3W<scalar_t> f1, Acc3W<scalar_t> f2,
    int N0, int N1, int N2,
    scalar_t drho, scalar_t rho_a, scalar_t dtdh, int scheme)
{
    int k = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int j = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;
    if (i >= N0 || j >= N1 || k >= N2) return;

    const Acc3<scalar_t> uu[3] = {u0, u1, u2};
    const int Nd[3] = {N0, N1, N2};
    f0(i, j, k) = Fface<scalar_t>(0, i, j, k, al, uu, drho, rho_a, scheme, dtdh, Nd[0]);
    f1(i, j, k) = Fface<scalar_t>(1, i, j, k, al, uu, drho, rho_a, scheme, dtdh, Nd[1]);
    f2(i, j, k) = Fface<scalar_t>(2, i, j, k, al, uu, drho, rho_a, scheme, dtdh, Nd[2]);
}

// ── kernel 1: conservative density update ────────────────────────────
template <typename scalar_t>
__global__ void cm_density_kernel(
    Acc3<scalar_t>  al,
    Acc3<scalar_t>  f0, Acc3<scalar_t> f1, Acc3<scalar_t> f2,
    Acc3W<scalar_t> rho_new,
    int N0, int N1, int N2,
    scalar_t drho, scalar_t rho_a, scalar_t dtdh)
{
    int k = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int j = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;
    if (i >= N0 || j >= N1 || k >= N2) return;

    const Acc3<scalar_t> fl[3] = {f0, f1, f2};
    scalar_t r = rho_at<scalar_t>(al, i, j, k, drho, rho_a);
    scalar_t rn = r;

    bool interior = (i >= 1 && i < N0 - 1) && (j >= 1 && j < N1 - 1)
                 && (k >= 1 && k < N2 - 1);
    if (interior) {
        scalar_t div = scalar_t(0);
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            int ip = i, jp = j, kp = k;
            step_axis(d, 1, ip, jp, kp);
            div += fl[d](ip, jp, kp) - fl[d](i, j, k);
        }
        rn = r - dtdh * div;
        if (rn < rho_a) rn = rho_a;                 // reference: clamp_min(rho_a)
    }
    rho_new(i, j, k) = rn;
}

// ── kernel 2: consistent momentum transport for one component ────────
template <typename scalar_t>
__global__ void cm_momentum_kernel(
    int ci,
    Acc3<scalar_t>  al,
    Acc3<scalar_t>  u0, Acc3<scalar_t> u1, Acc3<scalar_t> u2,
    Acc3<scalar_t>  f0, Acc3<scalar_t> f1, Acc3<scalar_t> f2,
    Acc3<scalar_t>  rho_new,
    Acc3W<scalar_t> out,
    int N0, int N1, int N2,
    scalar_t drho, scalar_t rho_a, scalar_t dtdh, scalar_t dt_g, int scheme)
{
    int k = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int j = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;
    if (i >= N0 || j >= N1 || k >= N2) return;

    const Acc3<scalar_t> uu[3] = {u0, u1, u2};
    const Acc3<scalar_t> fl[3] = {f0, f1, f2};
    const Acc3<scalar_t>& uc = uu[ci];
    const int Nd[3] = {N0, N1, N2};

    bool interior = (i >= 1 && i < N0 - 1) && (j >= 1 && j < N1 - 1)
                 && (k >= 1 && k < N2 - 1);
    if (!interior) {                                // reference: u_start.clone()
        out(i, j, k) = uc(i, j, k);
        return;
    }

    // face density on axis ci: 0.5*(rho[k-1] + rho[k])
    int il = i, jl = j, kl = k;
    step_axis(ci, -1, il, jl, kl);
    scalar_t rfi = scalar_t(0.5) * (rho_at<scalar_t>(al, il, jl, kl, drho, rho_a)
                                  + rho_at<scalar_t>(al, i,  j,  k,  drho, rho_a));
    scalar_t mu = rfi * uc(i, j, k);

    scalar_t dmu = scalar_t(0);
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        if (d == ci) {
            // self-advection, cell-centred mass flux
            int ip = i, jp = j, kp = k;  step_axis(ci, 1, ip, jp, kp);
            int im = i, jm = j, km = k;  step_axis(ci, -1, im, jm, km);

            scalar_t F_m = fl[ci](im, jm, km);
            scalar_t F_c = fl[ci](i,  j,  k);
            scalar_t F_p = fl[ci](ip, jp, kp);

            scalar_t Mc_hi = scalar_t(0.5) * (F_c + F_p);
            scalar_t Mc_lo = scalar_t(0.5) * (F_m + F_c);
            scalar_t u_hi, u_lo;
            if (scheme < 2) {
                u_hi = (Mc_hi >= scalar_t(0)) ? uc(i, j, k) : uc(ip, jp, kp);
                u_lo = (Mc_lo >= scalar_t(0)) ? uc(im, jm, km) : uc(i, j, k);
            } else {
                // the "cells" of this reconstruction are the ci-faces
                u_hi = muscl_between<scalar_t>(uc, i,  j,  k,  ci, Nd[ci],
                                               Mc_hi >= scalar_t(0));
                u_lo = muscl_between<scalar_t>(uc, im, jm, km, ci, Nd[ci],
                                               Mc_lo >= scalar_t(0));
            }
            dmu += Mc_hi * u_hi - Mc_lo * u_lo;
        } else {
            // cross-advection on the (ci, d) edge: average F[d] along ci
            int ci_m_i = i, ci_m_j = j, ci_m_k = k;
            step_axis(ci, -1, ci_m_i, ci_m_j, ci_m_k);

            // Med[jd] = Me at d-index jd+1
            int hi_i = i, hi_j = j, hi_k = k;   step_axis(d, 1, hi_i, hi_j, hi_k);
            int hm_i = ci_m_i, hm_j = ci_m_j, hm_k = ci_m_k;
            step_axis(d, 1, hm_i, hm_j, hm_k);
            scalar_t Me_hi = scalar_t(0.5)
                * (fl[d](hm_i, hm_j, hm_k) + fl[d](hi_i, hi_j, hi_k));

            // Med[jd-1] = Me at d-index jd
            scalar_t Me_lo = scalar_t(0.5)
                * (fl[d](ci_m_i, ci_m_j, ci_m_k) + fl[d](i, j, k));
            int lo_i = i, lo_j = j, lo_k = k;   step_axis(d, -1, lo_i, lo_j, lo_k);

            scalar_t v_hi, v_lo;
            if (scheme < 2) {
                v_hi = (Me_hi >= scalar_t(0)) ? uc(i, j, k) : uc(hi_i, hi_j, hi_k);
                v_lo = (Me_lo >= scalar_t(0)) ? uc(lo_i, lo_j, lo_k) : uc(i, j, k);
            } else {
                v_hi = muscl_between<scalar_t>(uc, i,    j,    k,    d, Nd[d],
                                               Me_hi >= scalar_t(0));
                v_lo = muscl_between<scalar_t>(uc, lo_i, lo_j, lo_k, d, Nd[d],
                                               Me_lo >= scalar_t(0));
            }
            dmu += Me_hi * v_hi - Me_lo * v_lo;
        }
    }

    mu -= dtdh * dmu;
    mu += dt_g * rfi;                               // dt * g[ci] * rho_face^n

    scalar_t rfi2 = scalar_t(0.5) * (rho_new(il, jl, kl) + rho_new(i, j, k));
    out(i, j, k) = mu / rfi2;
}

// ── kernel 3: sync alpha from the evolved density ────────────────────
template <typename scalar_t>
__global__ void cm_alpha_kernel(
    Acc3<scalar_t>  rho_new,
    Acc3W<scalar_t> alpha_out,
    int N0, int N1, int N2,
    scalar_t inv_drho, scalar_t rho_a)
{
    int k = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    int j = (int)(blockIdx.y * blockDim.y) + (int)threadIdx.y;
    int i = (int)(blockIdx.z * blockDim.z) + (int)threadIdx.z;
    if (i >= N0 || j >= N1 || k >= N2) return;
    scalar_t a = (rho_new(i, j, k) - rho_a) * inv_drho;
    a = (a < scalar_t(0)) ? scalar_t(0) : ((a > scalar_t(1)) ? scalar_t(1) : a);
    alpha_out(i, j, k) = a;
}

// ── C++ wrapper ──────────────────────────────────────────────────────

template <typename T>
static Acc3<T> mk_r(const at::Tensor& t) {
    return Acc3<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2)};
}
template <typename T>
static Acc3W<T> mk_w(at::Tensor& t) {
    return Acc3W<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2)};
}

static void consistent_momentum_3d_cuda(
    const at::Tensor& alpha,
    const at::Tensor& u, const at::Tensor& v, const at::Tensor& w,
    at::Tensor& rho_new, at::Tensor& flux,
    at::Tensor& uo, at::Tensor& vo, at::Tensor& wo,
    at::Tensor& alpha_out,
    double rho_water, double rho_air, double dt, double h,
    double gx, double gy, double gz, int64_t flux_scheme)
{
    TORCH_CHECK(flux_scheme >= 0 && flux_scheme <= 2,
                "consistent_momentum_3d: flux_scheme must be 0, 1 or 2, got ",
                flux_scheme);
    TORCH_CHECK(alpha.dim() == 3, "consistent_momentum_3d: alpha must be 3-D");
    TORCH_CHECK(alpha.is_cuda() && u.is_cuda() && rho_new.is_cuda(),
                "consistent_momentum_3d: all tensors must be CUDA");
    TORCH_CHECK(rho_water != rho_air,
                "consistent_momentum_3d: rho_water must differ from rho_air");
    // NB: compare data_ptr, not is_same -- the dispatcher hands the impl
    // shallow Tensor copies, so is_same() is false even for the same storage.
    TORCH_CHECK(alpha.data_ptr() != alpha_out.data_ptr(),
                "consistent_momentum_3d: alpha_out must not alias alpha "
                "(alpha is declared an immutable input by the schema)");

    TORCH_CHECK(flux.dim() == 4 && flux.size(0) == 3,
                "consistent_momentum_3d: flux scratch must be (3, N0, N1, N2)");
    int N0 = (int)alpha.size(0), N1 = (int)alpha.size(1), N2 = (int)alpha.size(2);
    if (N0 <= 0 || N1 <= 0 || N2 <= 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int BX = 32, BY = 4, BZ = 2;
    dim3 blk(BX, BY, BZ);
    dim3 grd(cdiv_cm(N2, BX), cdiv_cm(N1, BY), cdiv_cm(N0, BZ));

    AT_DISPATCH_FLOATING_TYPES(alpha.scalar_type(), "consistent_momentum_3d", [&] {
        auto al = mk_r<scalar_t>(alpha);
        auto a0 = mk_r<scalar_t>(u), a1 = mk_r<scalar_t>(v), a2 = mk_r<scalar_t>(w);
        auto rn_w = mk_w<scalar_t>(rho_new);
        auto rn_r = mk_r<scalar_t>(rho_new);
        at::Tensor fx[3] = {flux[0], flux[1], flux[2]};
        auto fw0 = mk_w<scalar_t>(fx[0]), fw1 = mk_w<scalar_t>(fx[1]),
             fw2 = mk_w<scalar_t>(fx[2]);
        auto fr0 = mk_r<scalar_t>(fx[0]), fr1 = mk_r<scalar_t>(fx[1]),
             fr2 = mk_r<scalar_t>(fx[2]);

        const scalar_t drho  = (scalar_t)(rho_water - rho_air);
        const scalar_t rho_a = (scalar_t)rho_air;
        const scalar_t dtdh  = (scalar_t)(dt / h);

        cm_flux_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            al, a0, a1, a2, fw0, fw1, fw2, N0, N1, N2,
            drho, rho_a, dtdh, (int)flux_scheme);

        cm_density_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            al, fr0, fr1, fr2, rn_w, N0, N1, N2, drho, rho_a, dtdh);

        at::Tensor* outs[3] = {&uo, &vo, &wo};
        const double gs[3] = {gx, gy, gz};
        for (int ci = 0; ci < 3; ++ci) {
            auto ow = mk_w<scalar_t>(*outs[ci]);
            cm_momentum_kernel<scalar_t><<<grd, blk, 0, stream>>>(
                ci, al, a0, a1, a2, fr0, fr1, fr2, rn_r, ow, N0, N1, N2,
                drho, rho_a, dtdh, (scalar_t)(dt * gs[ci]), (int)flux_scheme);
        }

        auto ao = mk_w<scalar_t>(alpha_out);
        cm_alpha_kernel<scalar_t><<<grd, blk, 0, stream>>>(
            rn_r, ao, N0, N1, N2, (scalar_t)(1.0 / (rho_water - rho_air)), rho_a);
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("consistent_momentum_3d", &consistent_momentum_3d_cuda);
}

}  // namespace lilytorch_kernels
