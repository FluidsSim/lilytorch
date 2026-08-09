// =====================================================================
//  consistent_momentum_cpu.cpp — CPU twin of the CUDA
//  ``consistent_momentum_3d`` kernel (Nangia et al. 2019 consistent
//  conservative mass/momentum transport).
//
//  Line-for-line the same arithmetic as csrc/cuda/consistent_momentum.cu,
//  so the two are bit-comparable on the same inputs.  See that file (and
//  the schema in ops.cpp) for the semantics and the MAC index convention.
//
//  Exists for two reasons: CPU-only runs, and — the important one — it is
//  the parity oracle for the CUDA kernel.  Kept deliberately naive
//  (at::parallel_for over the slowest axis, no blocking): it is not on any
//  hot path.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

namespace lilytorch_kernels {

namespace {

template <typename T>
struct View3 {
    T* p;
    int64_t s0, s1, s2;
    inline T& at(int i, int j, int k) const {
        return p[(int64_t)i * s0 + (int64_t)j * s1 + (int64_t)k * s2];
    }
};

inline void step_axis_cpu(int d, int s, int& i, int& j, int& k) {
    if (d == 0)      i += s;
    else if (d == 1) j += s;
    else             k += s;
}
inline int axis_idx_cpu(int d, int i, int j, int k) {
    return (d == 0) ? i : ((d == 1) ? j : k);
}

template <typename T>
inline T rho_at_cpu(const View3<T>& al, int i, int j, int k, T drho, T rho_a) {
    return al.at(i, j, k) * drho + rho_a;
}

inline int clampi_cpu(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

template <typename T>
inline T cm_vleer_cpu(T db, T df) {
    T sum   = db + df;
    T denom = (sum == T(0)) ? T(1) : sum;
    T s     = T(2) * db * df / denom;
    return (db * df > T(0)) ? s : T(0);
}

template <typename T>
inline T at_off_cpu(const View3<T>& q, int i, int j, int k,
                    int d, int off, int Nd) {
    int t = clampi_cpu(axis_idx_cpu(d, i, j, k) + off, Nd - 1);
    int ii = i, jj = j, kk = k;
    if (d == 0)      ii = t;
    else if (d == 1) jj = t;
    else             kk = t;
    return q.at(ii, jj, kk);
}

template <typename T>
inline T muscl_between_cpu(const View3<T>& q, int i, int j, int k,
                           int d, int Nd, bool from_left) {
    if (from_left) {
        T c  = at_off_cpu<T>(q, i, j, k, d,  0, Nd);
        T m1 = at_off_cpu<T>(q, i, j, k, d, -1, Nd);
        T p1 = at_off_cpu<T>(q, i, j, k, d,  1, Nd);
        return c + T(0.5) * cm_vleer_cpu<T>(c - m1, p1 - c);
    }
    T c  = at_off_cpu<T>(q, i, j, k, d,  1, Nd);
    T m1 = at_off_cpu<T>(q, i, j, k, d,  0, Nd);
    T p1 = at_off_cpu<T>(q, i, j, k, d,  2, Nd);
    return c - T(0.5) * cm_vleer_cpu<T>(c - m1, p1 - c);
}

// Mass flux through the d-face LEFT of cell (i,j,k); 0 at index 0 on axis d.
template <typename T>
inline T Fface_cpu(int d, int i, int j, int k,
                   const View3<T>& al, const View3<T>* uu, T drho, T rho_a,
                   int scheme, T dtdh, int Nd) {
    if (axis_idx_cpu(d, i, j, k) == 0) return T(0);
    T ud = uu[d].at(i, j, k);
    if (scheme == 0) {
        int im = i, jm = j, km = k;
        step_axis_cpu(d, -1, im, jm, km);
        T r_up = (ud >= T(0)) ? rho_at_cpu<T>(al, im, jm, km, drho, rho_a)
                              : rho_at_cpu<T>(al, i, j, k, drho, rho_a);
        return ud * r_up;
    }
    T C   = ud * dtdh;
    T ak  = at_off_cpu<T>(al, i, j, k, d,  0, Nd) * drho + rho_a;
    T am1 = at_off_cpu<T>(al, i, j, k, d, -1, Nd) * drho + rho_a;
    T am2 = at_off_cpu<T>(al, i, j, k, d, -2, Nd) * drho + rho_a;
    T ap1 = at_off_cpu<T>(al, i, j, k, d,  1, Nd) * drho + rho_a;
    T f_pos = am1 + T(0.5) * (T(1) - C) * cm_vleer_cpu<T>(am1 - am2, ak - am1);
    T f_neg = ak  - T(0.5) * (T(1) + C) * cm_vleer_cpu<T>(ak - am1, ap1 - ak);
    return ud * ((C >= T(0)) ? f_pos : f_neg);
}

template <typename scalar_t>
void consistent_momentum_3d_impl(
    const at::Tensor& alpha,
    const at::Tensor& u, const at::Tensor& v, const at::Tensor& w,
    at::Tensor& rho_new, at::Tensor& flux,
    at::Tensor& uo, at::Tensor& vo, at::Tensor& wo,
    at::Tensor& alpha_out,
    double rho_water, double rho_air, double dt, double h,
    double gx, double gy, double gz, int scheme)
{
    const int N0 = (int)alpha.size(0);
    const int N1 = (int)alpha.size(1);
    const int N2 = (int)alpha.size(2);

    auto mk = [](const at::Tensor& t) {
        return View3<scalar_t>{const_cast<scalar_t*>(t.data_ptr<scalar_t>()),
                               t.stride(0), t.stride(1), t.stride(2)};
    };
    View3<scalar_t> al = mk(alpha);
    View3<scalar_t> uu[3] = {mk(u), mk(v), mk(w)};
    View3<scalar_t> rn = mk(rho_new);
    at::Tensor fx0 = flux[0], fx1 = flux[1], fx2 = flux[2];
    View3<scalar_t> fl[3] = {mk(fx0), mk(fx1), mk(fx2)};
    View3<scalar_t> outs[3] = {mk(uo), mk(vo), mk(wo)};
    View3<scalar_t> ao = mk(alpha_out);

    const scalar_t drho  = (scalar_t)(rho_water - rho_air);
    const scalar_t rho_a = (scalar_t)rho_air;
    const scalar_t dtdh  = (scalar_t)(dt / h);
    const double   gs[3] = {gx, gy, gz};
    const int      Nd[3] = {N0, N1, N2};

    // ---- 0: materialise the SHARED mass flux once (see the .cu note) ----
    at::parallel_for(0, N0, 0, [&](int64_t lo, int64_t hi) {
        for (int i = (int)lo; i < (int)hi; ++i)
            for (int j = 0; j < N1; ++j)
                for (int k = 0; k < N2; ++k)
                    for (int d = 0; d < 3; ++d)
                        fl[d].at(i, j, k) = Fface_cpu<scalar_t>(
                            d, i, j, k, al, uu, drho, rho_a, scheme, dtdh, Nd[d]);
    });

    // ---- 1: conservative density update -----------------------------
    at::parallel_for(0, N0, 0, [&](int64_t lo, int64_t hi) {
        for (int i = (int)lo; i < (int)hi; ++i) {
            for (int j = 0; j < N1; ++j) {
                for (int k = 0; k < N2; ++k) {
                    scalar_t r = rho_at_cpu<scalar_t>(al, i, j, k, drho, rho_a);
                    scalar_t val = r;
                    bool interior = (i >= 1 && i < N0 - 1)
                                 && (j >= 1 && j < N1 - 1)
                                 && (k >= 1 && k < N2 - 1);
                    if (interior) {
                        scalar_t div = scalar_t(0);
                        for (int d = 0; d < 3; ++d) {
                            int ip = i, jp = j, kp = k;
                            step_axis_cpu(d, 1, ip, jp, kp);
                            div += fl[d].at(ip, jp, kp) - fl[d].at(i, j, k);
                        }
                        val = r - dtdh * div;
                        if (val < rho_a) val = rho_a;
                    }
                    rn.at(i, j, k) = val;
                }
            }
        }
    });

    // ---- 2: consistent momentum transport, per component ------------
    for (int ci = 0; ci < 3; ++ci) {
        const View3<scalar_t>& uc = uu[ci];
        View3<scalar_t>& out = outs[ci];
        const scalar_t dt_g = (scalar_t)(dt * gs[ci]);
        at::parallel_for(0, N0, 0, [&](int64_t lo, int64_t hi) {
            for (int i = (int)lo; i < (int)hi; ++i) {
                for (int j = 0; j < N1; ++j) {
                    for (int k = 0; k < N2; ++k) {
                        bool interior = (i >= 1 && i < N0 - 1)
                                     && (j >= 1 && j < N1 - 1)
                                     && (k >= 1 && k < N2 - 1);
                        if (!interior) { out.at(i, j, k) = uc.at(i, j, k); continue; }

                        int il = i, jl = j, kl = k;
                        step_axis_cpu(ci, -1, il, jl, kl);
                        scalar_t rfi = scalar_t(0.5)
                            * (rho_at_cpu<scalar_t>(al, il, jl, kl, drho, rho_a)
                             + rho_at_cpu<scalar_t>(al, i,  j,  k,  drho, rho_a));
                        scalar_t mu = rfi * uc.at(i, j, k);

                        scalar_t dmu = scalar_t(0);
                        for (int d = 0; d < 3; ++d) {
                            if (d == ci) {
                                int ip = i, jp = j, kp = k;
                                step_axis_cpu(ci, 1, ip, jp, kp);
                                int im = i, jm = j, km = k;
                                step_axis_cpu(ci, -1, im, jm, km);
                                scalar_t F_m = fl[ci].at(im, jm, km);
                                scalar_t F_c = fl[ci].at(i,  j,  k);
                                scalar_t F_p = fl[ci].at(ip, jp, kp);
                                scalar_t Mc_hi = scalar_t(0.5) * (F_c + F_p);
                                scalar_t Mc_lo = scalar_t(0.5) * (F_m + F_c);
                                scalar_t u_hi, u_lo;
                                if (scheme < 2) {
                                    u_hi = (Mc_hi >= scalar_t(0)) ? uc.at(i, j, k)
                                                                  : uc.at(ip, jp, kp);
                                    u_lo = (Mc_lo >= scalar_t(0)) ? uc.at(im, jm, km)
                                                                  : uc.at(i, j, k);
                                } else {
                                    u_hi = muscl_between_cpu<scalar_t>(uc, i, j, k, ci,
                                               Nd[ci], Mc_hi >= scalar_t(0));
                                    u_lo = muscl_between_cpu<scalar_t>(uc, im, jm, km, ci,
                                               Nd[ci], Mc_lo >= scalar_t(0));
                                }
                                dmu += Mc_hi * u_hi - Mc_lo * u_lo;
                            } else {
                                int cm_i = i, cm_j = j, cm_k = k;
                                step_axis_cpu(ci, -1, cm_i, cm_j, cm_k);

                                int hi_i = i, hi_j = j, hi_k = k;
                                step_axis_cpu(d, 1, hi_i, hi_j, hi_k);
                                int hm_i = cm_i, hm_j = cm_j, hm_k = cm_k;
                                step_axis_cpu(d, 1, hm_i, hm_j, hm_k);
                                scalar_t Me_hi = scalar_t(0.5)
                                    * (fl[d].at(hm_i, hm_j, hm_k)
                                     + fl[d].at(hi_i, hi_j, hi_k));
                                scalar_t Me_lo = scalar_t(0.5)
                                    * (fl[d].at(cm_i, cm_j, cm_k) + fl[d].at(i, j, k));
                                int lo_i = i, lo_j = j, lo_k = k;
                                step_axis_cpu(d, -1, lo_i, lo_j, lo_k);
                                scalar_t v_hi, v_lo;
                                if (scheme < 2) {
                                    v_hi = (Me_hi >= scalar_t(0)) ? uc.at(i, j, k)
                                                                  : uc.at(hi_i, hi_j, hi_k);
                                    v_lo = (Me_lo >= scalar_t(0)) ? uc.at(lo_i, lo_j, lo_k)
                                                                  : uc.at(i, j, k);
                                } else {
                                    v_hi = muscl_between_cpu<scalar_t>(uc, i, j, k, d,
                                               Nd[d], Me_hi >= scalar_t(0));
                                    v_lo = muscl_between_cpu<scalar_t>(uc, lo_i, lo_j, lo_k,
                                               d, Nd[d], Me_lo >= scalar_t(0));
                                }
                                dmu += Me_hi * v_hi - Me_lo * v_lo;
                            }
                        }
                        mu -= dtdh * dmu;
                        mu += dt_g * rfi;
                        scalar_t rfi2 = scalar_t(0.5)
                            * (rn.at(il, jl, kl) + rn.at(i, j, k));
                        out.at(i, j, k) = mu / rfi2;
                    }
                }
            }
        });
    }

    // ---- 3: sync alpha from the evolved density ---------------------
    const scalar_t inv_drho = (scalar_t)(1.0 / (rho_water - rho_air));
    at::parallel_for(0, N0, 0, [&](int64_t lo, int64_t hi) {
        for (int i = (int)lo; i < (int)hi; ++i) {
            for (int j = 0; j < N1; ++j) {
                for (int k = 0; k < N2; ++k) {
                    scalar_t a = (rn.at(i, j, k) - rho_a) * inv_drho;
                    a = (a < scalar_t(0)) ? scalar_t(0)
                                          : ((a > scalar_t(1)) ? scalar_t(1) : a);
                    ao.at(i, j, k) = a;
                }
            }
        }
    });
}

void consistent_momentum_3d_cpu(
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
    TORCH_CHECK(!alpha.device().is_cuda(),
                "consistent_momentum_3d CPU: expected CPU tensors");
    TORCH_CHECK(alpha.dim() == 3,
                "consistent_momentum_3d: alpha must be 3-D");
    TORCH_CHECK(rho_water != rho_air,
                "consistent_momentum_3d: rho_water must differ from rho_air");
    // NB: compare data_ptr, not is_same -- the dispatcher hands the impl
    // shallow Tensor copies, so is_same() is false even for the same storage.
    TORCH_CHECK(alpha.data_ptr() != alpha_out.data_ptr(),
                "consistent_momentum_3d: alpha_out must not alias alpha "
                "(alpha is declared an immutable input by the schema)");
    AT_DISPATCH_FLOATING_TYPES(alpha.scalar_type(), "consistent_momentum_3d_cpu", [&] {
        consistent_momentum_3d_impl<scalar_t>(
            alpha, u, v, w, rho_new, flux, uo, vo, wo, alpha_out,
            rho_water, rho_air, dt, h, gx, gy, gz, (int)flux_scheme);
    });
}

}  // namespace

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("consistent_momentum_3d", &consistent_momentum_3d_cpu);
}

}  // namespace lilytorch_kernels
