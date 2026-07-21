// =====================================================================
//  ops_2d.cpp — 2-D CPU implementations (at::parallel_for).
//
//  Contains forces_post_2d, apply_bcs_2d, interp_2d CPU twins.
//  See ops_3d.cpp for the 3-D counterparts.
// =====================================================================
//
//  Hot loops are parallelised with ``at::parallel_for`` (PyTorch's
//  intra-op thread pool); see the rationale comment at the top of
//  ``streaming_sdf_cpu.cpp``.
// =====================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include "common/bc_ops.h"
#include "common/interp.h"

namespace lilytorch_kernels {

// =====================================================================
//  Per-cell update body for 2-D streaming-min update.  Each cell is
//  touched once per launch -> no atomics needed.
//
//  In 2-D the rigid-body angular velocity is a scalar ``omega``
//  (out-of-plane), and the face velocities at (xc-h/2, yc) and
//  (xc, yc-h/2) are:
//      u_face = lv_x - omega * (yc - cm_y)
//      v_face = lv_y + omega * (xc - cm_x)
//  These are the 3-D formulas with (av_x, av_y, av_z) = (0, 0, omega)
//  and zc - cm_z = 0 collapsed.
// =====================================================================
template <typename scalar_t>
static inline void update_cell_2d(
    const scalar_t* F,
    const int Mx, const int My,
    const scalar_t bx0, const scalar_t by0,
    const scalar_t inv_dx, const scalar_t inv_dy,
    const scalar_t r00, const scalar_t r01,
    const scalar_t r10, const scalar_t r11,
    const scalar_t bp_x, const scalar_t bp_y,
    const scalar_t cm_x, const scalar_t cm_y,
    const scalar_t lv_x, const scalar_t lv_y,
    const scalar_t omega,
    // Body-frame face deltas: Δ_u = -half_h * col0(R_T),
    //                          Δ_v = -half_h * col1(R_T).
    const scalar_t du_x, const scalar_t du_y,
    const scalar_t dv_x, const scalar_t dv_y,
    const scalar_t xc, const scalar_t yc,
    const std::int64_t g_idx, const std::int64_t sparse_idx,
    scalar_t* sdf_cc, scalar_t* sdf_u, scalar_t* sdf_v,
    scalar_t* bU, scalar_t* bV,
    scalar_t* sparse_cc,
    // 0 = bilinear (default), 1 = biquadratic (Lagrange 3x3).
    const int interp_method)
{
    // Body-frame CC point (single 2x2 rotation; the 2 face points are
    // derived from it by precomputed deltas).
    const scalar_t dxw = xc - bp_x, dyw = yc - bp_y;
    const scalar_t bxq = r00 * dxw + r01 * dyw;
    const scalar_t byq = r10 * dxw + r11 * dyw;

    auto sample = [&](scalar_t xqs, scalar_t yqs) -> scalar_t {
        if (interp_method == 1) {
            return biquadratic_sample_uniform_2d<scalar_t>(
                F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
        }
        return bilinear_sample_uniform_2d<scalar_t>(
            F, Mx, My, bx0, by0, inv_dx, inv_dy, xqs, yqs);
    };

    // ---- cc ----
    {
        const scalar_t s = sample(bxq, byq);
        sparse_cc[sparse_idx] = s;
        if (s < sdf_cc[g_idx]) sdf_cc[g_idx] = s;
    }
    // ---- u-face: world (xc - h/2, yc) ----
    {
        const scalar_t s = sample(bxq + du_x, byq + du_y);
        if (s < sdf_u[g_idx]) {
            sdf_u[g_idx] = s;
            bU[g_idx] = lv_x - omega * (yc - cm_y);
        }
    }
    // ---- v-face: world (xc, yc - h/2) ----
    {
        const scalar_t s = sample(bxq + dv_x, byq + dv_y);
        if (s < sdf_v[g_idx]) {
            sdf_v[g_idx] = s;
            bV[g_idx] = lv_y + omega * (xc - cm_x);
        }
    }
}

// =====================================================================
//  apply_bcs_2d  (Phase H 2-D analogue of apply_bcs_3d)
//
//  Each op writes a single 1-D ghost line of u or v.  Ops run in three
//  ordered stages (Neumann → Dirichlet → reflective); within a stage the
//  write-ownership rule in bc_ops.h gives every cell — corner ghosts
//  included — one writer and a race-free source, so CPU and CUDA agree
//  cell for cell.  See bc_ops.h.
//
//  shapes  : int64 [2,2]      -> per-component (Nx, Ny)
//  neu_desc: int32 [N_neu, 3] -> (comp, axis, side)
//                                   comp in {0=u, 1=v}
//                                   axis in {0=x, 1=y}
//                                   side in {0=lo, 1=hi}
//  dir_desc: int32 [N_dir, 3] -> (comp, axis, offset)
//                                offset is signed: dst = offset if >=0
//                                else (sz + offset).
//  dir_val : float[N_dir]
// =====================================================================

template <typename scalar_t>
static void apply_bcs_2d_one_line(
    scalar_t* base,
    const std::int64_t* shapes_p,
    const int kind, const int* desc, const int nops, const int op,
    const scalar_t* vals,
    const int comp, const int axis,
    const int dst_along, const int src_along,
    const int Ny, const int dim0_max)
{
    using namespace lilytorch_kernels::bcs;

    // Flat layout: base[i, j] = base[i*Ny + j].
    at::parallel_for(0, dim0_max, /*grain_size=*/1024, [&](int64_t _begin, int64_t _end) {
    for (int64_t t = _begin; t < _end; ++t) {
        const int i = (int)t;

        int c[2];
        if (axis == 0) { c[0] = dst_along; c[1] = i;         }
        else           { c[0] = i;         c[1] = dst_along; }

        int s[2];
        if (!bc_own_and_source(kind, desc, nops, op, shapes_p, /*ndim=*/2,
                               comp, axis, src_along, c, s))
            continue;                              // a lower-indexed op owns it

        const std::int64_t dst_lin = (std::int64_t)c[0] * Ny + c[1];
        const std::int64_t src_lin = (std::int64_t)s[0] * Ny + s[1];

        if      (kind == BC_KIND_NEUMANN)   base[dst_lin] = base[src_lin];
        else if (kind == BC_KIND_DIRICHLET) base[dst_lin] = vals[op];
        else base[dst_lin] = scalar_t(2) * vals[op] - base[src_lin];
    }
    });
}

void apply_bcs_2d_cpu(
    at::Tensor u, at::Tensor v,
    const at::Tensor& shapes,
    const at::Tensor& neu_desc,
    const at::Tensor& dir_desc,
    const at::Tensor& dir_val,
    const at::Tensor& ref_desc,
    const at::Tensor& ref_val,
    const int64_t /*max_line_dim*/)
{
    TORCH_CHECK(u.is_contiguous() && v.is_contiguous(),
                "apply_bcs_2d_cpu: u/v must be contiguous");
    TORCH_CHECK(u.scalar_type() == v.scalar_type(),
                "apply_bcs_2d_cpu: u/v must share dtype");
    TORCH_CHECK(shapes.scalar_type() == at::kLong &&
                shapes.dim() == 2 && shapes.size(0) == 2 && shapes.size(1) == 2,
                "apply_bcs_2d_cpu: shapes must be int64[2,2]");
    TORCH_CHECK(neu_desc.scalar_type() == at::kInt && neu_desc.dim() == 2 &&
                neu_desc.size(1) == 3,
                "apply_bcs_2d_cpu: neu_desc must be int32[N,3]");
    TORCH_CHECK(dir_desc.scalar_type() == at::kInt && dir_desc.dim() == 2 &&
                dir_desc.size(1) == 3,
                "apply_bcs_2d_cpu: dir_desc must be int32[N,3]");
    TORCH_CHECK(ref_desc.scalar_type() == at::kInt && ref_desc.dim() == 2 &&
                ref_desc.size(1) == 4,
                "apply_bcs_2d_cpu: ref_desc must be int32[N,4]");

    const int N_neu = (int)neu_desc.size(0);
    const int N_dir = (int)dir_desc.size(0);
    const int N_ref = (int)ref_desc.size(0);
    if (N_neu + N_dir + N_ref == 0) return;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "apply_bcs_2d_cpu", [&] {
        using namespace lilytorch_kernels::bcs;

        const int64_t*  shapes_p  = shapes.data_ptr<int64_t>();
        const int*      neu_p     = (N_neu > 0) ? neu_desc.data_ptr<int>() : nullptr;
        const int*      dir_p     = (N_dir > 0) ? dir_desc.data_ptr<int>() : nullptr;
        const scalar_t* dir_val_p = (N_dir > 0) ? dir_val.data_ptr<scalar_t>() : nullptr;
        const int*      ref_p     = (N_ref > 0) ? ref_desc.data_ptr<int>() : nullptr;
        const scalar_t* ref_val_p = (N_ref > 0) ? ref_val.data_ptr<scalar_t>() : nullptr;

        scalar_t* u_p = u.data_ptr<scalar_t>();
        scalar_t* v_p = v.data_ptr<scalar_t>();

        // One stage per op kind, in the same order as the CUDA launches.
        const int   kinds[3] = {BC_KIND_NEUMANN, BC_KIND_DIRICHLET, BC_KIND_REFLECTIVE};
        const int*  descs[3] = {neu_p, dir_p, ref_p};
        const scalar_t* valss[3] = {nullptr, dir_val_p, ref_val_p};
        const int   counts[3] = {N_neu, N_dir, N_ref};

        for (int st = 0; st < 3; ++st) {
            const int kind = kinds[st];
            const int* desc = descs[st];
            const int nops = counts[st];

            for (int op = 0; op < nops; ++op) {
                int comp, axis, dst_along, src_along;
                bc_decode(kind, desc, op, shapes_p, /*ndim=*/2,
                          comp, axis, dst_along, src_along);

                const int Nx = (int)shapes_p[comp*2 + 0];
                const int Ny = (int)shapes_p[comp*2 + 1];

                // axis==0 -> sweep along j (size Ny). axis==1 -> sweep along i (size Nx).
                const int dim0_max = (axis == 0) ? Ny : Nx;

                scalar_t* base = (comp == 0) ? u_p : v_p;

                apply_bcs_2d_one_line<scalar_t>(
                    base, shapes_p, kind, desc, nops, op, valss[st],
                    comp, axis, dst_along, src_along,
                    Ny, dim0_max);
            }
        }
    });
}

// =====================================================================
//  interp_2d_cpu: scattered-point bilinear / biquadratic sampling
// =====================================================================
static void interp_2d_cpu(
    const at::Tensor& F,
    const at::Tensor& xq, const at::Tensor& yq,
    const double bx0, const double by0,
    const double inv_dx, const double inv_dy,
    const int64_t Mx, const int64_t My,
    const int64_t interp_method,
    at::Tensor& G)
{
    const int N = (int)xq.numel();
    if (N == 0) return;

    // Bind temporaries to named locals: ``.contiguous().to(...)`` returns a
    // fresh tensor whose storage is freed at the end of the full-expression
    // unless held.  Without this the raw ``data_ptr`` would dangle for
    // non-contiguous or differently-dtyped inputs and the parallel_for loop
    // below would read freed memory.
    auto F_c  = F.contiguous();
    auto xq_c = xq.contiguous().to(F.scalar_type());
    auto yq_c = yq.contiguous().to(F.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "interp_2d_cpu", [&] {
        const scalar_t* Fp  = F_c.data_ptr<scalar_t>();
        const scalar_t* xqp = xq_c.data_ptr<scalar_t>();
        const scalar_t* yqp = yq_c.data_ptr<scalar_t>();
        scalar_t* Gp = G.data_ptr<scalar_t>();
        const int iMx = (int)Mx, iMy = (int)My;
        const scalar_t bx0s = (scalar_t)bx0, by0s = (scalar_t)by0;
        const scalar_t idx  = (scalar_t)inv_dx, idy = (scalar_t)inv_dy;
        const int method    = (int)interp_method;

        at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (method == 1) {
                    Gp[i] = biquadratic_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                } else {
                    Gp[i] = bilinear_sample_uniform_2d<scalar_t>(
                        Fp, iMx, iMy, bx0s, by0s, idx, idy,
                        xqp[i], yqp[i]);
                }
            }
        });
    });
}

// =====================================================================
void streaming_sdf_forces_post_2d_cpu(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    const double h_grid, const int64_t /*max_vol_per_body*/,
    const at::Tensor& sdf_cc,
    const int64_t interp_method,
    const at::Tensor& u_prev, const at::Tensor& v_prev, const at::Tensor& p_prev,
    const at::Tensor& nu_rho_field,
    const double eps_body,
    const double sample_offset_pressure,
    const double sample_offset_friction,
    const double h2,
    const int64_t delta_order,
    const int64_t force_submethod,
    const double ph_tau,
    at::Tensor out)
{
    const int B = (int)aabb_dim.size(0);
    if (B <= 0) return;

    TORCH_CHECK(out.scalar_type() == at::kDouble,
                "streaming_sdf_forces_post_2d_cpu: out must be float64");
    TORCH_CHECK(out.size(1) == 6,
                "streaming_sdf_forces_post_2d_cpu: out must have 6 channels");

    const int Ngy = (int)gy.numel();
    const int Ngx = (int)gx.numel();

    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "streaming_sdf_forces_post_2d_cpu", [&] {
        auto F_c   = F_flat.contiguous();
        auto gx_c  = gx.contiguous();
        auto gy_c  = gy.contiguous();
        auto sdf_c = sdf_cc.contiguous();
        auto up_c  = u_prev.contiguous();
        auto vp_c  = v_prev.contiguous();
        auto pp_c  = p_prev.contiguous();
        auto nr_c  = nu_rho_field.contiguous();

        const scalar_t* F_ptr   = F_c.data_ptr<scalar_t>();
        const int64_t*  F_off   = F_offsets.data_ptr<int64_t>();
        const int64_t*  shapes  = body_shapes.data_ptr<int64_t>();
        const scalar_t* meta    = body_meta.data_ptr<scalar_t>();
        const scalar_t* kin_ptr = kin.data_ptr<scalar_t>();
        const int64_t*  lo      = aabb_lo.data_ptr<int64_t>();
        const int64_t*  dim_    = aabb_dim.data_ptr<int64_t>();
        const scalar_t* gx_ptr  = gx_c.data_ptr<scalar_t>();
        const scalar_t* gy_ptr  = gy_c.data_ptr<scalar_t>();
        const scalar_t* sdf_cc_p= sdf_c.data_ptr<scalar_t>();
        const scalar_t* u_p     = up_c.data_ptr<scalar_t>();
        const scalar_t* v_p     = vp_c.data_ptr<scalar_t>();
        const scalar_t* p_p     = pp_c.data_ptr<scalar_t>();
        const scalar_t* nr_p    = nr_c.data_ptr<scalar_t>();
        const int64_t   nr_size = nu_rho_field.numel();

        const scalar_t inv_h_s    = (scalar_t)(1.0 / h_grid);
        const scalar_t off_pres   = (scalar_t)sample_offset_pressure;
        const scalar_t off_visc   = (scalar_t)sample_offset_friction;
        const scalar_t pi_v       = (scalar_t)3.141592653589793;
        const double   h2_d       = (double)h2;

        std::vector<double> accs(static_cast<size_t>(B) * 6, 0.0);

        // ---- submethod 0/2: UNIFIED union-measure + partition (2-D) ----
        //      Mirrors forces_post_union_blend_2d_kernel.  Writes into accs
        //      (the h2 factor is applied by the final accumulation below).
        const bool body_normal = (force_submethod == 2);
        const scalar_t inv_eps = (scalar_t)(1.0/eps_body);
        const scalar_t blend_eps = (scalar_t)ph_tau;
        const bool soft = blend_eps > (scalar_t)0;
        const scalar_t inv_blend = soft ? ((scalar_t)1/blend_eps) : (scalar_t)0;
        auto S2=[&](int i,int j)->scalar_t{ return sdf_cc_p[(std::int64_t)i*Ngy+j]; };
        auto Hoff=[&](int i,int j,scalar_t off)->scalar_t{ scalar_t x=(S2(i,j)-off)*inv_eps; x=x<(scalar_t)-1?(scalar_t)-1:(x>(scalar_t)1?(scalar_t)1:x); return (scalar_t)0.5*((scalar_t)1+x+std::sin(pi_v*x)/pi_v); };
        auto hgrad=[&](int i,int j,scalar_t off,scalar_t& gX,scalar_t& gY){
            gX=0;gY=0;
            if(Ngx>=3){ if(i==0) gX=((scalar_t)(-3)*Hoff(0,j,off)+(scalar_t)4*Hoff(1,j,off)-Hoff(2,j,off))*(scalar_t)0.5*inv_h_s; else if(i==Ngx-1) gX=((scalar_t)3*Hoff(Ngx-1,j,off)-(scalar_t)4*Hoff(Ngx-2,j,off)+Hoff(Ngx-3,j,off))*(scalar_t)0.5*inv_h_s; else gX=(Hoff(i+1,j,off)-Hoff(i-1,j,off))*(scalar_t)0.5*inv_h_s; } else if(Ngx==2) gX=(Hoff(1,j,off)-Hoff(0,j,off))*inv_h_s;
            if(Ngy>=3){ if(j==0) gY=((scalar_t)(-3)*Hoff(i,0,off)+(scalar_t)4*Hoff(i,1,off)-Hoff(i,2,off))*(scalar_t)0.5*inv_h_s; else if(j==Ngy-1) gY=((scalar_t)3*Hoff(i,Ngy-1,off)-(scalar_t)4*Hoff(i,Ngy-2,off)+Hoff(i,Ngy-3,off))*(scalar_t)0.5*inv_h_s; else gY=(Hoff(i,j+1,off)-Hoff(i,j-1,off))*(scalar_t)0.5*inv_h_s; } else if(Ngy==2) gY=(Hoff(i,1,off)-Hoff(i,0,off))*inv_h_s;
        };
        auto sgrad=[&](int i,int j,scalar_t& gX,scalar_t& gY){
            gX=0;gY=0;
            if(Ngx>=3){ if(i==0) gX=((scalar_t)(-3)*S2(0,j)+(scalar_t)4*S2(1,j)-S2(2,j))*(scalar_t)0.5*inv_h_s; else if(i==Ngx-1) gX=((scalar_t)3*S2(Ngx-1,j)-(scalar_t)4*S2(Ngx-2,j)+S2(Ngx-3,j))*(scalar_t)0.5*inv_h_s; else gX=(S2(i+1,j)-S2(i-1,j))*(scalar_t)0.5*inv_h_s; } else if(Ngx==2) gX=(S2(1,j)-S2(0,j))*inv_h_s;
            if(Ngy>=3){ if(j==0) gY=((scalar_t)(-3)*S2(i,0)+(scalar_t)4*S2(i,1)-S2(i,2))*(scalar_t)0.5*inv_h_s; else if(j==Ngy-1) gY=((scalar_t)3*S2(i,Ngy-1)-(scalar_t)4*S2(i,Ngy-2)+S2(i,Ngy-3))*(scalar_t)0.5*inv_h_s; else gY=(S2(i,j+1)-S2(i,j-1))*(scalar_t)0.5*inv_h_s; } else if(Ngy==2) gY=(S2(i,1)-S2(i,0))*inv_h_s;
        };
        auto samp2d=[&](const scalar_t* F,int Mx,int My,scalar_t bx0,scalar_t by0,scalar_t idx_,scalar_t idy_,scalar_t xq,scalar_t yq)->scalar_t{
            return (interp_method==1)?biquadratic_sample_uniform_2d<scalar_t>(F,Mx,My,bx0,by0,idx_,idy_,xq,yq):bilinear_sample_uniform_2d<scalar_t>(F,Mx,My,bx0,by0,idx_,idy_,xq,yq); };
        auto trigrad=[&](const scalar_t* F,int Mx,int My,scalar_t bx0,scalar_t by0,scalar_t idx_,scalar_t idy_,scalar_t xq,scalar_t yq,scalar_t& gbx,scalar_t& gby)->scalar_t{
            scalar_t tx=(xq-bx0)*idx_, ty=(yq-by0)*idy_;
            tx=std::max((scalar_t)0,std::min(tx,(scalar_t)(Mx-1))); ty=std::max((scalar_t)0,std::min(ty,(scalar_t)(My-1)));
            int ix=(int)tx; if(ix>Mx-2) ix=Mx-2; int iy=(int)ty; if(iy>My-2) iy=My-2;
            const scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy; const int s1=My, base=ix*s1+iy;
            const scalar_t c00=F[base],c01=F[base+1],c10=F[base+s1],c11=F[base+s1+1];
            const scalar_t omfx=(scalar_t)1-fx, omfy=(scalar_t)1-fy;
            const scalar_t val=omfx*(omfy*c00+fy*c01)+fx*(omfy*c10+fy*c11);
            const scalar_t dfx=omfy*(c10-c00)+fy*(c11-c01), dfy=omfx*(c01-c00)+fx*(c11-c10);
            gbx=dfx*idx_; gby=dfy*idy_; return val; };
        auto covers=[&](int c,int i,int j)->bool{ const int ci0=(int)lo[c*2],cj0=(int)lo[c*2+1],Ci=(int)dim_[c*2],Cj=(int)dim_[c*2+1]; return !(i<ci0||i>=ci0+Ci||j<cj0||j>=cj0+Cj); };
        auto phi_c=[&](int c,int i,int j)->scalar_t{ const scalar_t* Fc=F_ptr+F_off[c]; const int Mxc=(int)shapes[c*2],Myc=(int)shapes[c*2+1]; const scalar_t* Mc=meta+c*7; const scalar_t* Kc=kin_ptr+c*11; const scalar_t dxc=gx_ptr[i]-Kc[4],dyc=gy_ptr[j]-Kc[5]; return samp2d(Fc,Mxc,Myc,Mc[0],Mc[1],Mc[4],Mc[5],Kc[0]*dxc+Kc[1]*dyc,Kc[2]*dxc+Kc[3]*dyc); };
        for(int b=0;b<B;++b){
            const int Ai=(int)dim_[b*2],Aj=(int)dim_[b*2+1]; const int vol=Ai*Aj; if(vol<=0) continue;
            const int i0=(int)lo[b*2],j0=(int)lo[b*2+1];
            const scalar_t* Fb=F_ptr+F_off[b]; const int Mxb=(int)shapes[b*2],Myb=(int)shapes[b*2+1];
            const scalar_t* Mb=meta+b*7; const scalar_t* Kb=kin_ptr+b*11;
            double* lb=accs.data()+(size_t)b*6;
            for(int local=0;local<vol;++local){
                const int di=local/Aj, dj=local-di*Aj; const int i=i0+di,j=j0+dj; const std::int64_t g=(std::int64_t)i*Ngy+j;
                const scalar_t xc=gx_ptr[i], yc=gy_ptr[j];
                scalar_t gpx=0,gpy=0,gvx=0,gvy=0,meas_p=0,meas_v=0; bool has_p,has_v;
                if(!body_normal){ hgrad(i,j,off_pres,gpx,gpy); hgrad(i,j,off_visc,gvx,gvy); has_p=(gpx!=(scalar_t)0||gpy!=(scalar_t)0); has_v=(gvx!=(scalar_t)0||gvy!=(scalar_t)0); }
                else { scalar_t gux,guy; sgrad(i,j,gux,guy); const scalar_t gmag=std::sqrt(gux*gux+guy*guy); const scalar_t phi_u=S2(i,j); const scalar_t ddp=(phi_u-off_pres)*inv_eps,ddv=(phi_u-off_visc)*inv_eps,hie=(scalar_t)0.5*inv_eps; const scalar_t delp=(ddp>(scalar_t)-1&&ddp<(scalar_t)1)?((scalar_t)1+std::cos(pi_v*ddp))*hie:(scalar_t)0; const scalar_t delv=(ddv>(scalar_t)-1&&ddv<(scalar_t)1)?((scalar_t)1+std::cos(pi_v*ddv))*hie:(scalar_t)0; meas_p=delp*gmag; meas_v=delv*gmag; has_p=(meas_p!=(scalar_t)0); has_v=(meas_v!=(scalar_t)0); }
                if(!(has_p||has_v)) continue;
                const scalar_t dxw=xc-Kb[4],dyw=yc-Kb[5]; const scalar_t bxq=Kb[0]*dxw+Kb[1]*dyw, byq=Kb[2]*dxw+Kb[3]*dyw;
                scalar_t s_bself;
                if(!body_normal){ s_bself=samp2d(Fb,Mxb,Myb,Mb[0],Mb[1],Mb[4],Mb[5],bxq,byq); }
                else { scalar_t gbx,gby; s_bself=trigrad(Fb,Mxb,Myb,Mb[0],Mb[1],Mb[4],Mb[5],bxq,byq,gbx,gby); scalar_t nbx=gbx*Kb[0]+gby*Kb[2], nby=gbx*Kb[1]+gby*Kb[3]; const scalar_t nlen=std::sqrt(nbx*nbx+nby*nby); const scalar_t invn=nlen>(scalar_t)0?(scalar_t)1/nlen:(scalar_t)0; nbx*=invn;nby*=invn; gpx=meas_p*nbx;gpy=meas_p*nby;gvx=meas_v*nbx;gvy=meas_v*nby; }
                scalar_t wb=0;
                if(soft){ scalar_t Z=0; for(int c=0;c<B;++c){ if(!covers(c,i,j)) continue; const scalar_t s_c=(c==b)?s_bself:phi_c(c,i,j); Z+=(scalar_t)1/((scalar_t)1+std::exp(s_c*inv_blend)); } if(Z>(scalar_t)0) wb=((scalar_t)1/((scalar_t)1+std::exp(s_bself*inv_blend)))/Z; }
                else { scalar_t smin=(scalar_t)1e30; int bmin=-1; for(int c=0;c<B;++c){ if(!covers(c,i,j)) continue; const scalar_t s_c=(c==b)?s_bself:phi_c(c,i,j); if(s_c<smin){smin=s_c;bmin=c;} } wb=(bmin==b)?(scalar_t)1:(scalar_t)0; }
                if(wb==(scalar_t)0) continue;
                scalar_t fp_x=0,fp_y=0;
                if(has_p){ const scalar_t p_c=p_p[g]; fp_x=-p_c*gpx; fp_y=-p_c*gpy; }
                scalar_t fv_x=0,fv_y=0;
                if(has_v){
                    const scalar_t nu_rho_val=(nr_size==1)?nr_p[0]:nr_p[g];
                    const int im1=(i>0)?i-1:0,ip1=(i+1<Ngx)?i+1:i,im2=(i>1)?i-2:0,ip2=(i+2<Ngx)?i+2:(Ngx-1);
                    const int jm1=(j>0)?j-1:0,jp1=(j+1<Ngy)?j+1:j,jm2=(j>1)?j-2:0,jp2=(j+2<Ngy)?j+2:(Ngy-1);
                    scalar_t dudx; if(i+1<Ngx) dudx=(u_p[ip1*Ngy+j]-u_p[i*Ngy+j])*inv_h_s; else dudx=(u_p[i*Ngy+j]-u_p[im1*Ngy+j])*inv_h_s;
                    scalar_t dvdy; if(j+1<Ngy) dvdy=(v_p[i*Ngy+jp1]-v_p[i*Ngy+j])*inv_h_s; else dvdy=(v_p[i*Ngy+j]-v_p[i*Ngy+jm1])*inv_h_s;
                    const scalar_t u_cc_jm2=(scalar_t)0.5*(u_p[i*Ngy+jm2]+u_p[ip1*Ngy+jm2]); const scalar_t u_cc_jm1=(scalar_t)0.5*(u_p[i*Ngy+jm1]+u_p[ip1*Ngy+jm1]); const scalar_t u_cc_j0=(scalar_t)0.5*(u_p[i*Ngy+j]+u_p[ip1*Ngy+j]); const scalar_t u_cc_jp1=(scalar_t)0.5*(u_p[i*Ngy+jp1]+u_p[ip1*Ngy+jp1]); const scalar_t u_cc_jp2=(scalar_t)0.5*(u_p[i*Ngy+jp2]+u_p[ip1*Ngy+jp2]);
                    scalar_t dudy; if(Ngy>=3){ if(j==0) dudy=((scalar_t)(-3)*u_cc_j0+(scalar_t)4*u_cc_jp1-u_cc_jp2)*(scalar_t)0.5*inv_h_s; else if(j==Ngy-1) dudy=((scalar_t)3*u_cc_j0-(scalar_t)4*u_cc_jm1+u_cc_jm2)*(scalar_t)0.5*inv_h_s; else dudy=(u_cc_jp1-u_cc_jm1)*(scalar_t)0.5*inv_h_s; } else dudy=(u_cc_jp1-u_cc_jm1)*(scalar_t)0.5*inv_h_s;
                    const scalar_t v_cc_im2=(scalar_t)0.5*(v_p[im2*Ngy+j]+v_p[im2*Ngy+jp1]); const scalar_t v_cc_im1=(scalar_t)0.5*(v_p[im1*Ngy+j]+v_p[im1*Ngy+jp1]); const scalar_t v_cc_i0=(scalar_t)0.5*(v_p[i*Ngy+j]+v_p[i*Ngy+jp1]); const scalar_t v_cc_ip1=(scalar_t)0.5*(v_p[ip1*Ngy+j]+v_p[ip1*Ngy+jp1]); const scalar_t v_cc_ip2=(scalar_t)0.5*(v_p[ip2*Ngy+j]+v_p[ip2*Ngy+jp1]);
                    scalar_t dvdx; if(Ngx>=3){ if(i==0) dvdx=((scalar_t)(-3)*v_cc_i0+(scalar_t)4*v_cc_ip1-v_cc_ip2)*(scalar_t)0.5*inv_h_s; else if(i==Ngx-1) dvdx=((scalar_t)3*v_cc_i0-(scalar_t)4*v_cc_im1+v_cc_im2)*(scalar_t)0.5*inv_h_s; else dvdx=(v_cc_ip1-v_cc_im1)*(scalar_t)0.5*inv_h_s; } else dvdx=(v_cc_ip1-v_cc_im1)*(scalar_t)0.5*inv_h_s;
                    const scalar_t sxx=nu_rho_val*(scalar_t)2*dudx, syy=nu_rho_val*(scalar_t)2*dvdy, sxy=nu_rho_val*(dudy+dvdx);
                    fv_x=sxx*gvx+sxy*gvy; fv_y=sxy*gvx+syy*gvy;
                }
                const scalar_t arm_x=xc-Kb[6], arm_y=yc-Kb[7];
                const double fvx=(double)(wb*fv_x),fvy=(double)(wb*fv_y),fpx=(double)(wb*fp_x),fpy=(double)(wb*fp_y);
                lb[0]+=fvx; lb[1]+=fvy; lb[2]+=(double)arm_x*fvy-(double)arm_y*fvx;
                lb[3]+=fpx; lb[4]+=fpy; lb[5]+=(double)arm_x*fpy-(double)arm_y*fpx;
            }
        }

        double* out_ptr = out.data_ptr<double>();
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < 6; ++c)
                out_ptr[b*6 + c] += accs[(size_t)b * 6 + c] * h2_d;
    });
}

// =====================================================================
//  CPU registration. Schemas live in ops.cpp.
// =====================================================================

TORCH_LIBRARY_IMPL(lilytorch_kernels, CPU, m) {
    m.impl("streaming_sdf_forces_post_2d",          &streaming_sdf_forces_post_2d_cpu);
    m.impl("apply_bcs_2d",                          &apply_bcs_2d_cpu);
    m.impl("interp_2d",                        &interp_2d_cpu);
}

}  // namespace lilytorch_kernels
