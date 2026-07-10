// =====================================================================
//  streaming_sdf_direct.cu — B=1 fast-path kernels (2-D and 3-D)
// =====================================================================

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace lilytorch_kernels {

// ---- 2-D samplers (duplicated from streaming_sdf_2d.cu for linking) --------
template <typename scalar_t>
__device__ __forceinline__ scalar_t bilinear_sample_uniform_2d(
    const scalar_t* __restrict__ F, int Mx, int My,
    scalar_t bx0, scalar_t by0, scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx=(xq-bx0)*inv_dx, ty=(yq-by0)*inv_dy;
    scalar_t Mx_lim=(scalar_t)(Mx-1), My_lim=(scalar_t)(My-1);
    tx=max((scalar_t)0,min(tx,Mx_lim)); ty=max((scalar_t)0,min(ty,My_lim));
    int ix=(int)tx; if(ix>Mx-2)ix=Mx-2;
    int iy=(int)ty; if(iy>My-2)iy=My-2;
    scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy;
    scalar_t wx0=(scalar_t)1-fx, wx1=fx, wy0=(scalar_t)1-fy, wy1=fy;
    int s1=My, base=ix*s1+iy;
    return wx0*(wy0*F[base]+wy1*F[base+1])+wx1*(wy0*F[base+s1]+wy1*F[base+s1+1]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t biquadratic_sample_uniform_2d(
    const scalar_t* __restrict__ F, int Mx, int My,
    scalar_t bx0, scalar_t by0, scalar_t inv_dx, scalar_t inv_dy,
    scalar_t xq, scalar_t yq)
{
    scalar_t tx=(xq-bx0)*inv_dx, ty=(yq-by0)*inv_dy;
    scalar_t Mx_lim=(scalar_t)(Mx-1), My_lim=(scalar_t)(My-1);
    tx=max((scalar_t)0,min(tx,Mx_lim)); ty=max((scalar_t)0,min(ty,My_lim));
    int ix=(int)tx; if(ix>Mx-2)ix=Mx-2;
    int iy=(int)ty; if(iy>My-2)iy=My-2;
    if(ix<1||iy<1||Mx<3||My<3) return bilinear_sample_uniform_2d<scalar_t>(F,Mx,My,bx0,by0,inv_dx,inv_dy,xq,yq);
    scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy, half=(scalar_t)0.5;
    scalar_t wxm=half*fx*(fx-(scalar_t)1), wx0=(scalar_t)1-fx*fx, wxp=half*fx*(fx+(scalar_t)1);
    scalar_t wym=half*fy*(fy-(scalar_t)1), wy0=(scalar_t)1-fy*fy, wyp=half*fy*(fy+(scalar_t)1);
    int s1=My, base=(ix-1)*s1+(iy-1);
    scalar_t out=(scalar_t)0;
    for(int dx=0;dx<3;++dx){
        scalar_t wx=(dx==0)?wxm:(dx==1?wx0:wxp);
        int b0=base+dx*s1;
        out+=wx*(wym*F[b0]+wy0*F[b0+1]+wyp*F[b0+2]);
    }
    return out;
}

// ---- 3-D samplers (duplicated from streaming_sdf.cu for linking) -----------
template <typename scalar_t>
__device__ __forceinline__ scalar_t trilinear_sample_uniform(
    const scalar_t* __restrict__ F, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx=(xq-bx0)*inv_dx, ty=(yq-by0)*inv_dy, tz=(zq-bz0)*inv_dz;
    scalar_t Mx_lim=(scalar_t)(Mx-1), My_lim=(scalar_t)(My-1), Mz_lim=(scalar_t)(Mz-1);
    tx=max((scalar_t)0,min(tx,Mx_lim)); ty=max((scalar_t)0,min(ty,My_lim)); tz=max((scalar_t)0,min(tz,Mz_lim));
    int ix=(int)tx; if(ix>Mx-2)ix=Mx-2;
    int iy=(int)ty; if(iy>My-2)iy=My-2;
    int iz=(int)tz; if(iz>Mz-2)iz=Mz-2;
    scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy, fz=tz-(scalar_t)iz;
    scalar_t wx0=(scalar_t)1-fx, wx1=fx, wy0=(scalar_t)1-fy, wy1=fy, wz0=(scalar_t)1-fz, wz1=fz;
    int s2=Mz, s1=My*Mz, base=ix*s1+iy*s2+iz;
    return wx0*(wy0*(wz0*F[base]+wz1*F[base+1])+wy1*(wz0*F[base+s2]+wz1*F[base+s2+1]))
         + wx1*(wy0*(wz0*F[base+s1]+wz1*F[base+s1+1])+wy1*(wz0*F[base+s1+s2]+wz1*F[base+s1+s2+1]));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t triquadratic_sample_uniform(
    const scalar_t* __restrict__ F, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0,
    scalar_t inv_dx, scalar_t inv_dy, scalar_t inv_dz,
    scalar_t xq, scalar_t yq, scalar_t zq)
{
    scalar_t tx=(xq-bx0)*inv_dx, ty=(yq-by0)*inv_dy, tz=(zq-bz0)*inv_dz;
    scalar_t Mx_lim=(scalar_t)(Mx-1), My_lim=(scalar_t)(My-1), Mz_lim=(scalar_t)(Mz-1);
    tx=max((scalar_t)0,min(tx,Mx_lim)); ty=max((scalar_t)0,min(ty,My_lim)); tz=max((scalar_t)0,min(tz,Mz_lim));
    int ix=(int)tx; if(ix>Mx-2)ix=Mx-2;
    int iy=(int)ty; if(iy>My-2)iy=My-2;
    int iz=(int)tz; if(iz>Mz-2)iz=Mz-2;
    if(ix<1||iy<1||iz<1||Mx<3||My<3||Mz<3)
        return trilinear_sample_uniform<scalar_t>(F,Mx,My,Mz,bx0,by0,bz0,inv_dx,inv_dy,inv_dz,xq,yq,zq);
    scalar_t fx=tx-(scalar_t)ix, fy=ty-(scalar_t)iy, fz=tz-(scalar_t)iz, half=(scalar_t)0.5;
    scalar_t wxm=half*fx*(fx-(scalar_t)1), wx0=(scalar_t)1-fx*fx, wxp=half*fx*(fx+(scalar_t)1);
    scalar_t wym=half*fy*(fy-(scalar_t)1), wy0=(scalar_t)1-fy*fy, wyp=half*fy*(fy+(scalar_t)1);
    scalar_t wzm=half*fz*(fz-(scalar_t)1), wz0=(scalar_t)1-fz*fz, wzp=half*fz*(fz+(scalar_t)1);
    int s2=Mz, s1=My*Mz, base=(ix-1)*s1+(iy-1)*s2+(iz-1);
    scalar_t out=(scalar_t)0;
    for(int dx=0;dx<3;++dx){
        scalar_t wx=(dx==0)?wxm:(dx==1?wx0:wxp); int b0=base+dx*s1; scalar_t plane=(scalar_t)0;
        for(int dy=0;dy<3;++dy){
            scalar_t wy=(dy==0)?wym:(dy==1?wy0:wyp); int b1=b0+dy*s2;
            plane+=wy*(wzm*F[b1]+wz0*F[b1+1]+wzp*F[b1+2]);
        }
        out+=wx*plane;
    }
    return out;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sdf_sample_2d(int interp, const scalar_t* F, int Mx, int My,
    scalar_t bx0, scalar_t by0, scalar_t idx_, scalar_t idy_, scalar_t xq, scalar_t yq) {
    return interp==1 ? biquadratic_sample_uniform_2d<scalar_t>(F,Mx,My,bx0,by0,idx_,idy_,xq,yq)
                     : bilinear_sample_uniform_2d<scalar_t>(F,Mx,My,bx0,by0,idx_,idy_,xq,yq);
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t sdf_sample_3d(int interp, const scalar_t* F, int Mx, int My, int Mz,
    scalar_t bx0, scalar_t by0, scalar_t bz0, scalar_t idx_, scalar_t idy_, scalar_t idz_,
    scalar_t xq, scalar_t yq, scalar_t zq) {
    return interp==1 ? triquadratic_sample_uniform<scalar_t>(F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,xq,yq,zq)
                     : trilinear_sample_uniform<scalar_t>(F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,xq,yq,zq);
}

// =====================================================================
//  2-D direct kernel + launcher
// =====================================================================
template <typename scalar_t>
__global__ void streaming_sdf_stag_2d_direct_kernel(
    const scalar_t* __restrict__ F_flat, const int64_t* __restrict__ F_offsets,
    const int64_t* __restrict__ body_shapes, const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx, const scalar_t* __restrict__ gy,
    int Ngx, int Ngy, scalar_t half_h,
    scalar_t* __restrict__ sdf_cc, scalar_t* __restrict__ sdf_u, scalar_t* __restrict__ sdf_v,
    scalar_t* __restrict__ body_u, scalar_t* __restrict__ body_v, int interp)
{
    int b=blockIdx.y, local=blockIdx.x*blockDim.x+threadIdx.x;
    int Ai=(int)aabb_dim[b*2+0], Aj=(int)aabb_dim[b*2+1];
    if(local>=Ai*Aj) return;
    int di=local/Aj, dj=local-di*Aj;
    int i0=(int)aabb_lo[b*2+0], j0=(int)aabb_lo[b*2+1];
    int i=i0+di, j=j0+dj, g_idx=i*Ngy+j;

    const scalar_t* F=F_flat+F_offsets[b];
    int Mx=(int)body_shapes[b*2+0], My=(int)body_shapes[b*2+1];
    const scalar_t* M=body_meta+b*7;
    scalar_t bx0=M[0], by0=M[1], idx_=M[4], idy_=M[5];
    const scalar_t* K=kin+b*11;
    scalar_t r00=K[0], r01=K[1], r10=K[2], r11=K[3];
    scalar_t bp_x=K[4], bp_y=K[5], cm_x=K[6], cm_y=K[7], lv_x=K[8], lv_y=K[9], om=K[10];

    scalar_t xc=gx[i], yc=gy[j];
    scalar_t bxq=r00*(xc-bp_x)+r01*(yc-bp_y), byq=r10*(xc-bp_x)+r11*(yc-bp_y);
    scalar_t neg_hh=-half_h, du_x=neg_hh*r00, du_y=neg_hh*r10, dv_x=neg_hh*r01, dv_y=neg_hh*r11;

    scalar_t s=sdf_sample_2d<scalar_t>(interp,F,Mx,My,bx0,by0,idx_,idy_,bxq,byq);
    if(s<sdf_cc[g_idx]) sdf_cc[g_idx]=s;
    s=sdf_sample_2d<scalar_t>(interp,F,Mx,My,bx0,by0,idx_,idy_,bxq+du_x,byq+du_y);
    if(s<sdf_u[g_idx]){sdf_u[g_idx]=s; body_u[g_idx]=lv_x-om*(yc-cm_y);}
    s=sdf_sample_2d<scalar_t>(interp,F,Mx,My,bx0,by0,idx_,idy_,bxq+dv_x,byq+dv_y);
    if(s<sdf_v[g_idx]){sdf_v[g_idx]=s; body_v[g_idx]=lv_y+om*(xc-cm_x);}
}

void streaming_sdf_stag_2d_direct_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy,
    double h_grid, int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v,
    at::Tensor body_u, at::Tensor body_v, int64_t interp,
    int64_t, int64_t, int64_t, int64_t)
{
    int B=(int)aabb_dim.size(0); if(B<=0||max_vol_per_body<=0) return;
    cudaStream_t s=at::cuda::getCurrentCUDAStream();
    int Ngx=(int)gx.numel(), Ngy=(int)gy.numel();
    int bs=(max_vol_per_body<=128)?32:(max_vol_per_body<=4096)?128:256;
    auto aabb_dim_c = aabb_dim.contiguous();
    auto aabb_dim_cpu = aabb_dim_c.cpu();
    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_2d_direct", [&] {
        const int64_t* dim_ptr     = aabb_dim_cpu.data_ptr<int64_t>();   // CPU — for host read
        const int64_t* dim_ptr_gpu = aabb_dim_c.data_ptr<int64_t>();     // GPU — for kernel
        for(int b=0;b<B;++b){
            int Ai=(int)dim_ptr[b*2+0], Aj=(int)dim_ptr[b*2+1];
            int vol=Ai*Aj; if(vol<=0) continue;
            streaming_sdf_stag_2d_direct_kernel<scalar_t>
                <<<dim3((vol+bs-1)/bs,1,1), dim3(bs,1,1), 0, s>>>(
                    F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(), dim_ptr_gpu,
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
                    Ngx, Ngy, (scalar_t)(0.5*h_grid),
                    sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(), sdf_v.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(), (int)interp);
        }
    });
}

// =====================================================================
//  3-D direct kernel + launcher
// =====================================================================
template <typename scalar_t>
__global__ void streaming_sdf_stag_3d_direct_kernel(
    const scalar_t* __restrict__ F_flat, const int64_t* __restrict__ F_offsets,
    const int64_t* __restrict__ body_shapes, const scalar_t* __restrict__ body_meta,
    const scalar_t* __restrict__ kin,
    const int64_t* __restrict__ aabb_lo, const int64_t* __restrict__ aabb_dim,
    const scalar_t* __restrict__ gx, const scalar_t* __restrict__ gy, const scalar_t* __restrict__ gz,
    int Ngx, int Ngy, int Ngz, scalar_t half_h,
    scalar_t* __restrict__ sdf_cc, scalar_t* __restrict__ sdf_u, scalar_t* __restrict__ sdf_v, scalar_t* __restrict__ sdf_w,
    scalar_t* __restrict__ body_u, scalar_t* __restrict__ body_v, scalar_t* __restrict__ body_w, int interp)
{
    int b=blockIdx.y, local=blockIdx.x*blockDim.x+threadIdx.x;
    int Ai=(int)aabb_dim[b*3+0], Aj=(int)aabb_dim[b*3+1], Ak=(int)aabb_dim[b*3+2];
    if(local>=Ai*Aj*Ak) return;
    int rem=local/Ak, dk=local-rem*Ak, di=rem/Aj, dj=rem-di*Aj;
    int i0=(int)aabb_lo[b*3+0], j0=(int)aabb_lo[b*3+1], k0=(int)aabb_lo[b*3+2];
    int i=i0+di, j=j0+dj, k=k0+dk;
    int64_t g_idx=((int64_t)i*Ngy+j)*Ngz+k;

    const scalar_t* F=F_flat+F_offsets[b];
    int Mx=(int)body_shapes[b*3+0], My=(int)body_shapes[b*3+1], Mz=(int)body_shapes[b*3+2];
    const scalar_t* M=body_meta+b*10;
    scalar_t bx0=M[0], by0=M[1], bz0=M[2], idx_=M[6], idy_=M[7], idz_=M[8];
    const scalar_t* K=kin+b*21;
    scalar_t r00=K[0], r01=K[1], r02=K[2], r10=K[3], r11=K[4], r12=K[5], r20=K[6], r21=K[7], r22=K[8];
    scalar_t bp_x=K[9], bp_y=K[10], bp_z=K[11], cm_x=K[12], cm_y=K[13], cm_z=K[14];
    scalar_t lv_x=K[15], lv_y=K[16], lv_z=K[17], av_x=K[18], av_y=K[19], av_z=K[20];
    scalar_t xc=gx[i], yc=gy[j], zc=gz[k];
    scalar_t dxw=xc-bp_x, dyw=yc-bp_y, bzw=zc-bp_z;
    scalar_t bxq=r00*dxw+r01*dyw+r02*bzw, byq=r10*dxw+r11*dyw+r12*bzw, bzq=r20*dxw+r21*dyw+r22*bzw;
    scalar_t neg_hh=-half_h;
    scalar_t du_x=neg_hh*r00, du_y=neg_hh*r10, du_z=neg_hh*r20;
    scalar_t dv_x=neg_hh*r01, dv_y=neg_hh*r11, dv_z=neg_hh*r21;
    scalar_t dw_x=neg_hh*r02, dw_y=neg_hh*r12, dw_z=neg_hh*r22;

    scalar_t s=sdf_sample_3d<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq,byq,bzq);
    if(s<sdf_cc[g_idx]) sdf_cc[g_idx]=s;
    s=sdf_sample_3d<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+du_x,byq+du_y,bzq+du_z);
    if(s<sdf_u[g_idx]){sdf_u[g_idx]=s; body_u[g_idx]=lv_x+av_y*(zc-cm_z)-av_z*(yc-cm_y);}
    s=sdf_sample_3d<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+dv_x,byq+dv_y,bzq+dv_z);
    if(s<sdf_v[g_idx]){sdf_v[g_idx]=s; body_v[g_idx]=lv_y+av_z*(xc-cm_x)-av_x*(zc-cm_z);}
    s=sdf_sample_3d<scalar_t>(interp,F,Mx,My,Mz,bx0,by0,bz0,idx_,idy_,idz_,bxq+dw_x,byq+dw_y,bzq+dw_z);
    if(s<sdf_w[g_idx]){sdf_w[g_idx]=s; body_w[g_idx]=lv_z+av_x*(yc-cm_y)-av_y*(xc-cm_x);}
}

void streaming_sdf_stag_3d_direct_cuda(
    const at::Tensor& F_flat, const at::Tensor& F_offsets,
    const at::Tensor& body_shapes, const at::Tensor& body_meta, const at::Tensor& kin,
    const at::Tensor& aabb_lo, const at::Tensor& aabb_dim,
    const at::Tensor& gx, const at::Tensor& gy, const at::Tensor& gz,
    double h_grid, int64_t max_vol_per_body,
    at::Tensor sdf_cc, at::Tensor sdf_u, at::Tensor sdf_v, at::Tensor sdf_w,
    at::Tensor body_u, at::Tensor body_v, at::Tensor body_w, int64_t interp,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{
    int B=(int)aabb_dim.size(0); if(B<=0||max_vol_per_body<=0) return;
    cudaStream_t s=at::cuda::getCurrentCUDAStream();
    int Ngx=(int)gx.numel(), Ngy=(int)gy.numel(), Ngz=(int)gz.numel();
    int bs=(max_vol_per_body<=128)?32:(max_vol_per_body<=4096)?128:256;
    auto aabb_dim_c = aabb_dim.contiguous();
    auto aabb_dim_cpu = aabb_dim_c.cpu();
    AT_DISPATCH_FLOATING_TYPES(F_flat.scalar_type(), "sdf_stag_3d_direct", [&] {
        const int64_t* dim_ptr     = aabb_dim_cpu.data_ptr<int64_t>();   // CPU — for host read
        const int64_t* dim_ptr_gpu = aabb_dim_c.data_ptr<int64_t>();     // GPU — for kernel
        for(int b=0;b<B;++b){
            int Ai=(int)dim_ptr[b*3+0], Aj=(int)dim_ptr[b*3+1], Ak=(int)dim_ptr[b*3+2];
            int vol=Ai*Aj*Ak; if(vol<=0) continue;
            streaming_sdf_stag_3d_direct_kernel<scalar_t>
                <<<dim3((vol+bs-1)/bs,1,1), dim3(bs,1,1), 0, s>>>(
                    F_flat.data_ptr<scalar_t>(), F_offsets.data_ptr<int64_t>(),
                    body_shapes.data_ptr<int64_t>(), body_meta.data_ptr<scalar_t>(),
                    kin.data_ptr<scalar_t>(),
                    aabb_lo.data_ptr<int64_t>(), dim_ptr_gpu,
                    gx.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(), gz.data_ptr<scalar_t>(),
                    Ngx, Ngy, Ngz, (scalar_t)(0.5*h_grid),
                    sdf_cc.data_ptr<scalar_t>(), sdf_u.data_ptr<scalar_t>(),
                    sdf_v.data_ptr<scalar_t>(), sdf_w.data_ptr<scalar_t>(),
                    body_u.data_ptr<scalar_t>(), body_v.data_ptr<scalar_t>(), body_w.data_ptr<scalar_t>(),
                    (int)interp);
        }
    });
}

TORCH_LIBRARY_IMPL(lilytorch_kernels, CUDA, m) {
    m.impl("streaming_sdf_stag_2d_direct", &streaming_sdf_stag_2d_direct_cuda);
    m.impl("streaming_sdf_stag_3d_direct", &streaming_sdf_stag_3d_direct_cuda);
}
}  // namespace lilytorch_kernels
