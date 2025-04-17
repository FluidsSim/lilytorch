#include <Python.h>
#include <torch/extension.h>
#include <iostream>

using namespace torch::indexing; // None, ..., Slice

#define PRINT(x) for (int i = 0; i < x.size(0); i++) {for (int j = 0; j < x.size(1); j++) std::cout << x[i][j].item().toDouble() << " "; std::cout << std::endl;} std::cout << std::endl;

namespace poisson_cpp {

class PoissonSolver : public torch::CustomClassHolder {
private:
    double m_jcap_tol;
    double m_tol;
    double m_h2;
    int m_max_cycles;
    int m_nsmoothing;
    bool m_verbose;
    torch::DeviceType m_device;
    int m_BC;

public:
    PoissonSolver(
        const torch::Tensor& h2,
        const double& tol,
        const double& jcap_tol,
        const int64_t& max_cycles,
        const int64_t& nsmoothing,
        const int64_t& device,
        const int64_t& bc,
        const bool& verbose
    ) {
        m_tol = tol;
        m_jcap_tol = jcap_tol;
        m_h2 = h2.item().toDouble();
        m_max_cycles = max_cycles;
        m_nsmoothing = nsmoothing;
        m_device = device == 1 ? torch::kCUDA : torch::kCPU;
        m_BC = bc;
        m_verbose = verbose;
    }

private:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    build_operators(const torch::Tensor& c) {
        torch::Tensor c_ = c.to(m_device),
                    c_h = (c_.slice(0, 1, None) + c_.slice(0, 0, -1)) / 2,
                    c_v = (c_.slice(1, 1, None) + c_.slice(1, 0, -1)) / 2,
                    Jdiag = torch::zeros_like(c, torch::device(m_device));
        Jdiag.slice(0, 1, -1).slice(1, 1, -1)
            += c_h.slice(0, 1, None).slice(1, 1, -1)
            +  c_h.slice(0, 0, -1).slice(1, 1, -1)
            +  c_v.slice(0, 1, -1).slice(1, 1, None)
            +  c_v.slice(0, 1, -1).slice(1, 0, -1);
        torch::Tensor Jdiag_inv = torch::where(Jdiag < m_jcap_tol, 0, 1.0 / Jdiag);
        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
            (Jdiag, Jdiag_inv, c_h, c_v);
    }

    torch::Tensor LU(
        const torch::Tensor& p,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v
    ) {
        torch::Tensor res = torch::zeros_like(p, torch::device(m_device));
        res.slice(0, 1, -1).slice(1, 1, -1)
            += c_h.slice(0, 1, None).slice(1, 1, -1) * p.slice(0, 2, None).slice(1, 1, -1)
            +  c_h.slice(0, 0, -1).slice(1, 1, -1) * p.slice(0, 0, -2).slice(1, 1, -1)
            +  c_v.slice(0, 1, -1).slice(1, 1, None) * p.slice(0, 1, -1).slice(1, 2, None)
            +  c_v.slice(0, 1, -1).slice(1, 0, -1) * p.slice(0, 1, -1).slice(1, 0, -2);
        return res;
    }

    torch::Tensor inline Au(
        const torch::Tensor& u,
        const torch::Tensor& Jdiag,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v,
        const double& h2
    ) {
        return (LU(u, c_h, c_v) - Jdiag * u) / h2;
    }

    void Neumann_BC(torch::Tensor& q) {
        q[0] = q[1];
        q[-1] = q[-2];
        q.slice(1, 0, 1) = q.slice(1, 1, 2);
        q.slice(1, -1, None) = q.slice(1, -2, -1);
    }

    void Dirichlet_BC(torch::Tensor& q) {
        q[0] = 0;
        q[-1] = 0;
        q.slice(1, 0, 1) = 0;
        q.slice(1, -1, None) = 0;
    }

    void BC(torch::Tensor& q) {
        switch (m_BC) {
            case 0: Dirichlet_BC(q); break;
            case 1: Neumann_BC(q); break;
            default: break;
        }
    }

    void inline smooth(
        torch::Tensor& u,
        const torch::Tensor& f,
        const torch::Tensor& Jdiag_inv,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v,
        const double& h2
    ) {
        for (int i = 0; i < m_nsmoothing; i++)
            u = (LU(u, c_h, c_v) - f * h2) * Jdiag_inv;
        BC(u);
    }

    torch::Tensor restrict_simple(const torch::Tensor& r) {
        return r.slice(0, 0, r.size(0), 2).slice(1, 0, r.size(1), 2).clone();
    }

    torch::Tensor prolong(const torch::Tensor& err_coarse, int& n) {
        torch::Tensor err = torch::zeros({n + 1, n + 1}, torch::device(m_device));
        err.slice(0, 0, None, 2).slice(1, 0, None, 2) = err_coarse;
        err.slice(0, 1, None, 2).slice(1, 0, None, 2) =
            0.5 * (err_coarse.slice(0, 1, None) + err_coarse.slice(0, 0, -1));
        err.slice(0, 0, None, 2).slice(1, 1, None, 2) =
            0.5 * (err_coarse.slice(1, 1, None) + err_coarse.slice(1, 0, -1));
        err.slice(0, 1, None, 2).slice(1, 1, None, 2) = 0.25 * (
            err_coarse.slice(0, 0, -1).slice(1, 0, -1)
            + err_coarse.slice(0, 1, None).slice(1, 0, -1)
            + err_coarse.slice(0, 0, -1).slice(1, 1, None)
            + err_coarse.slice(0, 1, None).slice(1, 1, None)
        );
        return err;
    }

public:
    torch::Tensor CG(
        const torch::Tensor& f,
        torch::Tensor& u,
        const torch::Tensor& c,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v,
        double h2,
        int64_t maxit
    ) {
        auto [Jdiag, Jdiag_inv, c_h_, c_v_] = build_operators(c);
        torch::Tensor r = f - Au(u, Jdiag, c_h_, c_v_, h2),
                    d = r;
        double old_norm = torch::tensordot(r, r, {0, 1}, {0, 1}).item().toDouble(),
            new_norm = 0.;
        for (int i = 0; i < maxit; i++) {
            if (old_norm < m_tol) break;
            torch::Tensor Ad = Au(d, Jdiag, c_h_, c_v_, h2);
            double alpha = old_norm / torch::tensordot(d, Ad, {0, 1}, {0, 1}).item().toDouble();
            u += alpha * d;
            r -= alpha * Ad;
            new_norm = torch::tensordot(r, r, {0, 1}, {0, 1}).item().toDouble();
            d = r + new_norm / old_norm * d;
            old_norm = new_norm;
        }
        return u;
    }

    std::tuple<torch::Tensor, torch::Tensor> CG_jacobi_cond(
        const torch::Tensor& f,
        torch::Tensor& u,
        const torch::Tensor& c,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v,
        double h2,
        int64_t maxit
    ) {
        auto [Jdiag, Jdiag_inv, c_h_, c_v_] = build_operators(c);
        torch::Tensor r = f - Au(u, Jdiag, c_h_, c_v_, h2),
                    z = r * Jdiag_inv / h2,
                    d = z;
        double old_norm = torch::tensordot(r, z, {0, 1}, {0, 1}).item().toDouble(),
            new_norm = 0.;
        for (int i = 0; i < maxit; i++) {
            if (old_norm < m_tol) break;
            torch::Tensor Ad = Au(d, Jdiag, c_h_, c_v_, h2);
            double alpha = old_norm / torch::tensordot(d, Ad, {0, 1}, {0, 1}).item().toDouble();
            u += alpha * d;
            r -= alpha * Ad;
            r -= r.mean();
            z = r * Jdiag_inv / h2;
            new_norm = torch::tensordot(r, z, {0, 1}, {0, 1}).item().toDouble();
            d = z + new_norm / old_norm * d;
            old_norm = new_norm;
            BC(u);
        }
        return std::make_tuple(u, r);
    }

    std::tuple<torch::Tensor, torch::Tensor> multigrid(
        const torch::Tensor& f,
        torch::Tensor& u,
        const torch::Tensor& c,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v,
        double h2
    ) {
        int n = f.size(0) - 1;
        if (n == 2) {
            torch::Tensor r;
            std::tie(u, r) = CG_jacobi_cond(f, u, c, c_h, c_v, h2, 100);
            BC(u);
            return std::make_tuple(u, r);
        }
        
        auto [Jdiag_, Jdiag_inv_, c_h_, c_v_] = build_operators(c);
        smooth(u, f, Jdiag_inv_, c_h_, c_v_, h2);
        torch::Tensor r = torch::where(Jdiag_inv_ == 0, 0, f - Au(u, Jdiag_, c_h_, c_v_, h2));
        r -= r.mean();
        if (m_verbose) std::cout << "Multigrid - Steps: " << n << ", Residual: "
                                << r.abs().max().item().toDouble() << std::endl;
        torch::Tensor coarse_residual = restrict_simple(r),
                    c_coarse = c.slice(0, 0, None, 2).slice(1, 0, None, 2),
                    ch_coarse = c_h.slice(0, 0, None, 2).slice(1, 0, None, 2),
                    cv_coarse = c_v.slice(0, 0, None, 2).slice(1, 0, None, 2);
        BC(coarse_residual);
        BC(c_coarse);
        BC(ch_coarse);
        BC(cv_coarse);
        torch::Tensor u_ = torch::zeros_like(coarse_residual);
        auto [err_coarse, _] = multigrid(
            coarse_residual,
            u_,
            c_coarse,
            ch_coarse,
            cv_coarse,
            4*h2
        );
        u += prolong(err_coarse, n);
        smooth(u, f, Jdiag_inv_, c_h_, c_v_, h2);
        r = f - Au(u, Jdiag_, c_h_, c_v_, h2);
        if (m_verbose) std::cout << "Multigrid - Steps: " << n << ", Residual: "
                                << r.slice(0, 1, -1).slice(1, 1, -1).abs().max().item().toDouble() << std::endl;
        return std::make_tuple(u, r);
    }

    torch::Tensor solve_multigrid(
        const torch::Tensor& f,
        torch::Tensor& u,
        const torch::Tensor& c,
        const torch::Tensor& c_h,
        const torch::Tensor& c_v
    ) {
        int cycle = 0;
        double r_err = 1e33;
        torch::Tensor r;
        torch::Tensor f_ = f.to(m_device),
                    u_ = u.to(m_device),
                    c_ = c.to(m_device),
                    c_h_ = c_h.to(m_device),
                    c_v_ = c_v.to(m_device);
        for (; cycle < m_max_cycles; cycle++) {
            if (r_err < m_tol) break;
            std::tie(u, r) = multigrid(f_, u_, c_, c_h_, c_v_, m_h2);
            r_err = r.slice(0, 1, -1).slice(1, 1, -1).abs().max().to(torch::kCPU).item().toDouble();
            if (m_verbose) std::cout << "Cycle number = " << cycle+1 << " - residual = " << r_err << std::endl;
        }
        if (m_verbose) std::cout << "Multigrid residual = " << r_err << ", ncycles = " << cycle << std::endl;
        return u.to(torch::kCPU);
    }
};

} // poisson_cpp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(poisson_cpp, m) {
    m.class_<poisson_cpp::PoissonSolver>("PoissonSolver")
     .def(torch::init<const torch::Tensor, const double, const double, const int64_t, const int64_t, const int64_t, const int64_t, const bool>())
     .def("CG", &poisson_cpp::PoissonSolver::CG)
     .def("CG_jacobi_cond", &poisson_cpp::PoissonSolver::CG_jacobi_cond)
     .def("solve_multigrid", &poisson_cpp::PoissonSolver::solve_multigrid);
}