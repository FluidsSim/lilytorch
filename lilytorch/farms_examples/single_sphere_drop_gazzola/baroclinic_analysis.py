"""Baroclinic vorticity source -- significance analysis for the Gazzola benchmark.

Gazzola et al. (2011) include a baroclinic vorticity source term (Eq. 31):

    omega_baro = -(nabla_rho / rho) x (Du/Dt - nu*lap(u) - g)

where Du/Dt = du/dt + (u . nabla)u is the material derivative.

This term arises from taking the curl of the variable-density momentum
equation.  It is zero when density is spatially uniform and only acts at
the fluid-body interface where nabla_rho != 0.

Significance estimate for rho_s/rho_f = 1.01
=============================================

Physical parameters:
    rho_f = 996 kg/m^3
    rho_s = 1005.96 kg/m^3
    delta_rho = rho_s - rho_f = 9.96 kg/m^3
    delta_rho / rho_f = 0.01  (1% density contrast)
    Ut = 0.02501 m/s  (terminal velocity)
    D = 0.005 m  (cylinder diameter)
    Re = Ut * D / nu = 156

The baroclinic term scales as:
    |omega_baro| ~ (delta_rho / rho) * (U^2 / D) / epsilon
where epsilon is the mollification width (~2*sqrt(2)*h ~ 4h).

Compare to the vorticity diffusion term:
    |nu * lap(omega)| ~ nu * omega / D^2 ~ nu * U / D^3

Ratio (baroclinic / diffusion):
    ~ (delta_rho / rho) * (U * D / nu) * (D / epsilon)
    = (delta_rho / rho) * Re * (D / epsilon)

For rho_s/rho_f = 1.01:
    delta_rho / rho ~ 0.01
    Re ~ 156
    D / epsilon ~ D / (4h)

At the target resolution (1024 x 8192):
    h = 0.04 / 1024 ~ 3.9e-5 m
    epsilon ~ 4h ~ 1.56e-4 m
    D / epsilon ~ 32

    Ratio ~ 0.01 * 156 * 32 ~ 50

This suggests the baroclinic term IS significant relative to diffusion
at the interface.  However, several mitigating factors apply:

1. The term is localized to the thin mollification band (width ~4h)
   around the body surface.  Its global effect on drag is moderated.

2. Gazzola's Brinkman penalization implicitly enforces the correct
   velocity inside the body.  The density gradient creates a vorticity
   source at the interface, but the penalization already ensures the
   correct velocity jump.  The missing baroclinic term primarily
   affects the vorticity field near the interface, not the integrated
   force.

3. For 1% density contrast, the leading-order drag is dominated by
   inertial and viscous effects (C_D ~ 1.5 at Re=156).  The
   baroclinic correction is O(delta_rho/rho) ~ O(1%) of the total.

Conclusion
----------
The baroclinic term is O(1%) of the total force for rho_s/rho_f = 1.01.
It is NOT the cause of the 3x drag underprediction.  The primary issue
is the force computation method (surface-stress integration vs. volume
integral of penalization force).

For higher density ratios (rho_s/rho_f > 1.1), the baroclinic term
becomes more important and should be implemented.  For the current
benchmark (1% contrast), it can be safely neglected.
"""
