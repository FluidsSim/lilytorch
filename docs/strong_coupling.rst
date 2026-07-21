.. _strong_coupling:

Strong (implicit) FSI coupling
==============================

LilyTorch couples the fluid to immersed bodies in one of two ways. The
default is a **weakly partitioned (explicit)** scheme; an opt-in
**strongly coupled (implicit)** scheme cures the *added-mass instability*
that limits the explicit scheme for light or neutrally-buoyant bodies.

.. contents:: On this page
   :local:
   :depth: 2


The added-mass instability
--------------------------

In the explicit scheme the fluid is advanced once per step, the
hydrodynamic loads are read off, and they are pushed onto the body
integrator (FARMS/MuJoCo, or the standalone ``apply_force_feedback``
path). The body then moves, but the fluid never re-solves at the new
pose within the step.

This staggered scheme is **provably unstable** when the added
(displaced-fluid) mass is comparable to or larger than the body mass —
i.e. exactly the regime of a solid in water (:math:`\rho_b \lesssim
\rho_f`). The only stabiliser in the explicit path is a constant temporal
low-pass on the force (``force_relaxation`` < 1), which damps but does not
remove the instability and smears the true transient. It is the root
cause of the sphere-drop blow-ups and of case-by-case
``force_relaxation`` tuning.

For a body whose displaced mass exceeds its own mass the explicit
fixed-point map has spectral radius :math:`>1` and **diverges**; no
constant under-relaxation factor can stabilise it.


Fixed-point formulation
-----------------------

Let the **coupling variable** :math:`\mathbf{x}` parametrise the
interface — for a rigid body the loads applied to it, or equivalently its
end-of-step kinematic state. Let :math:`\mathcal{H}` be the composed
solver for one time step:

.. math::

   \tilde{\mathbf{x}} = \mathcal{H}(\mathbf{x}) :
   \quad
   \text{impose } \mathbf{x}
   \;\to\; \text{solve fluid}
   \;\to\; \text{read loads}
   \;\to\; \text{integrate structure over } \Delta t .

A converged time step is the fixed point :math:`\mathbf{x}^\star =
\mathcal{H}(\mathbf{x}^\star)`, i.e. the root of the interface residual

.. math::
   :label: fsi-residual

   \mathbf{r}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} .

The explicit scheme takes a single sweep :math:`\mathbf{x}^{n+1} =
\mathcal{H}(\mathbf{x}^n)` (Gauss–Seidel) and stops. The implicit scheme
iterates :eq:`fsi-residual` to convergence using an **interface
accelerator**.


Acceleration schemes
---------------------

LilyTorch ships the preCICE family of accelerators
(:mod:`lilytorch.integration.fsi_coupling`). Given the input
:math:`\mathbf{x}_k` and the solver output
:math:`\tilde{\mathbf{x}}_k`, each returns the next input
:math:`\mathbf{x}_{k+1}`.

Constant under-relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \mathbf{x}_{k+1} = \mathbf{x}_k + \omega\,\mathbf{r}_k .

This is the explicit scheme's low-pass (``force_relaxation``). Diverges
once the fixed-point map is expansive.

Aitken (Irons–Tuck)
^^^^^^^^^^^^^^^^^^^^

Adaptive scalar relaxation, recomputed each iteration from the last two
residuals:

.. math::

   \omega_k = -\,\omega_{k-1}\,
   \frac{\langle \mathbf{r}_{k-1},\, \mathbf{r}_k - \mathbf{r}_{k-1}\rangle}
        {\lVert \mathbf{r}_k - \mathbf{r}_{k-1}\rVert^2},
   \qquad
   \mathbf{x}_{k+1} = \mathbf{x}_k + \omega_k\,\mathbf{r}_k .

No per-problem tuning; far more robust than a fixed :math:`\omega`.

IQN-ILS (quasi-Newton)
^^^^^^^^^^^^^^^^^^^^^^^

Interface Quasi-Newton with Inverse Least Squares (Degroote et al. 2009)
— the preCICE workhorse, and the LilyTorch default. Over the iterations
of a step it collects, for :math:`k\ge1`,

.. math::

   V = [\,\Delta\mathbf{r}_k,\ \Delta\mathbf{r}_{k-1},\ \dots\,],
   \quad \Delta\mathbf{r}_i = \mathbf{r}_i - \mathbf{r}_{i-1},
   \qquad
   W = [\,\Delta\tilde{\mathbf{x}}_k,\ \dots\,],
   \quad \Delta\tilde{\mathbf{x}}_i = \tilde{\mathbf{x}}_i - \tilde{\mathbf{x}}_{i-1},

and approximates the inverse interface Jacobian by the multi-secant
relation :math:`W \approx J^{-1} V`. The next input is the quasi-Newton
step

.. math::

   \min_{\mathbf{c}} \lVert V\mathbf{c} + \mathbf{r}_k\rVert
   \quad\text{(least squares, QR with column filtering)},
   \qquad
   \mathbf{x}_{k+1} = \tilde{\mathbf{x}}_k + W\mathbf{c} .

With enough secant columns this is a Newton step on
:math:`\mathbf{r}(\mathbf{x})=0` and converges super-linearly,
**independently of the mass ratio** — which is why it removes the
added-mass instability. Columns from the previous ``reuse`` time steps
warm-start the least-squares system.

On a stable rigid body (e.g. the coquerelle sphere drop,
:math:`\rho_b > \rho_f`) IQN-ILS converges in **2–4 sweeps per step**
(residual :math:`\sim 10^{-7}`) and reproduces the explicit trajectory.
For a light/neutrally-buoyant body, where the explicit scheme diverges,
it remains stable.


How the coupled step works (Option A)
-------------------------------------

In coupled mode ``BDIMhandler.step`` runs inside FARMS' ``before_step``,
*before* MuJoCo integrates. To sub-iterate without double-integrating, the
implicit step (:meth:`BDIMhandler._step_implicit`) runs the whole
fixed-point loop **internally** using throwaway MuJoCo predictions, each
undone by a checkpoint restore, and commits only the converged force:

.. code-block:: text

   checkpoint MuJoCo state (mjSTATE_INTEGRATION) and fluid fields (u,v,[w],p)
   x  <- last step's load vector (warm start)
   repeat:
       restore MuJoCo -> s_n ;  restore fluid -> fluid_n
       apply candidate load x to xfrc ;  mj_step ;  mj_forward   # predict pose s~
       advance fluid at s~ (reads physics pose source)  ->  loads x~
       r = x~ - x ;  if ||r|| < tol: break
       x <- accelerator.relax(x, x~)
   commit: restore MuJoCo -> s_n, leave xfrc = converged force
   FARMS then integrates s_n under that force exactly once (one step)

Each sweep restores both the fluid *and* the full MuJoCo state
(``mujoco.mj_getState``/``mj_setState`` with ``mjSTATE_INTEGRATION``), so
every iteration re-solves the *same* step from the *same* state, varying
only the imposed interface motion. The fluid reads the predicted pose
directly from ``physics.data`` (the ``"physics"`` pose source) so the
internal ``mj_step`` is immediately visible without a sensor round-trip.

.. note::

   Implicit mode requires ``cb_sub_steps = 1`` and the FFT/multigrid
   Poisson fix described in :ref:`fft-vs-multigrid-projection`. See
   ``lilytorch/integration/STRONG_COUPLING_FARMS_DESIGN.md`` for the full
   design and the FARMS-pipeline specifics.


Using the implicit solver
--------------------------

Coupled (FARMS) sims
^^^^^^^^^^^^^^^^^^^^^

Add a ``coupling`` block to the ``body`` section of the ``bdim_yaml``
(e.g. in a ``simulation_config.yaml`` or via ``BaseSimConfig.coupling``):

.. code-block:: yaml

   body:
     coupling:
       scheme: implicit        # "explicit" (default) | "implicit"
       accelerator: iqn-ils    # iqn-ils | aitken | constant
       reuse: 2                # IQN-ILS time-windows reused (0 disables)
       tol: 1.0e-4             # relative interface-residual tolerance
       max_iter: 30            # max coupling sweeps per step

For a :class:`~lilytorch.examples.base_sim_config.BaseSimConfig`
swimmer, set the attribute instead::

   self.coupling = {"scheme": "implicit", "accelerator": "iqn-ils",
                    "reuse": 2, "tol": 1e-4, "max_iter": 30}

``scheme: explicit`` (or omitting ``coupling``) keeps the default
weakly-coupled behaviour; ``force_relaxation`` is ignored in implicit
mode. Cost is ``~N`` fluid solves per step (``N`` = sweeps, typically
2–4 after warm-up).

Standalone sims
^^^^^^^^^^^^^^^

For a standalone (non-FARMS) rigid body, drive
:class:`~lilytorch.integration.strong_coupling.StrongCoupledFSI` with the
:class:`~lilytorch.integration.strong_coupling.FluidSolverAdapter` and a
body implementing the ``RigidBodyCoupling`` protocol
(``get_state`` / ``set_coupling_state`` / ``predict`` / ``commit``). See
``lilytorch/integration/fsi_rigid_body.py`` and the runnable
``demo_real_fsi.py`` (a light circle pushed through quiescent 2-D fluid:
explicit diverges, IQN-ILS converges).


References
----------

* J. Degroote, K.-J. Bathe, J. Vierendeels (2009),
  *Performance of a new partitioned procedure versus a monolithic
  procedure in fluid–structure interaction*, Computers & Structures 87,
  793–801.
* B. Uekermann et al., *The preCICE coupling library* — quasi-Newton
  acceleration.
