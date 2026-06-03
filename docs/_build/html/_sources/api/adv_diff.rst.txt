.. _api_adv_diff:

``adv_diff`` — Advection–Diffusion (legacy)
============================================

.. warning::

   ``adv_diff.py`` is **legacy and no longer used** by the live code path.
   It was split into :mod:`lilytorch.src.advection` (convective schemes,
   flux, ``AdvDiffSolver``) and :mod:`lilytorch.src.diffusion` (diffusion
   operators). The file is kept on disk for reference only — do not import,
   edit, or extend it; add new advection/diffusion functionality to the
   split modules instead.

See :doc:`advection` and :doc:`diffusion`.
