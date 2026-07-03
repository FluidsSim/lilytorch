"""Deprecated shim package.

The Warp single-source kernels were flattened up into ``lilytorch.src`` and the
public dispatch API now lives in :mod:`lilytorch.src.facade`.  This package now
only holds the not-yet-relocated test modules; it is emptied of runtime code and
will be removed once the tests move to ``lilytorch/tests``.
"""
