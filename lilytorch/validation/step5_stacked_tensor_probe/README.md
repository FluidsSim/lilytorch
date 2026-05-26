# Step-5 stacked-tensor storage probe

CPU-only spike to test the hypothesis from `todo.md` Step 5 that replacing
per-axis field tensors on `FluidSolver` / `CompositeBody` with single
`(D, *grid_shape)` tensors is "free or better" on speed and memory.

- `REPORT.md` — side-by-side comparison tables and GO/NO-GO verdict.
- `probe.py`  — runner: 2-D cylinder + 3-D sphere analytical harnesses,
  stacked-shadow run, and microbenchmarks of the canonical hot ops.
- `_cpu_patches.py` — CPU-only runtime workarounds (probe-local; does
  not modify the source tree).
- `results.json` — raw numerical output of the most recent probe run.

Run with:

```
python -m lilytorch.validation.step5_stacked_tensor_probe.probe
```

**Decision: NO-GO**. See `REPORT.md` for details.
