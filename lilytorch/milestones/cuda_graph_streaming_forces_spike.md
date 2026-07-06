# CUDA-graph streaming + force readout — spike → shipped default

Date: 2026-07-06 · branch `warp_port` · status: **DONE — graphs are the
unconditional default on CUDA; no config flags exist**

## TL;DR

* **Streaming body_update**: graph replay is the default on CUDA (was opt-in
  `kernel_cuda_graph`, now flagless). Eager fallback only for velocity-blend
  and CPU. (`BDIMhandler._launch_body_update` → facade bridge `use_graph`.)
* **Force readout** (`forces_method2{,_3d}` streaming path): graph replay via
  `forces.ForcesPostGraph` is the default on CUDA for BOTH submethods —
  n·δ *and* deltaH. Eager fallback only for variable viscosity (fresh
  `nu_rho` per step) and CPU. No flags.
* **deltaH rewritten to be graphable**: the ∂H pressure pass used to compute
  a union AABB on the host from `aabb_lo.to("cpu")` (a hidden per-step D2H
  sync) and launch with a per-step dim. It is now a **full-grid static
  launch**; cells outside the ∂H band early-return at `gH == 0`, so results
  are identical and the pass captures cleanly. This also removed the hidden
  sync from the eager path.
* **Kernel micro-opt**: union pre-cull in `forces_post_{2,3}d_kernel` — one
  `sdf_cc` load culls far-outside fan threads before the (tri/bi)linear
  body-table gather (`sdf_cc = min_b s_b`, so `sdf_cc >= band_hi` implies
  every body is out of band; exactly result-preserving).

## Numbers (RTX 4080 SUPER, f32, B=8)

| config | eager submit | graph submit | graph wall (GPU) | parity |
|---|---|---|---|---|
| 2-D 384×256 | 95.6 µs | **12.4 µs (7.7×)** | 17.1 µs | 2.7e-15 |
| 3-D 96×64×64 | 105.7 µs | **19.1 µs (5.5×)** | 61.8 µs | 1.8e-16 |

The eager readout was 100 % host-bound (submit == wall). Graph submit is now
mostly the 3 async staging copies of kin/aabb. Suite: 228 passed / 1 skipped
with graphs default-on — i.e. every streaming/FSI/parity test now runs
through the graph paths.

## "Is the Warp force port as optimal as it could be?" — answer

Host side: yes now — the ~95–105 µs/call eager submit floor (13 zero-copy
`wp.array` wraps + `wp.launch` marshalling + `out.zero_()`) is replaced by a
~3 µs replay plus ~10–15 µs of staging copies. GPU side: the fan kernel costs
17 µs (2-D) / 62 µs (3-D) per step — well under 1 % of a coupled step
(salamander 2-D ≈ 6.2 ms/step). The union pre-cull bought the easy ~8 % in
3-D. Remaining theoretical levers — striped atomics to cut f64 atomic
contention on the 6/12 per-body accumulators, tile-level pre-reduction —
would shave tens of µs at most; not worth the complexity at <1 % of a step.

## Design (why static-shape capture is correct)

Fanned kernels: launch `B·max_vol` threads (n·δ) or full grid (∂H); each
thread decodes its (body, cell) from device-resident `aabb_lo/aabb_dim` and
early-returns outside the body's actual AABB volume / band. The launch config
is static while the worked region is data-driven, so a captured graph stays
correct for any pose with per-body AABB volume ≤ the frozen watermark.
`max_vol` is a grow-only watermark; growth drops graphs and recaptures
(`ForcesPostGraph._stage`). Padding = early-returning threads.

Pointer stability (the actual hard part, measured on a live loop:
u/v churn 19 unique signatures per 20 steps):

* kin/aabb: fresh tensors each step → staged into persistent buffers with
  async `copy_`.
* u/v[/w]: read from the persistent `self._vel` rows — at the only call
  sites, `self.u0, self.v0[,w0] = u, v[,w]` has just copied into them
  (pointer-stable, content-identical, zero extra copies).
* p: signature-keyed graph cache (≤8 signatures; allocator-stable in
  practice). Churn degrades to the eager launch — never wrong results — and
  warns once after 200 fallbacks with zero replays.
* sdf_cc / out / nu_rho scalar: persistent by construction.

## Files

* `src/forces.py` — `ForcesPostGraph`; graph branches in
  `forces_method2{,_3d}`; static full-grid deltaH kernels + launches; union
  pre-cull in both n·δ kernels.
* `integration/BDIMhandler.py` — `graph_mode = sdf_val.is_cuda and not blend`
  (flagless default).
* `src/solver.py`, `examples/base_sim_config.py`, salamander example — flag
  parsing/plumbing REMOVED (`kernel_cuda_graph` / `force_cuda_graph` no
  longer exist; stale YAMLs carrying the key are silently ignored).
* `tests/test_forces.py` — graph-vs-eager parity over an 8-step moving-pose
  sim (fresh per-step tensors, live-field mutation, per-step SDF re-stream,
  watermark growth → recapture), n·δ + deltaH, 2-D/3-D, f32/f64.
* `benchmarks/bench_forces_graph.py`.

## Not yet validated

A real FARMS-coupled run (needs MuJoCo scene; the suite's synthetic +
solver-level coverage is green). On the first salamander/zebrafish run, check
`fluid_solver._forces_post_graph_2d.replays` grows ~1/step and watch for the
ForcesPostGraph churn RuntimeWarning; compare drag records vs a pre-change
commit at ≤1e-9. Implicit coupling (`body.coupling`) runs multiple readouts
per step — same counters apply.
