"""Minimal test for whole-step CUDA-graph capture.

Verifies that ``WholeStepGraphRunner`` can capture a sequence of Warp kernel
launches into ONE graph and replay it without CUDA error 900.

Usage:
    python lilytorch/tests/test_whole_step_capture.py
"""

import pytest
import torch
import warp as wp

from lilytorch.src.graph_capture import WholeStepGraphRunner, capturing


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    """Ensure Warp is initialised once before any test in this module."""
    wp.init()
    yield


# ── Minimal Warp kernels ────────────────────────────────────────────────────

@wp.kernel
def _add_kernel(a: wp.array(dtype=wp.float32),
                b: wp.array(dtype=wp.float32),
                out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


@wp.kernel
def _scale_kernel(x: wp.array(dtype=wp.float32),
                  s: float,
                  out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    out[tid] = x[tid] * s


def test_whole_step_capture_basic():
    """Capture (add + scale) into one graph; replay; compare with eager."""
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    device = "cuda:0"
    N = 1024
    a = torch.ones(N, device="cuda", dtype=torch.float32)
    b = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)
    out = torch.zeros(N, device="cuda", dtype=torch.float32)
    tmp = torch.zeros(N, device="cuda", dtype=torch.float32)

    wa = wp.from_torch(a)
    wb = wp.from_torch(b)
    wout = wp.from_torch(out)
    wtmp = wp.from_torch(tmp)

    # Eager reference: out = (a + b) * 3.0
    def issue_eager():
        wp.launch(_add_kernel, dim=N, inputs=[wa, wb, wtmp])
        wp.launch(_scale_kernel, dim=N, inputs=[wtmp, 3.0, wout])

    issue_eager()
    torch.cuda.synchronize()
    expected = out.clone()

    # Reset
    out.zero_()
    tmp.zero_()

    # ── Test WholeStepGraphRunner capture/replay ─────────────────────
    runner = WholeStepGraphRunner(max_graphs=4)
    key = (a.data_ptr(), b.data_ptr(), out.data_ptr(), tmp.data_ptr())
    issue_count = [0]

    def issue():
        issue_count[0] += 1
        with capturing():
            pass  # not nested — we're calling raw launches here
        # Simulate per-kernel runners seeing in_capture() == True:
        # they issue raw wp.launch.
        wp.launch(_add_kernel, dim=N, inputs=[wa, wb, wtmp])
        wp.launch(_scale_kernel, dim=N, inputs=[wtmp, 3.0, wout])

    # First call: eager (sighting 1), JIT warm-up
    runner.run(key, device, issue, stage=None)
    torch.cuda.synchronize()
    assert issue_count[0] == 1
    assert runner.eager == 1
    assert runner.captures == 0
    assert runner.replays == 0
    # Result should match expected (eager path)
    assert torch.allclose(out, expected), "eager result mismatch"

    # Reset
    out.zero_()
    tmp.zero_()

    # Second call: eager warmup + capture (sighting 2)
    runner.run(key, device, issue, stage=None)
    torch.cuda.synchronize()
    assert issue_count[0] == 3  # 1 (first) + 1 (warmup) + 1 (capture record)
    assert runner.captures == 1
    assert runner.replays == 0
    assert torch.allclose(out, expected), "captured result mismatch"

    # Reset
    out.zero_()
    tmp.zero_()

    # Third call: replay
    runner.run(key, device, issue, stage=None)
    torch.cuda.synchronize()
    assert issue_count[0] == 3  # issue() NOT called during replay
    assert runner.replays == 1
    assert torch.allclose(out, expected), "replay result mismatch"

    # Additional replays
    for _ in range(5):
        out.zero_()
        tmp.zero_()
        runner.run(key, device, issue, stage=None)
        torch.cuda.synchronize()
        assert torch.allclose(out, expected), "replay result mismatch"
    assert runner.replays == 6

    print("PASS: whole-step capture basic test")
    print(f"  captures={runner.captures}, replays={runner.replays}, eager={runner.eager}")


def test_whole_step_capture_with_staging():
    """Capture with a staging callback (simulates bdim rect staging)."""
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    device = "cuda:0"
    N = 512

    # Persistent "staging" buffer (simulates bdim rect)
    staged = torch.zeros(1, device="cuda", dtype=torch.float32)
    wstaged = wp.from_torch(staged)

    @wp.kernel
    def _read_staged_kernel(val: wp.array(dtype=wp.float32),
                            out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        out[tid] = val[0]

    out = torch.zeros(N, device="cuda", dtype=torch.float32)
    wout = wp.from_torch(out)

    source = torch.tensor([42.0], device="cpu", dtype=torch.float32)
    staged_val = [1.0]  # mutable so stage() can change it

    def stage():
        staged_val[0] += 1.0
        staged.copy_(torch.tensor([staged_val[0]], device="cuda", dtype=torch.float32))
        torch.cuda.synchronize(device)

    def issue():
        wp.launch(_read_staged_kernel, dim=N, inputs=[wstaged, wout])

    runner = WholeStepGraphRunner(max_graphs=4)
    key = (out.data_ptr(),)  # staged buffer not in key (contents change)

    # Sighting 1: eager
    runner.run(key, device, issue, stage=stage)
    torch.cuda.synchronize()
    assert out[0].item() == 2.0  # stage() incremented to 2.0
    assert runner.eager == 1

    # Sighting 2: capture
    out.zero_()
    runner.run(key, device, issue, stage=stage)
    torch.cuda.synchronize()
    assert out[0].item() == 3.0  # stage() incremented to 3.0
    assert runner.captures == 1

    # Replays: each should see the freshly-staged value
    for expected_val in [4.0, 5.0, 6.0]:
        out.zero_()
        runner.run(key, device, issue, stage=stage)
        torch.cuda.synchronize()
        assert out[0].item() == expected_val, \
            f"replay: expected {expected_val}, got {out[0].item()}"
    assert runner.replays == 3

    print("PASS: whole-step capture with staging test")
    print(f"  captures={runner.captures}, replays={runner.replays}, eager={runner.eager}")


def test_whole_step_capture_no_cuda_error_900():
    """Run many capture/replay cycles — no CUDA error 900 should occur."""
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    device = "cuda:0"
    N = 2048

    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.zeros(N, device="cuda", dtype=torch.float32)
    tmp = torch.zeros(N, device="cuda", dtype=torch.float32)

    wa = wp.from_torch(a)
    wb = wp.from_torch(b)
    wout = wp.from_torch(out)
    wtmp = wp.from_torch(tmp)

    def issue():
        wp.launch(_add_kernel, dim=N, inputs=[wa, wb, wtmp])
        wp.launch(_scale_kernel, dim=N, inputs=[wtmp, 2.0, wout])

    runner = WholeStepGraphRunner(max_graphs=4)
    key = (a.data_ptr(), b.data_ptr(), out.data_ptr(), tmp.data_ptr())

    # Warmup: 2 eager calls → capture on 2nd sighting
    for i in range(2):
        out.zero_()
        tmp.zero_()
        runner.run(key, device, issue, stage=None)
    torch.cuda.synchronize()
    assert runner.captures >= 1, "graph should have been captured"

    # Many replays — stress test for CUDA error 900
    for i in range(100):
        out.zero_()
        tmp.zero_()
        runner.run(key, device, issue, stage=None)
    torch.cuda.synchronize()
    assert runner.replays >= 100

    print("PASS: no CUDA error 900 over 100 replays")
    print(f"  captures={runner.captures}, replays={runner.replays}, eager={runner.eager}")


if __name__ == "__main__":
    wp.init()
    test_whole_step_capture_basic()
    test_whole_step_capture_with_staging()
    test_whole_step_capture_no_cuda_error_900()
    print("\nAll tests passed!")
