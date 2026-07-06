"""Headless, short, instrumented salamander 2-D coupled run — per-step profile.

Measures the reference coupled case (salamander_gamepad 2-D, 1024x512) with
per-phase sync-attributed + submit-only timings, eager-launch/replay/host-sync
counters and a torch.profiler CUDA window.  See
milestones/perf_host_bound_plan.md for baseline numbers and protocol.

Usage:
    python lilytorch/benchmarks/prof_coupled_step.py
Report JSON lands next to this file (prof_coupled_step_report.json) unless
PROF_REPORT is set in the environment.
"""
import os
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lilytorch.examples.salamander_gamepad.gen_configs_swim_2d import SimConfig

PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "_prof_coupled_step_patch.py")


class ProfConfig(SimConfig):
    def __init__(self):
        super().__init__()
        self.headless = True
        self.n_iterations = 450
        self.bdim_nt = self.n_iterations
        self.save_frames = False
        self.stack_folder = "salamander_prof"

    def extra_simulation_extensions(self, output_folder):
        return []

    def _extra_run_patch(self):
        return f"exec(open('{PATCH}').read());"


if __name__ == "__main__":
    os.environ.setdefault(
        "PROF_REPORT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "prof_coupled_step_report.json"))
    ProfConfig().run()
