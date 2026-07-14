"""Debug helper: turn a silent host-RAM OOM SIGKILL into a traceback.

Injected into the FARMS subprocess via a config's ``_extra_run_patch``::

    def _extra_run_patch(self):
        return "import lilytorch.integration._setup_rss as _r;_r.install();"

``install()`` enables ``faulthandler`` and starts an RSS watchdog thread:
when resident memory crosses ``rss_dump_gb`` it dumps every thread's stack
(so the runaway allocation site is visible), and when it crosses
``rss_abort_gb`` it dumps again and hard-exits before the kernel OOM-killer
SIGKILLs the process with no diagnostics.

NOTE: an RLIMIT_AS cap does NOT work for this in a CUDA process — the CUDA
driver reserves address space far beyond RSS (VIRT >> RSS), so any cap low
enough to catch a host-RAM runaway also breaks legitimate device
allocations (observed: "Failed to allocate 4 bytes on device 'cuda:0'"
mid-step with a 48 GB cap while RSS was only ~3.5 GB).
"""

import faulthandler
import os
import sys
import threading
import time

_GB = 1024 ** 3
_PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE / _GB


def _watchdog(dump_gb, abort_gb):
    dumped = False
    while True:
        rss = _rss_gb()
        if rss > dump_gb and not dumped:
            dumped = True
            sys.stderr.write(
                f"\n[_setup_rss] RSS {rss:.1f} GB crossed {dump_gb} GB — "
                "dumping all thread stacks:\n")
            faulthandler.dump_traceback(file=sys.stderr)
            sys.stderr.flush()
        if rss > abort_gb:
            sys.stderr.write(
                f"\n[_setup_rss] RSS {rss:.1f} GB crossed the abort limit "
                f"{abort_gb} GB — dumping stacks and exiting before the "
                "OOM-killer:\n")
            faulthandler.dump_traceback(file=sys.stderr)
            sys.stderr.flush()
            os._exit(42)
        time.sleep(0.2)


def install(rss_dump_gb=16, rss_abort_gb=26):
    faulthandler.enable()
    sys.stderr.write(
        f"[_setup_rss] faulthandler on; RSS stack-dump at {rss_dump_gb} GB, "
        f"abort at {rss_abort_gb} GB\n")
    t = threading.Thread(target=_watchdog, args=(rss_dump_gb, rss_abort_gb),
                         daemon=True)
    t.start()
