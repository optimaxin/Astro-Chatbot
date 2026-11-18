"""Resident memory of this process, without a third-party dependency (for /health and the shadow checks)."""

from __future__ import annotations

import sys


def rss_mb() -> float | None:
    """Current resident set size in MiB, or None when it cannot be read on this platform."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="ascii") as fh:
                pages = int(fh.read().split()[1])
            import os

            return round(pages * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / 1048576, 1)
        else:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # macOS: bytes; others: KiB
            return round(peak / (1048576 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001 — a monitoring figure, never an error
        return None
    return None


__all__ = ["rss_mb"]
