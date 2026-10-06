#!/usr/bin/env python3
"""Drive pattern-rate probe: N worker threads doing blocking random preads.

Mirrors the real decode fanout shape (pool workers, one blocking pread each)
so its ceiling is the decode ceiling, not a QD-32 synthetic. O_DIRECT by
default (measures the drive, not readahead/page-cache); --buffered for the
page-cache path. 4096-byte aligned offsets/buffers throughout.

Usage: scripts/probe-drive-rate.py FILE [matrix...]
Matrix entry: bs=NUM,jobs=NUM[,secs=NUM][,buffered]
"""
import ctypes
import os
import random
import sys
import threading
import time

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
libc.posix_memalign.restype = ctypes.c_int
libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
libc.pread.restype = ctypes.c_ssize_t


def run_arm(path, bs, jobs, secs, buffered):
    flags = os.O_RDONLY
    if not buffered:
        flags |= getattr(os, "O_DIRECT", 0)
    fd = os.open(path, flags)
    size = os.fstat(fd).st_size
    buf = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf), 4096, bs) != 0:
        raise OSError("posix_memalign failed")
    total = [0] * jobs
    stop = threading.Event()
    max_off = (size - bs) // 4096

    def worker(wid):
        rng = random.Random(wid * 99991 + 7)
        pread = libc.pread
        b = buf
        n = 0
        while not stop.is_set():
            off = rng.randint(0, max_off) * 4096
            r = pread(fd, b, bs, off)
            if r <= 0:
                break
            n += r
        total[wid] = n

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(jobs)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    time.sleep(secs)
    stop.set()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    os.close(fd)
    gbs = sum(total) / 2**30 / wall
    print(f"bs={bs // 1024}k jobs={jobs} {'buffered' if buffered else 'direct':8s} "
          f"-> {gbs:6.2f} GB/s  ({sum(total) / 2**20:.0f} MiB in {wall:.1f}s)", flush=True)
    return gbs


def main():
    path = sys.argv[1]
    arms = sys.argv[2:] or [
        "bs=1048576,jobs=1", "bs=1048576,jobs=8", "bs=1048576,jobs=24",
        "bs=1048576,jobs=48", "bs=262144,jobs=24", "bs=4194304,jobs=8",
    ]
    for arm in arms:
        kv = {"secs": 15}
        buffered = False
        for part in arm.split(","):
            if part == "buffered":
                buffered = True
            else:
                k, v = part.split("=")
                kv[k] = int(v)
        run_arm(path, kv["bs"], kv["jobs"], kv["secs"], buffered)


if __name__ == "__main__":
    main()
