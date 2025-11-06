import os

def get_num_workers():
    if hasattr(os, "sched_getaffinity"):
        num_cpus = len(os.sched_getaffinity(0))
    else:
        num_cpus = os.cpu_count() or 1

    return max(1, min(8, num_cpus // 2))