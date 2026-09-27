"""Stage wall time; separate logical CUDA peaks and raw physical NVML observations."""
from contextlib import contextmanager
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
import torch
from .storage import csv_write


def environment(devices):
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'faiss-gpu', 'faiss-cpu', 'timm', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed under this distribution name'
    gpus = []
    for name in sorted(set(devices)):
        d = torch.device(name)
        if d.type == 'cuda':
            p = torch.cuda.get_device_properties(d)
            free, total = torch.cuda.mem_get_info(d)
            gpus.append(dict(logical_device=str(d), model=p.name, total_bytes=total, free_bytes_at_start=free,
                             uuid=str(getattr(p, 'uuid', 'unavailable'))))
    return dict(python=platform.python_version(), platform=platform.platform(), versions=packages,
                torch_cuda=torch.version.cuda, devices=gpus, CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
                pid=os.getpid(), nvidia_smi_mapping='raw physical UUID/index; do not assume logical index = physical index')


class Resources:
    def __init__(self, directory, devices, category=None):
        self.directory = directory
        self.category = category
        self.devices = [torch.device(d) for d in sorted(set(devices)) if torch.device(d).type == 'cuda']
        self.rows = []

    def observe(self, stage):
        row = dict(category=self.category, stage=stage, unix_time=time.time(), pid=os.getpid())
        for key, query in (
            ('whole_cards', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free'),
            ('compute_processes', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory'),
        ):
            try:
                run = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=4)
                row[key] = run.stdout.strip() if run.returncode == 0 else 'unavailable: '+run.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                row[key] = 'unavailable: '+str(e)
        # Observed samples, not true allocator peaks. Physical UUIDs stay explicit.
        for line in row['whole_cards'].splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 6:
                try:
                    key = fields[1]+'_whole_card_used_MiB_observed_max'
                    self.observed[key] = max(self.observed.get(key, 0), float(fields[4]))
                except ValueError:
                    pass
        for line in row['compute_processes'].splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 3 and fields[1] == str(os.getpid()):
                try:
                    key = fields[0]+'_this_pid_used_MiB_observed_max'
                    self.observed[key] = max(self.observed.get(key, 0), float(fields[2]))
                except ValueError:
                    pass
        with (self.directory / 'nvidia_smi_samples.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')

    @contextmanager
    def measure(self, stage):
        for d in self.devices:
            torch.cuda.synchronize(d)
            torch.cuda.reset_peak_memory_stats(d)
        stop = threading.Event()
        self.observed = {}
        def poll():
            self.observe(stage)
            while not stop.wait(.5):
                self.observe(stage)
        thread = threading.Thread(target=poll, daemon=True) if self.devices else None
        if thread:
            thread.start()
        start = time.perf_counter()
        status = 'failed'
        try:
            yield
            status = 'complete'
        finally:
            for d in self.devices:
                torch.cuda.synchronize(d)
            elapsed = time.perf_counter()-start
            stop.set()
            if thread:
                thread.join()
            row = dict(category=self.category, stage=stage, status=status, seconds=elapsed, **self.observed,
                       nvml_sampling='nvidia-smi start then 0.5s + query latency; raw UUID; missing means unavailable, not zero')
            for d in self.devices:
                row[str(d)+'_allocated_peak_bytes'] = torch.cuda.max_memory_allocated(d)
                row[str(d)+'_reserved_peak_bytes'] = torch.cuda.max_memory_reserved(d)
            self.rows.append(row)
            csv_write(self.directory / 'resources.csv', self.rows)
