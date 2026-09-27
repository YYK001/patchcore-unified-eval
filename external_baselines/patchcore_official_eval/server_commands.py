"""Run MANUALLY on server. Stdlib stage logger; no SSH, upload, environment install.

--dry-run prints exact argv and performs no work. Every executed phase has a
unique log/exit JSON; any nonzero exit stops dependent phases immediately.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['btad', 'mvtec', 'visa'], required=True)
    p.add_argument('--category', default='all')
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--gpu-model', type=int, required=True)
    p.add_argument('--gpu-coreset', type=int, required=True)
    p.add_argument('--phase', choices=['check', 'benchmark', 'full', 'fit', 'predict', 'evaluate'], required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--projection-chunk', type=int, default=8192)
    p.add_argument('--distance-chunk', type=int, default=65536)
    p.add_argument('--query-chunk', type=int, default=4096)
    p.add_argument('--faiss-temp-mb', type=int, default=256)
    p.add_argument('--skip-completed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def phase_arguments(a, phase):
    command = [phase, '--dataset', a.dataset, '--category', a.category,
               '--dataset-root', a.dataset_root, '--output-dir', a.output_dir,
               '--metric-device', f'cuda:{a.gpu_model}']
    if phase != 'evaluate':
        command += ['--model-device', f'cuda:{a.gpu_model}', '--coreset-device', f'cuda:{a.gpu_coreset}',
                    '--nn-device', f'cuda:{a.gpu_model}', '--backbone-weights', a.weights,
                    '--batch-size', str(a.batch_size), '--projection-chunk', str(a.projection_chunk),
                    '--distance-chunk', str(a.distance_chunk), '--query-chunk', str(a.query_chunk),
                    '--faiss-temp-mb', str(a.faiss_temp_mb)]
    if a.skip_completed:
        command += ['--skip-completed']
    return command


def main():
    a = parser().parse_args()
    if min(a.gpu_model, a.gpu_coreset) < 0 or a.gpu_model == a.gpu_coreset:
        raise ValueError('Two different nonnegative GPU indices required')
    phases = ['fit', 'predict', 'evaluate'] if a.phase == 'full' else [a.phase]
    from .__main__ import build_parser, validate_args
    calls = []
    for phase in phases:
        argv = phase_arguments(a, phase)
        validate_args(build_parser().parse_args(argv))
        calls.append((phase, [sys.executable, '-m', 'external_baselines.patchcore_official_eval', *argv]))
    if a.dry_run:
        for _, command in calls:
            print(shlex.join(command))
        return
    # Sibling log directory: never pollutes a new run root before its ownership check.
    root = Path(a.output_dir).expanduser().resolve()
    log_dir = root.parent/(root.name+'_logs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    log_dir.mkdir(parents=True, exist_ok=False)
    for phase, command in calls:
        (log_dir/(phase+'.argv.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
        with (log_dir/(phase+'.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                code = process.wait()
                (log_dir/(phase+'.exit.json')).write_text(json.dumps(dict(exit_code=code, interrupted=True)), encoding='utf-8')
                raise
        (log_dir/(phase+'.exit.json')).write_text(json.dumps(dict(exit_code=code)), encoding='utf-8')
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
