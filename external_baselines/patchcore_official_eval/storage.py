"""Small atomic writers and fail-closed stage lifecycle; no result overwrites."""
import csv
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def npz_write(path, **values):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(temporary, path)


def start_stage(path, identity, skip=False):
    path = Path(path)
    if path.exists():
        complete = path / 'complete.json'
        if skip and complete.is_file():
            saved = read_json(complete)
            if saved.get('identity') == identity and saved.get('status') == 'complete':
                return False
        raise FileExistsError(f'{path}: existing/incomplete/incompatible output; use a NEW output directory')
    path.mkdir(parents=True)
    json_write(path / 'state.json', dict(status='running', identity=identity))
    return True


def finish_stage(path, identity, **extra):
    json_write(Path(path) / 'complete.json', dict(status='complete', identity=identity, **extra))
