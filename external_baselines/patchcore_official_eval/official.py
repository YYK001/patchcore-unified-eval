"""Import the pinned upstream source without altering it."""
import hashlib
import subprocess
import sys
from .protocol import OFFICIAL_ROOT, SOURCE


def verify_source():
    commit = subprocess.check_output(['git', '-C', str(OFFICIAL_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(OFFICIAL_ROOT), 'status', '--porcelain',
                                     '--', 'src', 'LICENSE', 'NOTICE'], text=True).strip()
    if commit != SOURCE['commit'] or dirty:
        raise RuntimeError(f'Pinned official source required: commit={commit}, changes={dirty}')
    return dict(SOURCE, checkout=str(OFFICIAL_ROOT))


def install_import_path():
    source = str(OFFICIAL_ROOT / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    import patchcore
    if not __import__('pathlib').Path(patchcore.__file__).resolve().is_relative_to(OFFICIAL_ROOT.resolve()):
        raise RuntimeError('Another patchcore package is already imported')


def verify_weights(path):
    # One backbone checksum, not a per-image/file hash protocol. Accept only the
    # original torchvision pretrained=True (V1) distribution, even if renamed.
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    if not value.startswith('95faca4d'):
        raise ValueError('Expected original wide_resnet50_2-95faca4d.pth (IMAGENET1K_V1), not DEFAULT/V2/repacked weights')
    return value
