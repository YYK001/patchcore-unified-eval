"""CLI imports no numerical packages until after argument parsing (including help)."""
import argparse
from pathlib import Path
from .protocol import categories


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def build_parser():
    parser = argparse.ArgumentParser(description='Official PatchCore + full-frame256 external reevaluation; no server connection')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'benchmark', 'fit', 'predict', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', choices=['btad', 'mvtec', 'visa'], required=True)
        p.add_argument('--category', default='all')
        p.add_argument('--output-dir', type=Path, required=True, help='dedicated new run root; never existing experiment results')
        p.add_argument('--dataset-root', type=Path, required=name != 'evaluate',
                       help='evaluate needs this only when compressed GT has not yet been saved')
        p.add_argument('--skip-completed', action='store_true', help='skip only compatible fully completed categories')
        p.add_argument('--cpu-check', action='store_true', help='explicit nonformal CPU check/benchmark/evaluation only')
        p.add_argument('--metric-device', default='cuda:0')
        if name != 'evaluate':
            p.add_argument('--backbone-weights', type=Path, required=name in ('check', 'benchmark', 'fit'),
                           help='original torchvision IMAGENET1K_V1 file; predict uses saved shared copy')
            p.add_argument('--model-device', default='cuda:0')
            p.add_argument('--coreset-device', default='cuda:1')
            p.add_argument('--nn-device', default='cuda:0')
            p.add_argument('--batch-size', type=positive, default=8)
            p.add_argument('--num-workers', type=int, default=0)
            p.add_argument('--projection-chunk', type=positive, default=8192)
            p.add_argument('--distance-chunk', type=positive, default=65536)
            p.add_argument('--query-chunk', type=positive, default=4096)
            p.add_argument('--faiss-temp-mb', type=positive, default=256)
        if name == 'benchmark':
            p.add_argument('--benchmark-train-images', type=positive, default=2)
            p.add_argument('--benchmark-test-images', type=positive, default=2)
        if name == 'evaluate':
            p.add_argument('--evaluation-name', default='metrics', help='new name permits independent recomputation without overwrite')
    return parser


def validate_args(args):
    categories(args.dataset, args.category)
    if args.cpu_check and args.command in ('fit', 'predict'):
        raise ValueError('Formal fit/predict require CUDA; use benchmark --cpu-check for CPU checks')
    if getattr(args, 'num_workers', 0) < 0:
        raise ValueError('num-workers cannot be negative')
    if args.command == 'evaluate':
        name = args.evaluation_name
        if not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('evaluation-name must be one directory name')
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.dataset_root:
        args.dataset_root = args.dataset_root.expanduser().resolve()
        if (args.output_dir.is_relative_to(args.dataset_root)
                or args.dataset_root.is_relative_to(args.output_dir)):
            raise ValueError('Output and dataset roots must be disjoint')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    from .runner import run
    run(args)


if __name__ == '__main__':
    main()
