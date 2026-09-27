"""Runnable locally using ONLY Python stdlib. These are not numerical tests."""
import ast
import json
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from external_baselines.patchcore_official_eval import protocol
from external_baselines.patchcore_official_eval.__main__ import build_parser, validate_args
from external_baselines.patchcore_official_eval.storage import start_stage, finish_stage, read_json

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]


class LocalChecks(unittest.TestCase):
    def test_all_sources_parse(self):
        for path in PACKAGE.rglob('*.py'):
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_cli_help_without_torch(self):
        for command in ([], ['check'], ['benchmark'], ['fit'], ['predict'], ['evaluate']):
            run = subprocess.run([sys.executable, '-m', 'external_baselines.patchcore_official_eval',
                                  *command, '--help'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_fixed_configuration(self):
        p = protocol.PROTOCOL
        self.assertEqual((p['target_embed_dimension'], p['projection_dimension'], p['coreset_ratio']), (1024, 128, .1))
        self.assertEqual(p['anomaly_scorer_num_nn'], 1)
        self.assertEqual(p['input_hw'], [256, 256])
        self.assertEqual(p['seed'], 42)
        self.assertEqual([len(protocol.CATEGORIES[k]) for k in ('btad', 'mvtec', 'visa')], [3, 15, 12])

    def test_non_square_geometry(self):
        self.assertEqual(protocol.evaluation_hw((301, 509)), (75, 127))
        with self.assertRaises(ValueError):
            protocol.evaluation_hw((3, 20))

    def test_no_overwrite_or_incompatible_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fit'
            ident = dict(protocol=protocol.PROTOCOL)
            self.assertTrue(start_stage(path, ident))
            with self.assertRaises(FileExistsError):
                start_stage(path, ident, True)
            finish_stage(path, ident)
            self.assertFalse(start_stage(path, ident, True))
            with self.assertRaises(FileExistsError):
                start_stage(path, dict(protocol='other'), True)
            self.assertEqual(read_json(path/'complete.json')['status'], 'complete')

    def test_formal_cpu_rejected(self):
        for command in ('fit', 'predict'):
            cmd = [command, '--dataset', 'btad', '--dataset-root', '/data', '--output-dir', '/output', '--cpu-check']
            if command == 'fit':
                cmd += ['--backbone-weights', '/weights.pth']
            args = build_parser().parse_args(cmd)
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_evaluate_needs_no_data_or_weights(self):
        args = build_parser().parse_args(['evaluate', '--dataset', 'btad', '--output-dir', '/output'])
        self.assertIsNone(args.dataset_root)
        self.assertFalse(hasattr(args, 'backbone_weights'))
        tree = ast.parse((PACKAGE/'evaluation.py').read_text())
        imported = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(x in ('adapter', 'official', 'patchcore') for x in imported))

    def test_protocol_and_benchmark_mismatch_rejected(self):
        state = dict(protocol=protocol.PROTOCOL, dataset='btad', category='01', benchmark=False)
        protocol.compatible(state, 'btad', '01')
        with self.assertRaises(ValueError):
            protocol.compatible(state, 'btad', '02')
        with self.assertRaises(ValueError):
            protocol.compatible(state, 'btad', '01', benchmark=True)

    def test_server_command_manifest_parses(self):
        examples = json.loads((PACKAGE/'command_examples.json').read_text())
        parser = build_parser()
        for command in examples:
            args = parser.parse_args(command)
            validate_args(args)
        self.assertEqual({x[0] for x in examples}, {'check', 'benchmark', 'fit', 'predict', 'evaluate'})

    def test_pinned_source(self):
        from external_baselines.patchcore_official_eval.official import verify_source
        self.assertEqual(verify_source()['commit'], protocol.SOURCE['commit'])

    def test_readme_cli_commands_parse(self):
        from external_baselines.patchcore_official_eval.server_commands import parser as server_parser, phase_arguments
        text = (PACKAGE/'README.md').read_text(encoding='utf-8').replace('\\\n', ' ')
        values = dict(BTAD_ROOT='/data/btad', MVTEC_ROOT='/data/mvtec', VISA_ROOT='/data/visa',
                      RUNS='/runs/new', WEIGHTS='/weights/v1.pth', GPU_MODEL='0', GPU_CORESET='1')
        count = 0
        for line in text.splitlines():
            if not line.startswith('python -m external_baselines.patchcore_official_eval '):
                if not line.startswith('python -m external_baselines.patchcore_official_eval.server_commands '):
                    continue
            for key, value in values.items():
                line = line.replace('$'+key, value)
            argv = shlex.split(line)
            if argv[2].endswith('server_commands'):
                args = server_parser().parse_args(argv[3:])
                phases = ['fit', 'predict', 'evaluate'] if args.phase == 'full' else [args.phase]
                for phase in phases:
                    validate_args(build_parser().parse_args(phase_arguments(args, phase)))
            else:
                validate_args(build_parser().parse_args(argv[3:]))
            count += 1
        self.assertGreaterEqual(count, 11)

    def test_server_dry_run_is_inert(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            run = subprocess.run([sys.executable, '-m',
                'external_baselines.patchcore_official_eval.server_commands', '--phase', 'full',
                '--dataset', 'btad', '--dataset-root', str(Path(tmp)/'data'),
                '--output-dir', str(output), '--weights', str(Path(tmp)/'v1.pth'),
                '--gpu-model', '0', '--gpu-coreset', '1', '--dry-run'],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(run.stdout.strip().splitlines()), 3)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_server_log_exit_and_dependency_stop(self):
        from external_baselines.patchcore_official_eval import server_commands
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            argv = ['server_commands', '--phase', 'full', '--dataset', 'btad',
                    '--dataset-root', str(Path(tmp)/'data'), '--output-dir', str(output),
                    '--weights', str(Path(tmp)/'weights.pth'), '--gpu-model', '0', '--gpu-coreset', '1']
            process = MagicMock()
            process.stdout = iter(['synthetic failure\n'])
            process.wait.return_value = 7
            with patch.object(sys, 'argv', argv), patch.object(server_commands.subprocess, 'Popen', return_value=process) as popen:
                with self.assertRaises(SystemExit) as error:
                    server_commands.main()
            self.assertEqual(error.exception.code, 7)
            self.assertEqual(popen.call_count, 1)
            logs = list((Path(tmp)/'run_logs').glob('*/fit.exit.json'))
            self.assertEqual(len(logs), 1)
            self.assertEqual(read_json(logs[0])['exit_code'], 7)
            self.assertEqual(logs[0].with_name('fit.log').read_text(), 'synthetic failure\n')
            self.assertFalse(list((Path(tmp)/'run_logs').glob('*/predict.log')))


if __name__ == '__main__':
    unittest.main()
