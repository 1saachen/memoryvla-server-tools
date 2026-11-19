import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'perf_compare.py'


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('perf_compare', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_selected_cases_launch_and_report_completion(self):
        original = (MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8')
        for extra, expected, code, label in [
            ([], [2, 4], 0, 'COMPARISON_COMPLETE'),
            (['--gpus', '4'], [4], 0, 'BENCHMARK_COMPLETE'),
            (['--gpus', '4'], [4], 1, 'BENCHMARK_INCOMPLETE'),
        ]:
            with self.subTest(extra=extra, code=code), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                source = root / 'script/train/robomme/train.sh'
                source.parent.mkdir(parents=True)
                source.write_text(original)
                launched = []

                def launch(command, **kwargs):
                    launcher = Path(command[1])
                    cards = int(launcher.stem.removeprefix('perf'))
                    launched.append(cards)
                    if code == 0:
                        run_dir = launcher.parent / 'runs' / f'perf{cards}--image_aug'
                        run_dir.mkdir(parents=True)
                        (run_dir / 'config.json').write_text(json.dumps({
                            'group_size': 16, 'vla': {'expected_world_size': cards,
                            'per_device_batch_size': 64 // cards, 'global_batch_size': 64}}))
                        rows = [{'VLA Train/Step': s, 'VLA Train/Step Time': 2,
                                 'VLA Train/Loss': 0.01} for s in range(1, 101)]
                        (run_dir / f'{run_dir.name}.jsonl').write_text(
                            ''.join(json.dumps(r) + '\n' for r in rows))
                    return SimpleNamespace(wait=lambda **kw: code)

                fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a: None)
                output = io.StringIO()
                with patch.dict('sys.modules', {'fcntl': fake_fcntl}), \
                     patch.object(self.mod.sys, 'platform', 'linux'), \
                     patch.object(self.mod, 'training_processes', return_value={}), \
                     patch.object(self.mod.subprocess, 'run'), \
                     patch.object(self.mod.subprocess, 'Popen', side_effect=launch), \
                     patch('sys.stdout', output):
                    if code:
                        with self.assertRaises(SystemExit) as error:
                            self.mod.main(['--root', str(root)] + extra)
                        self.assertEqual(error.exception.code, 1)
                    else:
                        self.mod.main(['--root', str(root)] + extra)
                self.assertEqual(launched, expected)
                self.assertIn(label + ':', output.getvalue())
                directory = Path((root / 'log/robomme/latest_perf_dir.txt').read_text().strip())
                self.assertEqual(sorted(p.name for p in directory.glob('perf*.sh')),
                                 [f'perf{c}.sh' for c in expected])
                self.assertEqual([r['gpus'] for r in json.loads((directory / 'summary.json').read_text())], expected)

    def test_launcher_preserves_training_options_and_complete_groups(self):
        original = (MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8')
        for cards, batch in [(2, 32), (4, 16)]:
            script = self.mod.make_launcher(original, Path('/tmp/bench'), cards, batch)
            self.assertIn(f'n_gpu={cards}', script)
            self.assertIn(f'bs={batch}', script)
            self.assertIn('--vla.max_steps 100', script)
            self.assertIn('--group_size 16', script)
            self.assertIn('--hf_token "${hf_token}"', script)
            self.assertIn('--trackers \'["jsonl"]\'', script)
            self.assertIn('--vla.learning_rate 2e-5', script)
            self.assertIn('--repeated_diffusion_steps ${dp_step}', script)
            self.assertIn('--worker', script)
            self.assertNotIn('--wandb_entity', script)
            self.assertEqual(cards * batch, 64)

    def test_changed_launcher_is_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            self.mod.make_launcher('echo unexpected', Path('/tmp/bench'), 2, 32)

    def test_server_launcher_with_existing_fixes(self):
        original = (MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8')
        original = original.replace('n_gpu=2', 'n_gpu=8').replace('bs=32', 'bs=8')
        original = original.replace('CUDA_VISIBLE_DEVICES=0,1 ', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ')
        original = original.replace('--nproc_per_node=2', '--nproc_per_node=8')
        original = original.replace("--wandb_project 'memvla'", '--trackers \'["jsonl"]\'')
        original = original.replace('--hf_token ${hf_token}', '--hf_token "${hf_token}"')
        original = 'export HF_ENDPOINT=https://huggingface.co\n' + original
        script = self.mod.make_launcher(original, Path('/tmp/bench'), 4, 16)
        self.assertEqual(script.count('--trackers'), 1)
        self.assertEqual(script.count('--group_size'), 1)
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1,2,3 ', script)
        self.assertIn('--nproc_per_node=4', script)
        self.assertIn('export HF_ENDPOINT=https://huggingface.co', script)

    def test_worker_skips_checkpoint_only_and_forwards_arguments(self):
        strategy = type('Strategy', (), {'save_checkpoint': lambda *a, **kw: 'saved'})
        fake_module = SimpleNamespace(FSDPStrategy=strategy)
        with patch.dict('sys.modules', {'training.strategies.fsdp': fake_module}):
            with patch.object(self.mod.sys, 'argv', ['perf_compare.py', '--worker', '--vla.max_steps', '100']):
                with patch('runpy.run_path') as run:
                    self.mod.run_worker()
                    self.assertEqual(self.mod.sys.argv, ['train.py', '--vla.max_steps', '100'])
                    self.assertIsNone(strategy.save_checkpoint())
                    run.assert_called_once_with('train.py', run_name='__main__')

    def test_statistics_ignore_warmup_and_reject_missing_or_nan_steps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'metrics.jsonl'
            rows = [{'VLA Train/Step': s, 'VLA Train/Step Time': 100 if s <= 20 else 2,
                     'VLA Train/Loss': 0.01} for s in range(1, 101)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result = self.mod.summarize(p)
            self.assertEqual(result['measured_steps'], 80)
            self.assertEqual(result['mean_step_seconds'], 2)
            self.assertEqual(result['samples_per_second'], 32)
            rows[-1]['VLA Train/Loss'] = float('nan')
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):
                self.mod.summarize(p)
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaises(ValueError):
                self.mod.summarize(p)

    def test_process_scope_excludes_other_repos_and_jobs(self):
        root = Path('/data/embodied-ai/yaolin/MemoryVLA')
        self.assertTrue(self.mod.is_training_process(root, root, ['python', 'train.py', '--x', '1']))
        self.assertTrue(self.mod.is_training_process(root, root, ['/env/bin/torchrun', 'train.py']))
        self.assertFalse(self.mod.is_training_process(root, Path('/data/other'), ['python', 'train.py']))
        self.assertFalse(self.mod.is_training_process(root, root, ['python', 'deploy.py']))
        self.assertFalse(self.mod.is_training_process(root, root, ['bash', '-c', 'echo train.py']))


if __name__ == '__main__':
    unittest.main()
