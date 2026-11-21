import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import nullcontext


MODULE = Path(__file__).resolve().parents[1] / 'train_run.py'


class FormalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('train_run', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_launcher_saves_checkpoints_and_uses_agreed_training_settings(self):
        source = (MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8')
        launcher = self.mod.make_training_launcher(source, Path('/tmp/formal'), ['GPU-3', 'GPU-4', 'GPU-5', 'GPU-6'])
        for text in ['n_gpu=4', 'bs=32', 'save_interval=500', '--vla.max_steps 2000',
                     '--vla.global_batch_size 128', '--vla.learning_rate 2e-5', '--group_size 16',
                     'CUDA_VISIBLE_DEVICES=GPU-3,GPU-4,GPU-5,GPU-6 ',
                     '--master_port=29834', '--tee 3', '--log-dir',
                     'is_resume=False', 'resume_step=0', 'resume_epoch=0']:
            self.assertIn(text, launcher)
        self.assertNotIn('--worker', launcher)
        self.assertNotIn('perf_compare.py', launcher)
        self.assertNotIn('--wandb_project', launcher)

    def test_generated_launcher_executes_original_train_entry_with_checkpoint_interval(self):
        bash = 'F:/Git/bin/bash.exe' if os.name == 'nt' else 'bash'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = (MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8')
            script = root / 'train.sh'
            script.write_text(self.mod.make_training_launcher(source, Path('/tmp/formal'),
                              ['GPU-3', 'GPU-4', 'GPU-5', 'GPU-6'], 5000, 1000), newline='\n')
            # Replace torchrun only in this test with a Bash function to capture expanded arguments.
            result = subprocess.run([bash, '-c', 'torchrun() { printf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"; }; source "$1"',
                                     'test', str(script)], check=True, capture_output=True, text=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[0], 'GPU-3,GPU-4,GPU-5,GPU-6')
            self.assertIn('train.py', args)
            self.assertEqual(args[args.index('--save_interval') + 1], '1000')
            self.assertEqual(args[args.index('--vla.max_steps') + 1], '5000')
            self.assertEqual(args[args.index('--vla.global_batch_size') + 1], '128')
            self.assertEqual(args[args.index('--tee') + 1], '3')

    def test_gpu_snapshots_keep_process_ids_and_survive_monitor_failure(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            with patch.object(self.mod.subprocess, 'run', return_value=SimpleNamespace(stdout='GPU 6: PID 777 memory 100000 MiB\n', stderr='')), \
                 patch.object(self.mod, 'training_processes', return_value={123: 'train worker'}):
                self.mod.record_gpu_snapshot(directory, Path('/data/MemoryVLA'))
            log = (directory / 'gpu_monitor.log').read_text()
            self.assertIn('PID 777', log)
            self.assertIn('train.py PIDs: [123]', log)
            with patch.object(self.mod.subprocess, 'run', side_effect=OSError('nvidia-smi unavailable')):
                self.mod.record_gpu_snapshot(directory, Path('/data/MemoryVLA'))
            self.assertIn('nvidia-smi unavailable', (directory / 'gpu_monitor.log').read_text())

    def test_stop_scope_excludes_video_jobs_other_roots_and_other_tools(self):
        root = Path('/data/MemoryVLA').resolve()
        tool = str(self.mod.BENCHMARK)
        self.assertTrue(self.mod.is_benchmark_process(root, root, ['python', tool, '--worker']))
        self.assertTrue(self.mod.is_benchmark_process(root, Path('/tmp'),
                        ['python', tool, '--root', str(root), '--gpus', '4']))
        for cwd, argv in [(root, ['python', 'run_video_full.py']),
                          (Path('/data/other'), ['python', tool, '--worker']),
                          (root, ['python', '/other/perf_compare.py']),
                          (root, ['bash', '-c', tool]),
                          (Path('/tmp'), ['python', tool, '--root', '/data/other'])]:
            self.assertFalse(self.mod.is_benchmark_process(root, cwd, argv))

    def test_checkpoint_check_requires_every_requested_save(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / 'runs/memvla_robomme_4gpu_bs32--image_aug/checkpoints'
            run_dir.mkdir(parents=True)
            for step in (500, 1000, 1500):
                (run_dir / f'step-{step:06d}-epoch-00-loss=0.01.pt').write_bytes(b'checkpoint')
            with self.assertRaisesRegex(RuntimeError, '2000'):
                self.mod.check_checkpoints(Path(d))
            last = run_dir / 'step-002000-epoch-00-loss=0.01.pt'
            last.touch()
            with self.assertRaises(RuntimeError):
                self.mod.check_checkpoints(Path(d))
            last.write_bytes(b'checkpoint')
            self.assertEqual(len(self.mod.check_checkpoints(Path(d))), 4)

    def test_stop_rechecks_pid_identity_before_signalling(self):
        with patch.object(self.mod, 'benchmark_processes', side_effect=[{123: 'old'}, {}]), \
             patch.object(self.mod, 'start_id', return_value='reused'), \
             patch.object(self.mod.os, 'kill') as kill:
            self.mod.stop_benchmarks(Path('/data/MemoryVLA'))
            kill.assert_not_called()

    def test_main_stops_benchmark_then_launches_formal_training_and_checks_outputs(self):
        for code, old_training in [(0, {}), (1, {}), (0, {999: 'existing'})]:
            with self.subTest(code=code, old_training=old_training), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                source = root / 'script/train/robomme/train.sh'
                source.parent.mkdir(parents=True)
                source.write_text((MODULE.parent / 'tests/fixtures/train.sh').read_text(encoding='utf-8'))
                order = []

                def launch(command, **kwargs):
                    order.append('launch')
                    self.assertEqual(kwargs['cwd'], root)
                    self.assertTrue(kwargs['start_new_session'])
                    self.assertNotIn('--worker', Path(command[1]).read_text())
                    if code == 0:
                        directory = Path(command[1]).parent
                        ckpts = directory / 'runs/memvla_robomme_4gpu_bs32--image_aug/checkpoints'
                        ckpts.mkdir(parents=True)
                        for step in (500, 1000, 1500, 2000):
                            (ckpts / f'step-{step:06d}-epoch-00-loss=0.01.pt').write_bytes(b'checkpoint')
                    return SimpleNamespace(wait=lambda: code)

                fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a: None)
                with patch.dict('sys.modules', {'fcntl': fake_fcntl}), \
                     patch.object(self.mod.sys, 'platform', 'linux'), \
                     patch.object(self.mod, 'resolve_devices', return_value=['GPU-3', 'GPU-4', 'GPU-5', 'GPU-6']), \
                     patch.object(self.mod, 'stop_benchmarks', side_effect=lambda root: order.append('stop')), \
                     patch.object(self.mod, 'training_processes', return_value=old_training), \
                     patch.object(self.mod, 'monitor_gpu', return_value=nullcontext()), \
                     patch.object(self.mod.subprocess, 'run'), \
                     patch.object(self.mod.subprocess, 'Popen', side_effect=launch) as popen, \
                     patch.object(self.mod.signal, 'signal'), patch('sys.stdout', io.StringIO()):
                    if old_training:
                        with self.assertRaisesRegex(RuntimeError, 'Existing train.py'):
                            self.mod.main(['--root', str(root), '--stop-benchmark'])
                        popen.assert_not_called()
                    elif code:
                        with self.assertRaises(SystemExit) as error:
                            self.mod.main(['--root', str(root), '--stop-benchmark'])
                        self.assertEqual(error.exception.code, 1)
                    else:
                        self.mod.main(['--root', str(root), '--stop-benchmark'])
                        directory = Path((root / 'log/robomme/latest_train_dir.txt').read_text().strip())
                        self.assertTrue((directory / 'completed.json').exists())
                self.assertEqual(order, ['stop'] if old_training else ['stop', 'launch'])


if __name__ == '__main__':
    unittest.main()
