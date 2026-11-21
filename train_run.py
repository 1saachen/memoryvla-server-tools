#!/usr/bin/env python3
"""Start a 4-GPU, batch-128 RoboMME training run."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

from perf_compare import make_launcher, resolve_devices, start_id, training_processes

BENCHMARK = Path(__file__).resolve().with_name('perf_compare.py')


def make_training_launcher(original, directory, devices, max_steps=2000, save_interval=500):
    text = make_launcher(original, directory, 4, 32, 128, devices)
    replacements = [
        (r'^[ \t]*run_id=.*$', 'run_id=memvla_robomme_4gpu_bs32'),
        (r'^[ \t]*save_interval=.*$', f'save_interval={save_interval}'),
        (r'^[ \t]*is_resume=.*$', 'is_resume=False'),
        (r'^[ \t]*resume_step=.*$', 'resume_step=0'),
        (r'^[ \t]*resume_epoch=.*$', 'resume_epoch=0'),
        (r'--vla\.max_steps 100\b', f'--vla.max_steps {max_steps}'),
        (r'--master_port=29824[^\n]*', '--master_port=29834 --log-dir ' +
         shlex.quote(str(directory / 'rank_logs')) + ' --tee 3 train.py \\'),
    ]
    for pattern, replacement in replacements:
        text, count = re.subn(pattern, lambda match: replacement, text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'Training launcher differs from expected format: {pattern!r}')
    return text


def is_benchmark_process(root, cwd, argv):
    if not argv or not (Path(argv[0]).name.startswith('python') or Path(argv[0]).name == 'torchrun'):
        return False
    if str(BENCHMARK) not in argv[1:]:
        return False
    for i, arg in enumerate(argv):
        if arg == '--root' and i + 1 < len(argv):
            return (cwd / argv[i + 1]).resolve() == root
        if arg.startswith('--root='):
            return (cwd / arg.split('=', 1)[1]).resolve() == root
    return cwd == root


def benchmark_processes(root):
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [arg.decode(errors='replace') for arg in (proc / 'cmdline').read_bytes().split(b'\0') if arg]
            if is_benchmark_process(root, (proc / 'cwd').resolve(), argv):
                found[int(proc.name)] = start_id(proc.name)
        except (OSError, IndexError):
            continue
    return found


def stop_benchmarks(root):
    for pid, identity in benchmark_processes(root).items():
        try:
            if start_id(pid) == identity:
                print(f'Stopping benchmark PID {pid}', flush=True)
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, FileNotFoundError):
            pass
    deadline = time.monotonic() + 45
    while benchmark_processes(root):
        if time.monotonic() >= deadline:
            raise RuntimeError('Benchmark processes remain; no formal training started. Inspect them manually.')
        time.sleep(0.5)


def check_checkpoints(directory, max_steps=2000, save_interval=500):
    checkpoints = []
    for step in [*range(save_interval, max_steps, save_interval), max_steps]:
        files = [p for p in (directory / 'runs').glob(f'*/checkpoints/step-{step:06d}-*.pt')
                 if p.is_file() and p.stat().st_size > 0]
        if len(files) != 1:
            raise RuntimeError(f'Expected one non-empty checkpoint at step {step}; found {len(files)}')
        checkpoints.append(str(files[0]))
    return checkpoints


def record_gpu_snapshot(directory, root):
    try:
        result = subprocess.run(['nvidia-smi', '-i', '3,4,5,6'], capture_output=True,
                                text=True, timeout=5, check=False)
        details = f'train.py PIDs: {sorted(training_processes(root))}\n{result.stdout}{result.stderr or ""}'
    except (OSError, subprocess.SubprocessError) as error:
        details = f'GPU monitor error: {error}\n'
    with (directory / 'gpu_monitor.log').open('a') as output:
        output.write(f'\n{datetime.now().isoformat()}\n{details}')


@contextmanager
def monitor_gpu(directory, root):
    stop = threading.Event()

    def sample():
        while not stop.wait(2):
            record_gpu_snapshot(directory, root)

    record_gpu_snapshot(directory, root)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=6)
        record_gpu_snapshot(directory, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--stop-benchmark', action='store_true', help='Stop only this tool repository\'s benchmark processes for the chosen training root')
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.save_interval < 1:
        parser.error('--max-steps and --save-interval must both be positive')
    if sys.platform != 'linux':
        parser.error('Run this script on the Linux training server')
    root = args.root.resolve()
    logs = root / 'log/robomme'
    logs.mkdir(parents=True, exist_ok=True)
    devices = resolve_devices([3, 4, 5, 6])
    original = (root / 'script/train/robomme/train.sh').read_text()
    import fcntl
    with (logs / 'train_run.lock').open('w') as training_lock, (logs / 'perf_compare.lock').open('a') as perf_lock:
        try:
            fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another formal training launcher is already running')
        directory = logs / datetime.now().strftime('formal_%Y%m%d_%H%M%S_%f')
        directory.mkdir(mode=0o700)
        launcher = directory / 'train.sh'
        launcher.write_text(make_training_launcher(original, directory, devices, args.max_steps, args.save_interval))
        launcher.chmod(0o600)
        subprocess.run(['bash', '-n', str(launcher)], check=True)
        if args.stop_benchmark:
            stop_benchmarks(root)
        try:
            fcntl.flock(perf_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Benchmark still running; no formal training started')
        if training_processes(root):
            raise RuntimeError('Existing train.py processes found; no new formal training started')
        (logs / 'latest_train_dir.txt').write_text(str(directory) + '\n')
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        env.setdefault('HF_HOME', str(root.parent / 'hf_cache'))
        env.setdefault('HF_ENDPOINT', 'https://huggingface.co')
        env['PYTHONUNBUFFERED'] = '1'
        print(f'FORMAL_TRAINING_START: physical GPUs=[3, 4, 5, 6]; UUIDs={devices}; batch/GPU=32; global batch=128; steps={args.max_steps}; save every={args.save_interval}; directory={directory}', flush=True)

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        with monitor_gpu(directory, root):
            child = subprocess.Popen(['bash', str(launcher)], cwd=root, env=env, start_new_session=True)
            try:
                code = child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        print(f'TRAIN_EXIT_CODE={code}', flush=True)
        if code:
            raise SystemExit(code if code > 0 else 128 - code)
        checkpoints = check_checkpoints(directory, args.max_steps, args.save_interval)
        (directory / 'completed.json').write_text(json.dumps({'checkpoint_files': checkpoints}, indent=2) + '\n')
        print(f'FORMAL_TRAINING_COMPLETE: {len(checkpoints)} non-empty checkpoint files found; directory={directory}', flush=True)


if __name__ == '__main__':
    main()
