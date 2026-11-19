#!/usr/bin/env python3
"""Run isolated MemoryVLA throughput checks; Linux server, standard library only."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime


def make_launcher(original, directory, cards, batch):
    text = original

    def required(pattern, replacement):
        nonlocal text
        text, count = re.subn(pattern, lambda m: replacement, text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'Launcher differs from expected format: {pattern!r} ({count} matches)')

    required(r'^[ \t]*n_gpu=.*$', f'n_gpu={cards}')
    required(r'^[ \t]*bs=.*$', f'bs={batch}')
    required(r'^[ \t]*shuffle_buffer_size=.*$', 'shuffle_buffer_size=1024')
    required(r'^[ \t]*run_root_dir=.*$', 'run_root_dir=' + shlex.quote(str(directory / 'runs')))
    required(r'^[ \t]*run_id=.*$', f'run_id=perf{cards}')
    required(r'^[ \t]*CUDA_VISIBLE_DEVICES=[^ \t\n]+', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, range(cards))))
    required(r'--nproc_per_node(?:=|[ \t]+)\d+', f'--nproc_per_node={cards}')
    required(r'--master_port(?:=|[ \t]+)\d+', f'--master_port={29820 + cards}')
    required(r'(?<![\w/])train\.py(?=\s)', shlex.quote(str(Path(__file__).resolve())) + ' --worker')
    required(r'--vla\.max_steps[ \t]+\d+', '--vla.max_steps 100')
    # Discard W&B/old tracker and group settings in this generated copy only.
    text = re.sub(r'^[ \t]*--(?:wandb_project|wandb_entity|trackers|group_size)(?:[ \t]+|=)[^\n]*\n?', '', text, flags=re.MULTILINE)
    required(r'^[ \t]*--hf_token(?:[ \t]+|=)[^\n]*$', '  --trackers \'["jsonl"]\' \\\n  --group_size 16 \\\n  --hf_token "${hf_token}" \\')
    return text


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    measured = {r['VLA Train/Step']: r for r in rows if 21 <= r['VLA Train/Step'] <= 100}
    if set(measured) != set(range(21, 101)):
        raise ValueError('Missing measured steps; expected all steps 21 through 100')
    times = []
    for row in measured.values():
        seconds, loss = row['VLA Train/Step Time'], row['VLA Train/Loss']
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(loss):
            raise ValueError('Invalid step time or non-finite loss')
        times.append(seconds)
    return dict(measured_steps=len(times), mean_step_seconds=statistics.mean(times),
                median_step_seconds=statistics.median(times), samples_per_second=64 / statistics.mean(times))


def is_training_process(root, cwd, argv):
    if cwd != root or not argv:
        return False
    executable = Path(argv[0]).name
    if not (executable.startswith('python') or executable == 'torchrun'):
        return False
    return any(arg in ('train.py', str(root / 'train.py')) for arg in argv[1:])


def start_id(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def training_processes(root):
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [v.decode(errors='replace') for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
            if is_training_process(root, (proc / 'cwd').resolve(), argv):
                found[int(proc.name)] = (start_id(proc.name), argv)
        except (OSError, IndexError):
            continue
    return found


def stop_training(root):
    targets = training_processes(root)
    for pid, (identity, argv) in targets.items():
        try:
            if start_id(pid) == identity:
                print(f'Stopping existing training PID {pid}: {shlex.join(argv[:3])}', flush=True)
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    deadline = time.monotonic() + 45
    while training_processes(root):
        if time.monotonic() >= deadline:
            raise RuntimeError('Old training processes remain. No benchmark was started; inspect them manually.')
        time.sleep(0.5)


def run_worker():
    # Only benchmark subprocesses replace saving. Formal training files stay untouched.
    sys.path.insert(0, str(Path.cwd()))
    from training.strategies.fsdp import FSDPStrategy
    FSDPStrategy.save_checkpoint = lambda *args, **kwargs: None
    sys.argv = ['train.py'] + sys.argv[2:]
    import runpy
    runpy.run_path('train.py', run_name='__main__')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--stop-existing', action='store_true', help='SIGTERM train.py jobs whose cwd is exactly this repository')
    parser.add_argument('--gpus', type=int, choices=(2, 4), help='Run only this GPU count; default: compare 2 then 4 GPUs')
    args = parser.parse_args(argv)
    configurations = (args.gpus,) if args.gpus else (2, 4)
    if sys.platform != 'linux':
        parser.error('Run this script on the Linux training server')
    root = args.root.resolve()
    original = (root / 'script/train/robomme/train.sh').read_text()
    logs = root / 'log/robomme'
    logs.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (logs / 'perf_compare.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another throughput comparison is already running')
        directory = logs / datetime.now().strftime('perf_compare_%Y%m%d_%H%M%S_%f')
        directory.mkdir()
        for cards in configurations:
            launcher = directory / f'perf{cards}.sh'
            launcher.write_text(make_launcher(original, directory, cards, 64 // cards))
            subprocess.run(['bash', '-n', str(launcher)], check=True)
        (logs / 'latest_perf_dir.txt').write_text(str(directory) + '\n')
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        env.setdefault('HF_HOME', str(root.parent / 'hf_cache'))
        env.setdefault('HF_ENDPOINT', 'https://huggingface.co')
        env.update(PYTHONUNBUFFERED='1', NCCL_DEBUG='INFO')
        if args.stop_existing:
            stop_training(root)
        elif training_processes(root):
            raise RuntimeError('Existing training found; use --stop-existing to discard that run')
        results = []
        for cards in configurations:
            print(f'Starting {cards} GPUs, batch/GPU={64 // cards}; log={directory / ("perf"+str(cards)+".log")}', flush=True)
            with (directory / f'perf{cards}.log').open('w') as output:
                child = subprocess.Popen(['bash', str(directory / f'perf{cards}.sh')], cwd=root,
                                         env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=3600)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            if code:
                results.append(dict(gpus=cards, batch_per_gpu=64 // cards, status='failed', exit_code=code))
                (directory / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
                print(f'{cards}-GPU run failed (exit {code}); see its log.', flush=True)
                continue
            run_dirs = list((directory / 'runs').glob(f'perf{cards}*'))
            if len(run_dirs) != 1:
                raise ValueError(f'Expected one output directory for {cards} GPUs')
            run_dir = run_dirs[0]
            cfg = json.loads((run_dir / 'config.json').read_text())
            vla = cfg['vla']
            if (vla['expected_world_size'], vla['per_device_batch_size'], vla['global_batch_size'], cfg['group_size']) != (cards, 64 // cards, 64, 16):
                raise ValueError('Actual training configuration does not match the comparison')
            result = dict(gpus=cards, batch_per_gpu=64 // cards, status='passed', **summarize(run_dir / f'{run_dir.name}.jsonl'))
            results.append(result)
            (directory / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        passed = [r for r in results if r['status'] == 'passed']
        if len(passed) == len(configurations):
            if len(configurations) == 1:
                print(f'BENCHMARK_COMPLETE: configuration={args.gpus} GPUs; results={directory / "summary.json"}', flush=True)
            else:
                winner = min(passed, key=lambda r: r['mean_step_seconds'])
                print(f'COMPARISON_COMPLETE: faster configuration={winner["gpus"]} GPUs; results={directory / "summary.json"}', flush=True)
        else:
            label = 'BENCHMARK_INCOMPLETE' if len(configurations) == 1 else 'COMPARISON_INCOMPLETE'
            print(f'{label}: successful configurations={len(passed)}; results={directory / "summary.json"}', flush=True)
        print('No formal training was started. Compare these short-run results before choosing a training configuration.', flush=True)
        if len(passed) != len(configurations):
            raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        run_worker()
    else:
        main()
