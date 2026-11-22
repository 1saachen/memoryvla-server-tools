#!/usr/bin/env python3
"""Compare the 2000 and 5000 step checkpoints on RoboMME ButtonUnmask val."""
import argparse
from datetime import datetime
from functools import wraps
import json
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


STEPS = (2000, 5000)
TASK = 'ButtonUnmask'
PORT = 8052
SCRIPT = Path(__file__).resolve()


def evaluate(root, step, output):
    evaluator = root / 'evaluation/robomme/eval.py'
    sys.path.insert(0, str(evaluator.parent))
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    import env_runner

    @wraps(BenchmarkEnvBuilder)
    def build(*args, **kwargs):
        kwargs['dataset'] = 'val'
        builder = BenchmarkEnvBuilder(*args, **kwargs)
        indices = {episode for task, episode in builder.metadata_index if task == TASK}
        if indices != set(range(50)):
            raise RuntimeError(f'Expected validation episodes 0–49 for {TASK}; found {sorted(indices)}')
        print(f'ROBOMME_VAL: task={TASK} episodes={len(indices)} step={step}', flush=True)
        return builder

    env_runner.BenchmarkEnvBuilder = build
    sys.argv = [str(evaluator), '--args.model_seed', '7', '--args.model_ckpt_id', str(step),
                '--args.policy_name', 'memvla', '--args.port', str(PORT), '--args.only_tasks', TASK,
                '--args.save_dir', str(output)]
    runpy.run_path(str(evaluator), run_name='__main__')


def serve(root, checkpoint):
    sys.path.insert(0, str(root))
    deploy = root / 'deploy.py'
    sys.argv = [str(deploy), '--saved_model_path', str(checkpoint), '--unnorm_key', 'robomme',
                '--adaptive_ensemble_alpha', '0.1', '--cfg_scale', '1.5', '--port', str(PORT),
                '--action_chunking', '--action_chunking_window', '8', '--seed', '7']
    app = runpy.run_path(str(deploy))['app']
    app.run(host='127.0.0.1', port=PORT, debug=False)


def compare(run):
    root = Path.cwd()
    if not (root / 'deploy.py').is_file():
        raise RuntimeError('Run from the MemoryVLA repository root')
    checkpoints = []
    for step in STEPS:
        matches = list((run / 'checkpoints').glob(f'step-{step:06d}-*.pt'))
        if len(matches) != 1 or not matches[0].is_file() or not matches[0].stat().st_size:
            raise RuntimeError(f'Expected one non-empty checkpoint for step {step} in {run}')
        checkpoints.append(matches[0])
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', PORT))

    base = root.parent
    model_python = base / 'envs/memvla/bin/python'
    sim_python = base / 'envs/robomme/bin/python'
    vulkan = base / 'nvidia_vulkan/570.148.08/env.sh'
    for path in (model_python, sim_python, vulkan):
        if not path.is_file():
            raise RuntimeError(f'File not found: {path}')
    output = run / datetime.now().strftime('val_compare_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    print(f'VAL_COMPARE_DIR={output}', flush=True)
    model_env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-8363dc9c-84bf-dd42-cd2e-05df1e86bf98')
    model_env.setdefault('HF_HOME', str(base / 'hf_cache'))
    sim_env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-57d88094-a096-fc77-104a-2e6aa079e1b8')
    summary = {}
    for step, checkpoint in zip(STEPS, checkpoints):
        step_dir = output / f'step{step}'
        step_dir.mkdir()
        server_log = step_dir / 'server.log'
        client_log = step_dir / 'client.log'
        print(f'STEP={step} MODEL={checkpoint} CLIENT_LOG={client_log}', flush=True)
        with server_log.open('w') as log, TemporaryDirectory(dir=step_dir) as image_dir:
            # The server writes debug images into its working directory.
            with subprocess.Popen([model_python, '-u', SCRIPT, '--serve', root, checkpoint],
                                  cwd=step_dir, env=dict(model_env, TMPDIR=image_dir),
                                  stdout=log, stderr=subprocess.STDOUT) as server:
                try:
                    deadline = time.monotonic() + 1200
                    while f'Running on http://127.0.0.1:{PORT}' not in server_log.read_text(errors='replace'):
                        if server.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError(f'Model service did not start: see {server_log}')
                        time.sleep(2)
                    print(f'STEP={step} SERVER_READY', flush=True)
                    with client_log.open('w') as client_output:
                        subprocess.run(['bash', '-c', 'source "$1" && shift && exec "$@"',
                                        'robomme', vulkan, sim_python, '-u', SCRIPT,
                                        '--evaluate', root, str(step), step_dir], env=sim_env,
                                       stdout=client_output, stderr=subprocess.STDOUT, check=True)
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            server.kill()
        results = step_dir / f'memvla/ckpt{step}/seed7'
        episodes = json.loads((results / 'progress.json').read_text())[TASK]
        if set(episodes) != {str(i) for i in range(50)} or not (results / 'log.json').is_file():
            raise RuntimeError(f'Incomplete evaluation: see {client_log}')
        error_episodes = sorted({int(i) for i in re.findall(
            r'Error evaluating episode (\d+) for task ButtonUnmask:', client_log.read_text(errors='replace'))})
        successes = sum(value is True for value in episodes.values())
        summary[step] = {'successes': successes, 'episodes': len(episodes),
                         'success_rate': successes / len(episodes), 'episodes_with_logged_error': error_episodes}
        print(f'STEP={step} RESULT={successes}/{len(episodes)}', flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'VAL_COMPARE_COMPLETE={output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Training run directory containing checkpoints/')
    parser.add_argument('--serve', nargs=2, help=argparse.SUPPRESS)
    parser.add_argument('--evaluate', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(*map(Path, args.serve))
    elif args.evaluate:
        root, step, output = args.evaluate
        evaluate(Path(root), int(step), Path(output))
    elif args.run:
        def interrupt(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupt)
        compare(args.run.resolve())
    else:
        parser.error('--run is required')
