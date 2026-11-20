#!/usr/bin/env python3
"""Run RoboMME evaluation until its first error, preserving simulator diagnostics."""
import argparse
import functools
import importlib
from pathlib import Path
import runpy
import sys
import traceback


def install_diagnostics(module):
    def wrap(method_name):
        original = getattr(module.EnvRunner, method_name)

        @functools.wraps(original)
        def checked(self, *args, **kwargs):
            previous_obs = getattr(self, 'obs', None)
            try:
                return original(self, *args, **kwargs)
            except Exception:
                print(f'ROBOMME_DIAGNOSTIC_ERROR: method={method_name} '
                      f'task={getattr(self, "env_id", None)} '
                      f'episode={getattr(self, "episode_id", None)}', flush=True)
                if method_name == 'step':
                    action = args[0] if args else kwargs.get('action')
                    print(f'Action: {action}', flush=True)
                    if isinstance(previous_obs, dict):
                        print(f'Previous eef_state_list: {previous_obs.get("eef_state_list")}', flush=True)
                obs = getattr(self, 'obs', None)
                print(f'Observation keys/type: {list(obs) if isinstance(obs, dict) else type(obs).__name__}', flush=True)
                print(f'Simulator info: {getattr(self, "info", None)!r}', flush=True)
                traceback.print_exc()
                try:
                    self.close_env()
                except Exception:
                    traceback.print_exc()
                print('ROBOMME_DIAGNOSTIC_STOP: first error; no retry or failure score recorded for this attempt.', flush=True)
                # SystemExit bypasses the evaluator's broad except Exception/retry loop.
                raise SystemExit(1) from None

        return checked

    for method_name in ('get_init_obs', 'step'):
        setattr(module.EnvRunner, method_name, wrap(method_name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, required=True,
                        help='New diagnostic output directory; existing paths are rejected.')
    args = parser.parse_args()
    root = args.root.resolve()
    evaluator = root / 'evaluation/robomme/eval.py'
    if not evaluator.is_file():
        parser.error(f'Evaluator not found: {evaluator}')
    args.save_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(evaluator.parent))
    install_diagnostics(importlib.import_module('env_runner'))
    sys.argv = [str(evaluator), '--args.model_seed', '7', '--args.model_ckpt_id', '2000',
                '--args.policy_name', 'memvla', '--args.port', '8051',
                '--args.only_tasks', 'ButtonUnmask', '--args.save_dir', str(args.save_dir)]
    print(f'ROBOMME_DIAGNOSTIC_START: server=localhost:8051 task=ButtonUnmask '
          f'output={args.save_dir}', flush=True)
    runpy.run_path(str(evaluator), run_name='__main__')


if __name__ == '__main__':
    main()
