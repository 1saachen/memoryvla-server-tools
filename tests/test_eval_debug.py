import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'eval_debug.py'


class EvalDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('eval_debug', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_empty_observation_reports_original_ik_error_and_stops_retries(self):
        class Runner:
            env_id = 'ButtonUnmask'
            episode_id = 0
            closed = False

            def get_init_obs(self):
                return {'images': [1]}

            def step(self, action):
                self.obs = {}
                self.info = {'status': 'error', 'error_message': 'IK failed: distance 0.002'}
                return self.obs['front_rgb_list']

            def close_env(self):
                self.closed = True

        self.mod.install_diagnostics(SimpleNamespace(EnvRunner=Runner))
        runner = Runner()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as error:
                runner.step([0.1, 0.2, 0.3, 3.14, 0, 0, 1])
        self.assertEqual(error.exception.code, 1)
        self.assertTrue(runner.closed)
        for expected in ['IK failed: distance 0.002', 'front_rgb_list', 'ButtonUnmask',
                         'episode=0', '0.1', 'Traceback', 'ROBOMME_DIAGNOSTIC_STOP']:
            self.assertIn(expected, output.getvalue())

    def test_normal_observation_and_step_are_unchanged(self):
        class Runner:
            def get_init_obs(self):
                return {'images': [1]}

            def step(self, action):
                return ((1, 2, 3), False, 'running')

        self.mod.install_diagnostics(SimpleNamespace(EnvRunner=Runner))
        self.assertEqual(Runner().get_init_obs(), {'images': [1]})
        self.assertEqual(Runner().step([1]), ((1, 2, 3), False, 'running'))


if __name__ == '__main__':
    unittest.main()
