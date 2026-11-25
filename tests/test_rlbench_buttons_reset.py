import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rlbench_buttons.demo_reset import reset_demo


class DemoResetTest(unittest.TestCase):
    def test_reset_uses_requested_order_even_when_demo_metadata_says_zero(self):
        demo = SimpleNamespace(
            restore_state=Mock(),
            _observations=[SimpleNamespace(misc={"variation_index": 0})],
        )
        task = Mock()
        task._task.order = (1, 0, 2)
        task.reset.return_value = (["green, blue, pink"], object())

        self.assertEqual(reset_demo(task, demo, 2, (1, 0, 2)), task.reset.return_value)
        demo.restore_state.assert_called_once_with()
        task.set_variation.assert_called_once_with(2)
        task.reset.assert_called_once_with(demo)
        task.reset_to_demo.assert_not_called()

    def test_unexpected_order_after_reset_is_rejected(self):
        task = Mock()
        task._task.order = (0, 1, 2)
        task.reset.return_value = (["blue, green, pink"], object())
        with self.assertRaisesRegex(RuntimeError, "Expected button order"):
            reset_demo(task, Mock(), 2, (1, 0, 2))
