"""Restore live demonstrations using the requested RLBench variation."""


def reset_demo(task_env, demo, variation, expected_order):
    # On the first episode, Scene.init_task can overwrite the variation
    # recorded in the observations. Restore the RNG and placement attempt
    # count as reset_to_demo does, but use the caller's variation.
    demo.restore_state()
    task_env.set_variation(variation)
    descriptions, observation = task_env.reset(demo)
    actual_order = tuple(task_env._task.order)
    if actual_order != expected_order:
        raise RuntimeError(
            f"Expected button order {expected_order}, got {actual_order} after reset"
        )
    return descriptions, observation
