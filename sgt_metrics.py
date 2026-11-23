"""Episode and paired summaries for ShellGameTouch memory ablation."""

from collections import Counter


def summarize_results(results):
    """Keep every evaluated episode in each rate denominator."""
    num_eval = len(results)
    successes = sum(bool(result["success"]) for result in results)
    first_touch_successes = sum(bool(result["first_touch_correct"]) for result in results)
    first_reach_successes = sum(bool(result["first_reach_correct"]) for result in results)
    return {
        "num_eval": num_eval,
        "successes": successes,
        "success_rate": successes / num_eval if num_eval else None,
        "first_touch_successes": first_touch_successes,
        "first_touch_success_rate": first_touch_successes / num_eval if num_eval else None,
        "first_reach_successes": first_reach_successes,
        "first_reach_success_rate": first_reach_successes / num_eval if num_eval else None,
        "no_first_touch": sum(result["first_touch_step"] is None for result in results),
        "ambiguous_first_touch": sum(len(result["first_touched_cups"]) > 1 for result in results),
        "no_first_reach": sum(result["first_reach_step"] is None for result in results),
        "memory_clears": sum(result["memory_clear_step"] is not None for result in results),
    }


def summarize_pairs(keep_results, clear_results):
    """Compare outcomes for the same unique seeds in their evaluation order."""
    keep_seeds = [result["seed"] for result in keep_results]
    clear_seeds = [result["seed"] for result in clear_results]
    if keep_seeds != clear_seeds:
        raise ValueError("Keep and clear results must contain the same seeds in the same order")
    if not keep_seeds:
        raise ValueError("At least one paired seed is required")
    if len(set(keep_seeds)) != len(keep_seeds):
        raise ValueError("Paired seeds must be unique")

    pairs = list(zip(keep_results, clear_results))
    num_pairs = len(pairs)
    keep_memory_clears = sum(result["memory_clear_step"] is not None for result in keep_results)
    clear_memory_clears = sum(result["memory_clear_step"] is not None for result in clear_results)
    summary = {
        "num_pairs": num_pairs,
        "keep_memory_clears": keep_memory_clears,
        "clear_memory_clears": clear_memory_clears,
    }
    for metric, field in (
        ("official", "success"),
        ("first_touch", "first_touch_correct"),
        ("first_reach", "first_reach_correct"),
    ):
        outcomes = Counter((bool(keep[field]), bool(clear[field])) for keep, clear in pairs)
        keep_only = outcomes[True, False]
        clear_only = outcomes[False, True]
        summary[metric] = {
            "both_correct": outcomes[True, True],
            "keep_only": keep_only,
            "clear_only": clear_only,
            "neither": outcomes[False, False],
            "clear_minus_keep_percentage_points": 100 * (clear_only - keep_only) / num_pairs,
        }

    traces = {"exact_match": 0, "differing": 0, "missing": 0}
    touch_before_clear = {"keep": 0, "clear": 0}
    reach_before_clear = {"keep": 0, "clear": 0}
    for keep, clear in pairs:
        keep_trace = keep.get("pre_clear_trace_sha256")
        clear_trace = clear.get("pre_clear_trace_sha256")
        if keep_trace is None or clear_trace is None:
            traces["missing"] += 1
        elif keep_trace == clear_trace:
            traces["exact_match"] += 1
        else:
            traces["differing"] += 1

        clear_step = clear["memory_clear_step"]
        if clear_step is None:
            continue
        for arm, result in (("keep", keep), ("clear", clear)):
            touch_step = result["first_touch_step"]
            reach_step = result["first_reach_step"]
            touch_before_clear[arm] += touch_step is not None and touch_step <= clear_step
            reach_before_clear[arm] += reach_step is not None and reach_step <= clear_step

    summary["pre_clear_trace"] = traces
    summary["first_touch_at_or_before_clear"] = touch_before_clear
    summary["first_reach_at_or_before_clear"] = reach_before_clear
    return summary
