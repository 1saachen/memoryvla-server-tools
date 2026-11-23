"""Run the existing MemoryVLA deployment with local SGT ablation controls."""

import random
import runpy
import sys
from pathlib import Path


def configure_sgt(app, vla, seed_model):
    from flask import jsonify, request

    reset_cognitive = vla.cog_mem_bank.reset
    reset_perceptual = vla.per_mem_bank.reset
    process_frame = next(
        rule.endpoint for rule in app.url_map.iter_rules()
        if rule.rule == "/process_frame" and "POST" in rule.methods
    )
    inference = app.view_functions[process_frame]

    def process_sgt_frame():
        form = request.form
        seed_text = form.get("episode_seed")
        clear = form.get("clear_memory", "False")
        if seed_text is None and clear == "False":
            return inference()

        first_frame = form.get("episode_first_frame")
        if "image" not in request.files or "text" not in form:
            return jsonify(error="SGT controls require an image and instruction"), 400
        if clear not in {"True", "False"} or first_frame not in {"True", "False"}:
            return jsonify(error="Expected True or False for the frame and memory flags"), 400
        if clear == "True" and first_frame == "True":
            return jsonify(error="The ablation clear applies only within an episode"), 400

        confirmation = {}
        if seed_text is not None:
            if first_frame != "True":
                return jsonify(error="episode_seed applies only to the first frame"), 400
            try:
                seed = int(seed_text)
                if not 0 <= seed < 2**32:
                    raise ValueError
            except ValueError:
                return jsonify(error="episode_seed must be an integer between 0 and 4294967295"), 400
            seed_model(seed)
            confirmation["episode_seed"] = seed

        if clear == "True":
            # Keep the absolute model timestep and diffusion RNG unchanged.
            reset_cognitive()
            reset_perceptual()
            confirmation["memory_cleared"] = True

        response = app.make_response(inference())
        if not 200 <= response.status_code < 300:
            return response
        result = response.get_json()
        if not isinstance(result, dict) or "response" not in result:
            raise RuntimeError("Unexpected MemoryVLA inference response")
        return jsonify({**result, **confirmation})

    app.view_functions[process_frame] = process_sgt_frame

    @app.get("/sgt-ready")
    def ready():
        return "SGT_READY"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sgt_server.py DEPLOY_PY [deploy.py options]")
    entry = Path(sys.argv.pop(1)).resolve(strict=True)
    sys.path.insert(0, str(entry.parent))
    deploy = runpy.run_path(str(entry))

    import numpy as np
    import torch

    def seed_model(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    app = deploy["app"]
    configure_sgt(app, deploy["inferencer"].vla, seed_model)
    app.run(host="127.0.0.1", port=deploy["args"].port, threaded=False)


if __name__ == "__main__":
    main()
