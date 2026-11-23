"""Launch upstream MemoryVLA with Hugging Face's cached authentication."""

import sys
from pathlib import Path


# train.py writes its CLI configuration to disk. Keep credentials in the HF
# token cache or environment and let huggingface_hub resolve them implicitly.
sys.path.insert(0, str(Path.cwd()))
import train as upstream


load_vla = upstream.load_vla


def load_with_cached_auth(*args, **kwargs):
    kwargs["hf_token"] = None
    return load_vla(*args, **kwargs)


upstream.load_vla = load_with_cached_auth
upstream.train()
