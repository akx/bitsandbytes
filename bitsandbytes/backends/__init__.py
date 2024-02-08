from typing import Dict

import torch

from bitsandbytes.backends.base import Backend
from bitsandbytes.cextension import COMPILED_WITH_CUDA

backends: Dict[str, Backend] = {}

def register_backend(backend_name: str, backend_instance: Backend):
    backends[backend_name.lower()] = backend_instance

if COMPILED_WITH_CUDA:
    from .cuda import CUDABackend
    register_backend("cuda", CUDABackend())
