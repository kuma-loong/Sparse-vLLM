from __future__ import annotations

from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext

from .standard import StandardCacheManager


class OmniKVCacheManager(StandardCacheManager):
    def __init__(self, config: Config, parallel_context: ParallelContext):
        super().__init__(config, parallel_context)

