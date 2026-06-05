from __future__ import annotations

from .base import CacheManager, LayerBatchStates

__all__ = [
    "CacheManager",
    "LayerBatchStates",
    "StandardCacheManager",
    "StreamingLLMCacheManager",
    "SnapKVCacheManager",
    "MinferenceStandardCacheManager",
    "MinferenceSnapKVCacheManager",
    "QuestCacheManager",
    "OmniKVCacheManager",
    "DeltaKVCacheManager",
    "DeltaKVCacheTritonManager",
    "DeltaKVCacheTritonManagerV2",
    "DeltaKVCacheTritonManagerV3",
    "DeltaKVCacheTritonManagerV4",
    "DeltaKVDeltaQuantCacheManager",
    "DeltaKVStandaloneCacheManager",
    "DeltaKVSnapKVCacheManager",
]


def __getattr__(name: str):
    if name == "StandardCacheManager":
        from .standard import StandardCacheManager

        return StandardCacheManager
    if name == "StreamingLLMCacheManager":
        from .streamingllm import StreamingLLMCacheManager

        return StreamingLLMCacheManager
    if name == "SnapKVCacheManager":
        from .snapkv import SnapKVCacheManager

        return SnapKVCacheManager
    if name in {"MinferenceStandardCacheManager", "MinferenceSnapKVCacheManager"}:
        from . import minference as _minference

        return getattr(_minference, name)
    if name == "QuestCacheManager":
        from .quest import QuestCacheManager

        return QuestCacheManager
    if name == "OmniKVCacheManager":
        from .omnikv import OmniKVCacheManager

        return OmniKVCacheManager
    if name in {
        "DeltaKVCacheManager",
        "DeltaKVCacheTritonManager",
        "DeltaKVCacheTritonManagerV2",
        "DeltaKVCacheTritonManagerV3",
        "DeltaKVCacheTritonManagerV4",
    }:
        from . import deltakv as _deltakv

        return getattr(_deltakv, name)
    if name == "DeltaKVDeltaQuantCacheManager":
        from .deltakv_delta_quant import DeltaKVDeltaQuantCacheManager

        return DeltaKVDeltaQuantCacheManager
    if name == "DeltaKVStandaloneCacheManager":
        from .deltakv_standalone import DeltaKVStandaloneCacheManager

        return DeltaKVStandaloneCacheManager
    if name == "DeltaKVSnapKVCacheManager":
        from .deltakv_snapkv import DeltaKVSnapKVCacheManager

        return DeltaKVSnapKVCacheManager

    raise AttributeError(name)
