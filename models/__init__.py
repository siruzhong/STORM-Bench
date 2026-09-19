try:
    from .vstream_qwen2vl_model import (
        FlashVStreamQwen2VLConfig,
        FlashVStreamQwen2VLModel,
        get_real_grid_thw,
        get_spatial_real_grid_thw,
    )
    from .vstream_qwen2vl_processor import FlashVStreamQwen2VLProcessor
    from .flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG
except ImportError:
    FlashVStreamQwen2VLModel = None
    FlashVStreamQwen2VLConfig = None
    FlashVStreamQwen2VLProcessor = None
    DEFAULT_FLASH_MEMORY_CONFIG = None
    get_real_grid_thw = None
    get_spatial_real_grid_thw = None

try:
    from .prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration
except ImportError:
    PReMQwen2VLForConditionalGeneration = None

try:
    from .prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
except ImportError:
    PReMQwen2_5_VLForConditionalGeneration = None

try:
    from .prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration
except ImportError:
    PReMQwen3VLForConditionalGeneration = None
