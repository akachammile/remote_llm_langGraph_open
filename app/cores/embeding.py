from typing import Dict, Optional
from app.cores.config import config

# class Embeding:
#     """设置LLM属性以及方法"""
#     _instances: Dict[str, "Embeding"] = {}

#     # 单例模式
#     def __new__(cls, config_name: str = "default", embeding_config: Optional[EmbedingSettings] = None):
#         if config_name not in cls._instances:
#             instance = super().__new__(cls)
#             instance.__init__(config_name, embeding_config)
#             cls._instances[config_name] = instance
#         return cls._instances[config_name]
    
#     # Embeding模型属性初始化
#     def __init__(self, config_name: str = "default", embeding_config: Optional[EmbedingSettings] = None):
#         if not hasattr(self, "embeding"):  # 没有llm实例则初始化一个即可
#             embeding_config = embeding_config or config.embeding
#             embeding_config = embeding_config.get(config_name, embeding_config["default"])
#             self.model = embeding_config.model
#             self.max_tokens = embeding_config.max_tokens
#             self.temperature = embeding_config.temperature
#             self.api_type = embeding_config.api_type
#             self.api_key = embeding_config.api_key
#             self.api_version = embeding_config.api_version
#             self.base_url = embeding_config.base_url
#             self.top_k = embeding_config.top_k
#             self.top_p = embeding_config.top_p