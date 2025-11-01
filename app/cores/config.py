import os
import json
import tomllib
import threading
from pathlib import Path
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)

from typing import Dict, List, Optional, Literal, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()
def get_project_root() -> Path:
    """获得项目根路径"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT



class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),  # 默认 .env，可通过 ENV_FILE 切换
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    
    MODEL_BASE_URL: str | None = None
    MODEL_API_KEY: str | None = None
    MODEL_API_TYPE: str | None = None
    MODEL_NAME: str | None = None
    MODEL_TEMPERATURE: float | None = None
    MODEL_MAX_TOKENS: int | None = None
    MODEL_TOP_P: float | None = None
    MODEL_TOP_K: int | None = None


    
    # DATABASE配置
    DB_HOST:str | None = None
    DB_PORT:str | None = None
    DB_USER:str | None = None
    DB_PASSWORD: SecretStr | None = None
    DB_NAME:str | None = None
    UPLOAD_DIR:str | None = None
    
    
    # 分割模型配置
    # YOLO模型分割相关参数
    YOLO_SEGMENTATION_CONF:float | None = None
    YOLO_SEGMENTATION_IOU:float | None = None
    YOLO_IOU:float | None = None
    YOLO_MASK_THRESHOLD:float | None = None
    YOLO_USE_CUDA:bool = False
    YOLO_MAX_DET:int | None = None
    YOLO_AUGMENT:bool | None = None
    YOLO_MASK_FORMAT:str | None = None
    YOLO_SEGMENTATION_MODEL_NAME:str | None = None
    YOLO_SEGMENTATION_OUTPUT_DIR:str | None = None
    
    
    # 知识库文档
    #Embeding模型
    EMBEDING_MODEL:str | None = None
    EMBEDING_MODEL_BASE_URL:str | None = None
    CHUNK_SIZE: int = 750
    OVERLAP_SIZE: int = 150
    VECTOR_SEARCH_TOP_K: int = 3
    SCORE_THRESHOLD: float = 2.0
    ZH_TITLE_ENHANCE: bool = False
    # PDF_OCR_THRESHOLD: Tuple[float, float] = (0.6, 0.6)


                
    
    

    # def __new__(cls):
    #     if cls._instance is None:
    #         with cls._lock:
    #             if cls._instance is None:
    #                 cls._instance = super().__new__(cls)
    #     return cls._instance

    # def __init__(self):
    #     if not self._initialized:
    #         with self._lock:
    #             if not self._initialized:
    #                 self._config = None
    #                 self._load_initial_config()
    #                 # self._get_initial_embed_model_config()
    #                 self._initialized = True

    # def _load_initial_config(self):
    #     raw_config = self._load_config()

        # === embedding ===
        # base_embeding = raw_config.get("embeding", {})
        # embeding_overrides = {
        #     k: v for k, v in base_embeding.items() if isinstance(v, dict)
        # }

        # embed_default = {
        #     "model": base_embeding.get("model"),
        #     "base_url": base_embeding.get("base_url"),
        #     "api_key": base_embeding.get("api_key"),
        #     "max_tokens": base_embeding.get("max_tokens", 4096),
        #     "max_input_tokens": base_embeding.get("max_input_tokens"),
        #     "temperature": base_embeding.get("temperature", 1.0),
        #     "api_type": base_embeding.get("api_type", ""),
        #     "api_version": base_embeding.get("api_version", ""),
        # }

        # embed_dict = {
        #     "default": embed_default,
        #     **{name: {**embed_default, **cfg} for name, cfg in embeding_overrides.items()},
        # }

        # === llm ===
        # base_llm = raw_config.get("llm", {})
        # llm_overrides = {k: v for k, v in base_llm.items() if isinstance(v, dict)}

        # llm_default = {
        #     "model": base_llm.get("model"),
        #     "base_url": base_llm.get("base_url"),
        #     "api_key": base_llm.get("api_key"),
        #     "max_tokens": base_llm.get("max_tokens", 4096),
        #     "max_input_tokens": base_llm.get("max_input_tokens"),
        #     "temperature": base_llm.get("temperature", 1.0),
        #     "api_type": base_llm.get("api_type", ""),
        #     "api_version": base_llm.get("api_version", ""),
        # }

        # llm_dict = {
        #     "default": llm_default,
        #     **{name: {**llm_default, **cfg} for name, cfg in llm_overrides.items()},
        # }

        # # === 最终构建 AppConfig ===
        # self._config = AppConfig(
        #     # embeding=embed_dict,
        #     llm=llm_dict
        # )




    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root  / "config" / "config.toml"
        if config_path.exists():
            return config_path
        bak_path = root /"config" / "config.bak.toml"
        if bak_path.exists():
            return bak_path
        raise FileNotFoundError("config文件夹下缺少配置文件")

    @property
    def workspace_root(self) -> Path:
        """获取workspace根目录"""
        return WORKSPACE_ROOT
    
    @property
    def kb_path(self) -> Path:
        """获取知识库路径"""
        return WORKSPACE_ROOT / "app/kb_base"

    @property
    def root_path(self) -> Path:
        """获取应用根目录"""
        return PROJECT_ROOT
    
    @property
    def model_path(self) -> Path:
        """获取模型存储路径"""
        model_dir = self.workspace_root / "weight"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    @property
    def file_post_process_path(self) -> Path:
        """获取模型存储路径"""
        model_dir = self.workspace_root / "temp"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    @property
    def doc_pre_path(self) -> Path:
        """获取模型存储路径"""
        model_dir = self.workspace_root / "temp" / "doc_temp"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    


config = Config()
