import os
import json
import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from dotenv import load_dotenv

def get_project_root() -> Path:
    """获得项目根路径"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT


class LLMSettings(BaseSettings):
    """大模型相关配置

    Args:
        BaseSettings (_type_): 继承Pydantic
    """
    model: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="模型部署路径")
    api_key: str = Field(..., description="线上部署的api-key")
    max_tokens: int = Field(4096, description="单次请求的最大Token数量")
    temperature: float = Field(0.2, description="采样温度")
    api_type: str = Field(..., description="API来源，是云还是本地")
    api_version: float = Field(0.1, description="API版本，估计没啥用")
    top_k: int = Field(40, description="固定数量前 K 个 token")
    top_p: float = Field(0.9, description="核采样，固定概率")
    max_input_tokens: Optional[int] = Field(None, description="在所有请求中可使用的最大输入 token 数（设置为 None 表示不限）")



class EmbedingSettings(BaseSettings):
    """Embeding模型配置

    Args:
        BaseSettings (_type_): 继承Pydantic
    """
    model: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="模型部署路径")
    api_key: str = Field(..., description="线上部署的api-key")
    max_tokens: int = Field(4096, description="单次请求的最大Token数量")
    temperature: float = Field(0.2, description="采样温度")
    api_type: str = Field(..., description="API来源，是云还是本地")
    api_version: float = Field(0.1, description="API版本，估计没啥用")
    top_k: int = Field(40, description="固定数量前 K 个 token")
    top_p: float = Field(0.9, description="核采样，固定概率")
    max_input_tokens: Optional[int] = Field(None, description="在所有请求中可使用的最大输入 token 数（设置为 None 表示不限）")

    # model_name: str = Field(
    #     default="text-embedding-3-small",
    #     description="Embedding 模型名称，例如 OpenAI 的 text-embedding-3-small"
    # )
    # dimension: int = Field(
    #     default=1536,
    #     description="向量维度，应与所选模型输出一致"
    # )
    # normalize: bool = Field(
    #     default=True,
    #     description="是否对生成的向量进行归一化（常用于 cosine 相似度）"
    # )

    # # ---- 存储/数据库相关 ----
    # backend: Literal["faiss", "milvus", "pinecone", "weaviate", "chroma"] = Field(
    #     default="faiss",
    #     description="向量数据库类型，可选 faiss/milvus/pinecone/weaviate/chroma"
    # )
    # connection_params: Dict[str, str] = Field(
    #     default_factory=dict,
    #     description="数据库连接参数，例如 host/port/api_key 等"
    # )
    # persist_path: Optional[str] = Field(
    #     default=None,
    #     description="本地持久化路径（适用于 faiss/chroma 等本地数据库）"
    # )

    # # ---- 检索/索引相关 ----
    # metric_type: Literal["L2", "cosine", "ip"] = Field(
    #     default="cosine",
    #     description="相似度度量方式：L2=欧式距离，cosine=余弦相似度，ip=内积"
    # )
    # search_params: Dict[str, int] = Field(
    #     default_factory=dict,
    #     description="检索参数，例如 faiss 的 nprobe 或 milvus 的 ef"
    # )

    # # ---- 运行配置 ----
    # batch_size: int = Field(
    #     default=32,
    #     description="批量处理的大小，影响吞吐量和显存使用"
    # )
    # max_text_length: int = Field(
    #     default=512,
    #     description="单条文本的最大长度，超过需截断或分片"
    # )
    # use_gpu: bool = Field(
    #     default=False,
    #     description="是否启用 GPU 加速（取决于数据库/模型支持）"
    # )
    
# class ProxySettings(BaseModel):
#     server: str = Field(None, description="Proxy server address")
#     username: Optional[str] = Field(None, description="Proxy username")
#     password: Optional[str] = Field(None, description="Proxy password")


# class SearchSettings(BaseModel):
#     engine: str = Field(default="Google", description="Search engine the llm to use")
#     fallback_engines: List[str] = Field(
#         default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
#         description="Fallback search engines to try if the primary engine fails",
#     )
#     retry_delay: int = Field(
#         default=60,
#         description="Seconds to wait before retrying all engines again after they all fail",
#     )
#     max_retries: int = Field(
#         default=3,
#         description="Maximum number of times to retry all engines when all fail",
#     )
#     lang: str = Field(
#         default="en",
#         description="Language code for search results (e.g., en, zh, fr)",
#     )
#     country: str = Field(
#         default="us",
#         description="Country code for search results (e.g., us, cn, uk)",
#     )


# class RunflowSettings(BaseModel):
#     use_data_analysis_agent: bool = Field(
#         default=False, description="Enable data analysis agent in run flow"
#     )


# class BrowserSettings(BaseModel):
#     headless: bool = Field(False, description="Whether to run browser in headless mode")
#     disable_security: bool = Field(
#         True, description="Disable browser security features"
#     )
#     extra_chromium_args: List[str] = Field(
#         default_factory=list, description="Extra arguments to pass to the browser"
#     )
#     chrome_instance_path: Optional[str] = Field(
#         None, description="Path to a Chrome instance to use"
#     )
#     wss_url: Optional[str] = Field(
#         None, description="Connect to a browser instance via WebSocket"
#     )
#     cdp_url: Optional[str] = Field(
#         None, description="Connect to a browser instance via CDP"
#     )
#     proxy: Optional[ProxySettings] = Field(
#         None, description="Proxy settings for the browser"
#     )
#     max_content_length: int = Field(
#         2000, description="Maximum length for content retrieval operations"
#     )


# class SandboxSettings(BaseModel):
#     """Configuration for the execution sandbox"""

#     use_sandbox: bool = Field(False, description="Whether to use the sandbox")
#     image: str = Field("python:3.12-slim", description="Base image")
#     work_dir: str = Field("/workspace", description="Container working directory")
#     memory_limit: str = Field("512m", description="Memory limit")
#     cpu_limit: float = Field(1.0, description="CPU limit")
#     timeout: int = Field(300, description="Default command timeout (seconds)")
#     network_enabled: bool = Field(
#         False, description="Whether network access is allowed"
#     )


# class MCPServerConfig(BaseModel):
#     """Configuration for a single MCP server"""

#     type: str = Field(..., description="Server connection type (sse or stdio)")
#     url: Optional[str] = Field(None, description="Server URL for SSE connections")
#     command: Optional[str] = Field(None, description="Command for stdio connections")
#     args: List[str] = Field(
#         default_factory=list, description="Arguments for stdio command"
#     )


# class MCPSettings(BaseModel):
#     """Configuration for MCP (Model Context Protocol)"""

#     server_reference: str = Field(
#         "app.mcp.server", description="Module reference for the MCP server"
#     )
#     servers: Dict[str, MCPServerConfig] = Field(
#         default_factory=dict, description="MCP server configurations"
#     )

#     @classmethod
#     def load_server_config(cls) -> Dict[str, MCPServerConfig]:
#         """Load MCP server configuration from JSON file"""
#         config_path = PROJECT_ROOT / "config" / "mcp.json"

#         try:
#             config_file = config_path if config_path.exists() else None
#             if not config_file:
#                 return {}

#             with config_file.open() as f:
#                 data = json.load(f)
#                 servers = {}

#                 for server_id, server_config in data.get("mcpServers", {}).items():
#                     servers[server_id] = MCPServerConfig(
#                         type=server_config["type"],
#                         url=server_config.get("url"),
#                         command=server_config.get("command"),
#                         args=server_config.get("args", []),
#                     )
#                 return servers
#         except Exception as e:
#             raise ValueError(f"Failed to load MCP server config: {e}")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    embeding: Dict[str, EmbedingSettings]
    # sandbox: Optional[SandboxSettings] = Field(
    #     None, description="Sandbox configuration"
    # )
    # browser_config: Optional[BrowserSettings] = Field(
    #     None, description="Browser configuration"
    # )
    # search_config: Optional[SearchSettings] = Field(
    #     None, description="Search configuration"
    # )
    # mcp_config: Optional[MCPSettings] = Field(None, description="MCP configuration")
    # run_flow_config: Optional[RunflowSettings] = Field(
    #     None, description="Run flow configuration"
    # )

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    # self._get_initial_embed_model_config()
                    self._initialized = True


    # def _get_initial_embed_model_config(self):
    #     raw_config = self._load_config()
    #     base_embeding = raw_config.get("embeding", "")
    #     embeding_overrides = {
    #         k: v for k, v in raw_config.get("embeding", {}).items() if isinstance(v, dict)
    #     }

    #     default_settings = {
    #         "model": base_embeding.get("model"),
    #         "base_url": base_embeding.get("base_url"),
    #         "api_key": base_embeding.get("api_key"),
    #         "max_tokens": base_embeding.get("max_tokens", 4096),
    #         "max_input_tokens": base_embeding.get("max_input_tokens"),
    #         "temperature": base_embeding.get("temperature", 1.0),
    #         "api_type": base_embeding.get("api_type", ""),
    #         "api_version": base_embeding.get("api_version", ""),
    #     }

    #     config_dict = {
    #         "embeding": {
    #             "default": default_settings,
    #             **{
    #                 name: {**default_settings, **override_config} for name, override_config in embeding_overrides.items()
    #             },
    #         },
    #     }
    #     self._config = AppConfig(**config_dict)

    # def _load_initial_llm_config(self):
    #     raw_config = self._load_config()
    #     base_llm = raw_config.get("llm", {})
    #     llm_overrides = {
    #         k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
    #     }

    #     default_settings = {
    #         "model": base_llm.get("model"),
    #         "base_url": base_llm.get("base_url"),
    #         "api_key": base_llm.get("api_key"),
    #         "max_tokens": base_llm.get("max_tokens", 4096),
    #         "max_input_tokens": base_llm.get("max_input_tokens"),
    #         "temperature": base_llm.get("temperature", 1.0),
    #         "api_type": base_llm.get("api_type", ""),
    #         "api_version": base_llm.get("api_version", ""),
    #     }

    #     config_dict = {
    #         "llm": {
    #             "default": default_settings,
    #             **{
    #                 name: {**default_settings, **override_config}
    #                 for name, override_config in llm_overrides.items()
    #             },
    #         },
    #     }

    #     self._config = AppConfig(**config_dict)

    def _load_initial_config(self):
        raw_config = self._load_config()

        # === embedding ===
        base_embeding = raw_config.get("embeding", {})
        embeding_overrides = {
            k: v for k, v in base_embeding.items() if isinstance(v, dict)
        }

        embed_default = {
            "model": base_embeding.get("model"),
            "base_url": base_embeding.get("base_url"),
            "api_key": base_embeding.get("api_key"),
            "max_tokens": base_embeding.get("max_tokens", 4096),
            "max_input_tokens": base_embeding.get("max_input_tokens"),
            "temperature": base_embeding.get("temperature", 1.0),
            "api_type": base_embeding.get("api_type", ""),
            "api_version": base_embeding.get("api_version", ""),
        }

        embed_dict = {
            "default": embed_default,
            **{name: {**embed_default, **cfg} for name, cfg in embeding_overrides.items()},
        }

        # === llm ===
        base_llm = raw_config.get("llm", {})
        llm_overrides = {k: v for k, v in base_llm.items() if isinstance(v, dict)}

        llm_default = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        llm_dict = {
            "default": llm_default,
            **{name: {**llm_default, **cfg} for name, cfg in llm_overrides.items()},
        }

        # === 最终构建 AppConfig ===
        self._config = AppConfig(
            embeding=embed_dict,
            llm=llm_dict
        )




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

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_llm_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
        }

        self._config = AppConfig(**config_dict)

         # handle browser config.
        # browser_config = raw_config.get("browser", {})
        # browser_settings = None

        # if browser_config:
        #     # handle proxy settings.
        #     proxy_config = browser_config.get("proxy", {})
        #     proxy_settings = None

        #     if proxy_config and proxy_config.get("server"):
        #         proxy_settings = ProxySettings(
        #             **{
        #                 k: v
        #                 for k, v in proxy_config.items()
        #                 if k in ["server", "username", "password"] and v
        #             }
        #         )

        #     # filter valid browser config parameters.
        #     valid_browser_params = {
        #         k: v
        #         for k, v in browser_config.items()
        #         if k in BrowserSettings.__annotations__ and v is not None
        #     }

        #     # if there is proxy settings, add it to the parameters.
        #     if proxy_settings:
        #         valid_browser_params["proxy"] = proxy_settings

        #     # only create BrowserSettings when there are valid parameters.
        #     if valid_browser_params:
        #         browser_settings = BrowserSettings(**valid_browser_params)

        # search_config = raw_config.get("search", {})
        # search_settings = None
        # if search_config:
        #     search_settings = SearchSettings(**search_config)
        # sandbox_config = raw_config.get("sandbox", {})
        # if sandbox_config:
        #     sandbox_settings = SandboxSettings(**sandbox_config)
        # else:
        #     sandbox_settings = SandboxSettings()

        # mcp_config = raw_config.get("mcp", {})
        # mcp_settings = None
        # if mcp_config:
        #     # Load server configurations from JSON
        #     mcp_config["servers"] = MCPSettings.load_server_config()
        #     mcp_settings = MCPSettings(**mcp_config)
        # else:
        #     mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        # run_flow_config = raw_config.get("runflow")
        # if run_flow_config:
        #     run_flow_settings = RunflowSettings(**run_flow_config)
        # else:
        #     run_flow_settings = RunflowSettings()

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm
    

    @property
    def embeding(self) -> Dict[str, LLMSettings]:
        return self._config.embeding

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


config = Config()
