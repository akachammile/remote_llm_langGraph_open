from typing import Union, List, Optional
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """LangGraph Agent 状态定义
    
    注意:
    - 所有字段均使用覆盖更新策略(新值直接替换旧值)
    - 节点不返回某字段时,该字段保持不变
    - MessagesState 已内置 messages 字段处理
    """
    
    # ========== 基础信息 ==========
    name: Optional[str]  # Agent名称
    question: str  # 当前问题
    last_agent: Optional[str]  # 上一个Agent
    
    # ========== 任务管理 ==========
    next_agent: List[str]  # 待执行Agent队列
    sub_task: List  # 子任务列表(planning结果)
    history: List[str]  # 历史记录
    
    # ========== 工具相关 ==========
    tool_require: Optional[str]  # 当前问题是否需要工具
    too_call: Optional[str]  # 需要调用的工具名
    
    # ========== 记忆与上下文 ==========
    memory: Optional[str]  # 对话记忆
    
    # ========== 图像相关 ==========
    image_data: Union[bytes, str, None]  # 图像数据(二进制/base64)
    image_path: Optional[str]  # 图像文件路径
    image_format: Optional[str]  # 图像格式: "png", "jpg"
    image_content: Optional[str]  # 图像内容描述
    image_uri: Optional[str]  # 持久化路径/URI
    initial_image_description: Optional[str]  # 初始描述
    processed_image_path: List[str]  # 已处理图像路径
    
    # ========== 文档相关 ==========
    processed_doc_path: Union[List, str, None]  # 已处理文档路径
    
    # ========== 控制标志 ==========
    reflection: Optional[bool]  # 是否需要反思
    tasks_initialized: bool  # 任务是否已初始化
    is_related: bool  # 是否与图像上下文相关
    
    # ========== 步骤控制 ==========
    max_steps: Optional[int]  # 最大步骤数
    repeat_step: Optional[int]  # 重复步骤数
    
    # state: str # Agent当前状态
   
    # current_step: int # 当前步骤数
