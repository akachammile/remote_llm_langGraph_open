from typing import Union, List, Optional
from langgraph.graph import MessagesState
from typing_extensions import Annotated


class AgentState(MessagesState):
    name: Annotated[Optional[str], None] # Agent名称
    question: Annotated[str, None] # 当前问题
    next_agent: Annotated[str, None] # 下一步提示
    last_agent: Annotated[str, None]
    tool_require: Annotated[Optional[str], None] # 当前问题 是否有使用的工具
    too_call: Annotated[Optional[str], None] # 当前问题需要使用的工具

    # TODO 历史信息，需进一步修改
    history: Annotated[Optional[List[str]], None]
    memory: Annotated[Optional[str], None] # 当前问题 # 需要使用的工具
    image_data: Annotated[Union[bytes, List[bytes]],None]# 图像数据，二进制格式
    image_path: Annotated[Optional[str], None]
    image_format: Annotated[str, None]# 图像格式，如 "png", "jpg"
    image_content: Annotated[str, None]# 图像内容描述
    processed_image_path: Union[List, str] # 分割图像存储路径
    processed_doc_path: Union[List, str]
    reflection: Annotated[Optional[bool], None]
    sub_task: Annotated[Optional[List[str]], None]
    
    max_steps: Annotated[Optional[int], None] # 最大步骤数
    repeat_step: Annotated[Optional[int], None]

    image_uri: Annotated[Optional[str], None] # 持久化的图像路径/URI
    initial_image_description: Annotated[Optional[str], None] # 图像的初始文字描述
    is_related: Annotated[bool, True] # 当前问题是否与图像上下文相关
    
    # state: str # Agent当前状态
   
    # current_step: int # 当前步骤数
