# 基于LangGraph的多模态智能体项目，后续会在此基础上形成一个类似于maｕｓ的项目
这是一个使用LangGraph框架构建的问答系统示例，展示了如何创建不同类型的问答图。

## 项目结构

```
remote_llm/
├── app/
│   ├── agents/          # 代理模块
│   │   ├── base.py    
│   │   ├── chat_agent.py # 对话模块
│   │   ├── doc_agent.py    
│   │   └── plan_agent.py
│   ├── graphs/          # 图模块
│   │   ├── qa_graph.py           # 基础问答图
│   │   └── advanced_qa_graph.py  # 进阶问答图
│   ├── model/           # 模型模块
│   │   └── llm.py      # LLM接口
│   ├── schemas/         # 数据模型
│   │   └── chat.py     # 聊天相关模型
│   └── tools/           # 工具模块
│       └── search_tool.py # 搜索和计算工具
├── examples/            # 示例代码
│   └── qa_examples.py  # 完整示例
├── main.py             # 主程序
├── simple_qa_demo.py   # 简单演示
└── pyproject.toml      # 项目配置
```

## 功能特性

### 1. 基础问答图
- 简单的问答处理
- 直接调用LLM生成回答

### 2. 上下文问答图
- 支持多轮对话
- 保持对话历史
- 基于上下文生成回答

### 3. 多步骤问答图
- 问题分析
- 回答验证
- 条件路由

### 4. 进阶问答图（带工具）
- 自动问题类型识别
- 集成搜索工具
- 集成计算工具
- 智能路由

## 快速开始

### 1. 激活虚拟环境

```bash
# Windows PowerShell
.venv\Scripts\activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 2. 运行简单演示

```bash
python simple_qa_demo.py
```

### 3. 运行主程序

```bash
python main.py
```

### 4. 运行完整示例

```bash
python examples/qa_examples.py
```

## 使用示例

### 基础问答

```python
from app.model.llm import MockLLM
from app.agents.qa_agent import QAAgent
from app.graphs.qa_graph import create_qa_graph

# 初始化组件
llm = MockLLM()
qa_agent = QAAgent(llm)
graph = create_qa_graph(qa_agent)

# 运行问答
result = await graph.ainvoke({
    "question": "你好",
    "answer": None,
    "context": {},
    "metadata": {}
})

print(result["answer"])
```

### 带工具的问答

```python
from app.graphs.advanced_qa_graph import create_advanced_qa_graph

# 创建进阶图
graph = create_advanced_qa_graph(qa_agent)

# 运行问答（自动使用工具）
result = await graph.ainvoke({
    "question": "什么是LangGraph?",
    "answer": None,
    "context": {},
    "metadata": {},
    "tool_results": {}
})
```

## 核心概念

### 1. 状态图 (StateGraph)
LangGraph的核心是状态图，它定义了数据如何在节点间流动：

```python
from langgraph.graph import StateGraph, END

# 创建状态图
workflow = StateGraph(QAState)

# 添加节点
workflow.add_node("process_question", qa_agent.process_question)

# 设置边
workflow.add_edge("process_question", END)

# 编译图
graph = workflow.compile()
```

### 2. 条件路由
使用条件边实现智能路由：

```python
workflow.add_conditional_edges(
    "analyze_question",
    route_question,
    {
        "process": "process_question",
        "end": END
    }
)
```

### 3. 状态管理
每个节点都可以修改状态：

```python
def process_question(state: QAState) -> QAState:
    # 处理问题
    answer = generate_answer(state["question"])
    
    # 更新状态
    state["answer"] = answer
    state["metadata"]["processed"] = True
    
    return state
```

## 问答系统类型

### 1. 基础问答 (Basic QA)
最简单的问答形式，直接处理用户问题并生成回答。

**特点：**
- 单轮对话
- 无上下文记忆
- 快速响应

**适用场景：**
- 简单问题回答
- 知识查询
- 快速原型开发

### 2. 上下文问答 (Contextual QA)
支持多轮对话，能够记住对话历史。

**特点：**
- 多轮对话支持
- 上下文记忆
- 对话连贯性

**适用场景：**
- 客服对话
- 教学助手
- 复杂问题解决

### 3. 多步骤问答 (Multi-step QA)
包含问题分析、处理和验证的完整流程。

**特点：**
- 问题质量检查
- 回答验证
- 智能路由

**适用场景：**
- 高质量问答系统
- 企业级应用
- 需要质量控制的服务

### 4. 工具增强问答 (Tool-enhanced QA)
集成外部工具来增强回答能力。

**特点：**
- 自动工具选择
- 多工具集成
- 智能增强

**适用场景：**
- 计算问题
- 知识检索
- 复杂任务处理

## 扩展指南

### 1. 添加新的LLM模型

```python
from app.model.llm import BaseLLM

class CustomLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        # 实现你的LLM调用逻辑
        return "自定义回答"
```

### 2. 添加新的工具

```python
class CustomTool:
    async def execute(self, input_data: str) -> str:
        # 实现工具逻辑
        return "工具结果"
```

### 3. 创建新的图

```python
def create_custom_graph() -> StateGraph:
    workflow = StateGraph(CustomState)
    
    # 添加你的节点和边
    workflow.add_node("custom_node", custom_function)
    workflow.add_edge("custom_node", END)
    
    return workflow.compile()
```

## 最佳实践

1. **状态设计**: 使用TypedDict定义清晰的状态结构
2. **错误处理**: 在节点中添加适当的错误处理
3. **异步支持**: 使用async/await处理异步操作
4. **工具集成**: 将外部工具封装为可重用的组件
5. **测试**: 为每个节点和图编写测试

## 注意事项

- 当前使用MockLLM进行演示，实际使用时需要替换为真实的LLM
- 工具执行是同步的，实际应用中可能需要异步处理
- 状态管理需要根据具体需求进行优化
- 生产环境中需要添加适当的日志和监控

## 演示结果

运行 `python simple_qa_demo.py` 会看到：

```
基于LangGraph的问答系统演示
==================================================
=== 基础问答演示 ===
问题: 你好
回答: 你好！我是基于LangGraph的问答助手。

问题: 什么是LangGraph
回答: LangGraph是一个用于构建LLM应用的框架，支持状态管理和工作流编排。

=== 多步骤问答演示 ===
问题: 什么是LangGraph?
回答: LangGraph是一个用于构建LLM应用的框架，支持状态管理和工作流编排。
元数据: {'analysis': 'proper_question', 'confidence': 'high', 'processed': True}
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！