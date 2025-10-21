"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List

from app.exceptions import ToolError
from app.logger import logger
from app.tools.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """已定义的工具集合."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"{name} 工具无效")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """按照集合顺序执行工具"""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """向工具集合中添加工具，若工具已经存在则跳过且log"""
        if tool.name in self.tool_map:
            logger.warning(f"工具 {tool.name} 已存在")
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """向工具集合添加多个工具, 若工具已经存在则跳过并且log"""
        for tool in tools:
            self.add_tool(tool)
        return self
