import traceback
from docx import Document
from typing import Dict
from docx.oxml.ns import qn
from app.logger import logger
from docx.shared import Pt, Inches
from app.agents.base import BaseAgent
from app.graphs.graph_state import AgentState
from app.prompts.doc_prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, START, END


class DocAgent(BaseAgent):
    name: str = "DocAgent"
    description: str = "用于将理文档相关的具体任务，可对文档进行读取, 也可生成DOCX文件"
    system_prompt: str = SYSTEM_PROMPT
    tool: Dict[str, str] = {
        "function_name": "analysis",
    }
    # def __init__(self, output_path="分析结果.docx"):
    #     self.output_path = output_path
    #     self.doc_template = None
    #     self._load_template()
    #     if not self.doc_template:
    #         self.doc_template = self._load_template()
    

    async def analysis(self, state: AgentState):
        try:
            logger.info("开始执行DocAgent")
            question: str = state["question"]
            message:str = state["messages"]
            doc_formatt = ""
            history = self.memory.get_recent_messages(n=30)
            prompt = self.system_prompt.format(question=question, doc_formatt=doc_formatt, content=message)
            analysis_result = await self.llm.ask(prompt, state)
            state["messages"] = analysis_result
            return state
        except Exception as e:
            logger.error(f"LLM调用失败: {traceback.print_exc()}")
            raise Exception(detail=str(e))
    def _load_template(self, template_path:str) -> Document:
        # TODO: 加载模板
        self.doc_template = Document
        return Document
    async def write_analysis(self, analysis_result: AgentState, title: str = "分析结果"):
        doc = Document()
        doc.add_heading(title, 0)

        # 添加正文
        para = doc.add_paragraph(analysis_result["messages"])

        # 设置字体为微软雅黑，字号 12pt
        for run in para.runs:
            run.font.size = Pt(12)
            run.font.name = '微软雅黑'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        if getattr(analysis_result, "segmented_image_path", None):
                img_path = analysis_result["image_path"]
                doc.add_picture(img_path, width=Inches(4))

        doc.save(self.output_path)

    def build_subgraph(self):
        # 创建子图
        doc_subgraph = StateGraph(AgentState)
        doc_subgraph.add_node("analysis", self.analysis)
        doc_subgraph.add_node("doc", self.write_analysis)
        doc_subgraph.add_edge(START, "analysis")
        doc_subgraph.add_edge("analysis", "doc")
        doc_subgraph.add_edge("doc", END) 
        chat_subgraph = doc_subgraph.compile()
        return chat_subgraph