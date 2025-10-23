import re
import io
import cv2
import json
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Literal
from ultralytics import YOLO
from app.logger import logger
from datetime import datetime
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from typing import Dict
from docx.oxml.ns import qn
from app.cores.config import config
from app.tools.base import BaseTool, ToolFailure, ToolResult

_FILE_PROCESS_TOOL = """
执行DOCX文档分析和填充任务，根据输入的文档模板内容，自动识别文档中的标题信息，并结合模型生成的内容，对文档进行智能填充。
输入可以是文档路径或文本内容，输出包含填充后的文档及相关元数据。
"""

class FileProcessTool(BaseTool):
    name: str = "file_process_tool"
    description: str = _FILE_PROCESS_TOOL
    parameters: dict = {
        "type": "object",
        "properties": {
            "doc_template_path": {
                "type": "string",
                "description": "待填充的文档模板路径，必须是有效的 DOCX 文件路径。",
            },
            "image_content": {
                "type": "string",
                "description": (
                    "模型分析得到的内容，键为文档标题（可能带编号），"
                    "值为要填充的内容（文本、列表或字典形式）。"
                ),
            },
            "question": {
                "type": "string",
                "description": "本次任务的描述或用户问题，用于指导模型生成内容。",
            },
            "output_path": {
                "type": "string",
                "description": "生成文档的输出路径，必须以 .docx 结尾。",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    # model_path: str = config.model_path / config.YOLO_SEGMENTATION_MODEL_NAME
    # conf: float = config.YOLO_SEGMENTATION_CONF
    # iou: float = config.YOLO_SEGMENTATION_IOU
    # output_path: str = (
    #     config.file_post_process_path / config.YOLO_SEGMENTATION_OUTPUT_DIR
    # )
 
    def execute(self, doc_template_path:str, image_content: str, title: str):
        """_summary_

        Args:
            image_content (str): 图像描述
            title (str): 文档标题

        Returns:
            tuple[bytes, list, str]: _description_
        """
        logger.info(f"✅ 执行doc_agent")
        content_to_fill = image_content
        document = Document(self.doc_template)
        cleaned_key_map = {re.sub(r'^\d+(\.\d+)*\s*', '', k).strip(): k for k in content_to_fill.keys()}

        for para in document.paragraphs:
            # 只处理标题段落
            if not para.style.name.startswith('标题'):
                continue
            
            para_text_cleaned = para.text.strip()

            # --- 调试输出 (非常重要) ---
            logger.debug(f"正在检查 DOCX 段落: '{para_text_cleaned}' (样式: {para.style.name})")
            
            # 使用清理过的文本进行匹配
            if para_text_cleaned in cleaned_key_map:
                # 找到匹配后，使用原始的、带编号的键来获取内容
                original_key = cleaned_key_map[para_text_cleaned]
                content_to_add = content_to_fill[original_key]
                
                logger.info(f"✅ 找到匹配: '{para_text_cleaned}' -> 准备填充内容...")
                
                # --- 填充内容的逻辑 ---
                new_p = OxmlElement("w:p") 
                para._p.addnext(new_p)
                new_run = new_p.makeelement(qn('w:r'))
                new_p.append(new_run)
                new_t = new_run.makeelement(qn('w:t'))
                new_run.append(new_t)
                
                if isinstance(content_to_add, (dict, list)):
                    content_to_add = json.dumps(content_to_add, ensure_ascii=False, indent=2)
                elif not isinstance(content_to_add, str):
                    content_to_add = str(content_to_add)

                new_t.text = content_to_add
                # new_t.text = content_to_add
            else:
                logger.warning(f"❌ 未找到匹配: '{para_text_cleaned}' 在 {list(cleaned_key_map.keys())} 中不存在。")
        # document.save(self.output_path)
        # state["next_agent"] = "exit"
        # state["sub_task"] = "文档生成并写入任务已完成"
        # state["processed_doc_path"] = [self.output_path]
        # return state