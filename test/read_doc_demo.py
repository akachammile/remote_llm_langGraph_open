from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def insert_paragraph_after(paragraph, text):
      """
      在指定的段落后面插入一个新段落。
      此版本修正了 OxmlElement 的兼容性问题。
      """
      # --- 修改之处 ---
      # 移除了旧版本不支持的 nsmap 参数
      new_p = OxmlElement("w:p") 
      # --- 修改结束 ---
      
      paragraph._p.addnext(new_p)
      new_run = new_p.makeelement(qn('w:r'))
      new_p.append(new_run)
      new_t = new_run.makeelement(qn('w:t'))
      new_run.append(new_t)
      new_t.text = text
      return new_p

def fill_content_by_headings(doc, content_map):
      """
      根据标题文本查找标题，并在其后填充内容。
      content_map 是一个字典 {heading_text: content_to_insert}
      """
      for para in doc.paragraphs:
          if para.style.name.startswith('标题') and para.text in content_map:
              print(f"找到标题: '{para.text}'，准备在其后填充内容...")
              content_to_add = content_map[para.text]
              insert_paragraph_after(para, content_to_add)

template_file_path = r"E:\1_LLM_PROJECT\remote_llm_langGraph\app\template\xinjiang.docx"  # <--- 修改这里

  # 2. 指定输出文件的路径
output_file_path = "filled_report.docx"  # <--- 修改这里

  # 3. 定义标题和要填充内容的映射
content_to_fill = {
      "目标坐落概述": "这是将要填充在“目标坐落概述”标题下的新内容。",
      "目标整体分析": "这是将要填充在“目标整体分析”标题下的新内容。",
  }

  # =================================================================
  # --- 主逻辑部分 ---
  # =================================================================

if __name__ == '__main__':
      try:
          document = Document(template_file_path)
          print(f"成功加载模板文件: {template_file_path}")

          fill_content_by_headings(document, content_to_fill)

          document.save(output_file_path)
          print(f"填充完成，新文件已保存至: {output_file_path}")

      except FileNotFoundError:
          print(f"错误：找不到文件 '{template_file_path}'。请检查文件路径是否正确。")
      except Exception as e:
          print(f"处理过程中发生错误: {e}")