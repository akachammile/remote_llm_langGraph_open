import streamlit as st
import requests
import json
from typing import List, Optional, Dict

# --- 配置 & 常量 ---
CHAT_BACKEND_URL = "http://127.0.0.1:7861/api/v1/chat/chat"
UPLOAD_BACKEND_URL = "http://127.0.0.1:7861/api/v1/chat/upload"

st.set_page_config(page_title="🤖 多模态智能体", layout="wide")
st.title("🤖 AI-Powered Agent")
st.caption("🚀 Qwen多模态智能体, 支持问答、文档解析！")

# --- API 调用函数 ---

# 1. 用于上传文件的函数
def call_upload_api(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[List[str]]:
    """调用后端的 /upload 接口，只上传文件。"""
    try:
        files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in files]
        response = requests.post(UPLOAD_BACKEND_URL, files=files_to_send, timeout=180)

        if response.status_code == 200:
            return response.json().get("uploaded_files")
        else:
            st.error(f"文件上传失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"文件上传时发生网络错误: {e}")
        return None

# 2. 聊天接口函数，发送JSON
def call_chat_api(query_text: str, metadata: Dict) -> Optional[str]:
    """以JSON格式调用后端的 /chat 接口。"""
    try:
        # 直接将原生 Python 对象放入 payload
        
        payload = {
            "query": query_text, 
            "metadata": json.dumps(metadata), # 直接传递字典
            "stream": False # 直接传递布尔值
        }
        print(payload)
        # requests的 `json` 参数会自动处理序列化
        response = requests.post(CHAT_BACKEND_URL, json=payload, timeout=180)

        if response.status_code == 200:
            return response.text
        else:
            st.error(f"后端请求失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"连接后端时发生网络错误: {e}")
        return None

# --- Session State 和侧边栏 (无变动) ---
if "conversations" not in st.session_state:
    st.session_state.conversations = [{"title": "对话 1", "messages": []}]
    st.session_state.current_chat_index = 0

with st.sidebar:
    st.title("✍️ 对话列表")
    if st.button("🆕 新对话"):
        new_chat_index = len(st.session_state.conversations)
        st.session_state.conversations.append({"title": f"对话 {new_chat_index + 1}", "messages": []})
        st.session_state.current_chat_index = new_chat_index
        st.rerun()
    st.divider()
    for i, conv in enumerate(st.session_state.conversations):
        if st.button(conv["title"], key=f"conv_{i}", use_container_width=True):
            st.session_state.current_chat_index = i
            st.rerun()

# --- 主聊天界面 ---
current_conv = st.session_state.conversations[st.session_state.current_chat_index]
st.header(current_conv["title"])

for msg in current_conv["messages"]:
    with st.chat_message(msg["role"]):
        if "content" in msg:
            st.write(msg["content"])
        if "files" in msg:
            for file_info in msg["files"]:
                st.image(file_info["data"], caption=file_info["name"], width=200)

if prompt_data := st.chat_input(
    "请输入消息或上传文件...", 
    accept_file="multiple", 
    file_type=["tif", "png", "jpeg", "jpg", "docx"]
):
    user_text = prompt_data.text
    uploaded_files = prompt_data.files

    st.chat_message("user").write(user_text)
    user_message = {"role": "user", "content": user_text}
    if uploaded_files:
        user_message["files"] = []
        with st.chat_message("user"):
            for file in uploaded_files:
                bytes_data = file.getvalue()
                st.image(bytes_data, caption=file.name, width=200)
                user_message["files"].append({"name": file.name, "data": bytes_data})
    current_conv["messages"].append(user_message)

    # --- 两步式提交流程 ---
    server_filenames = []
    upload_ok = True

    if uploaded_files:
        with st.spinner("正在上传文件..."):
            returned_names = call_upload_api(uploaded_files)
            if returned_names:
                server_filenames = returned_names
                st.success(f"文件 {', '.join(server_filenames)} 上传成功！")
            else:
                upload_ok = False
    
    if upload_ok:
        metadata_dict = {}
        if server_filenames:
            metadata_dict["files"] = [{"saved_path": name} for name in server_filenames]
        


        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考中..."):
                # 直接传递 Python 字典，而不是 str(metadata_dict)
                reply_content = call_chat_api(user_text, metadata_dict)
                print(reply_content, type(reply_content))
                reply_content = json.loads(reply_content)
                if reply_content:
                    st.write(next((m["content"] for m in reversed(reply_content["messages"])), ""))
                    assistant_message = {"role": "assistant", "content": reply_content["messages"][0]["content"]}
                    current_conv["messages"].append(assistant_message)
                # 处理后的图像
                # processed_files = reply_content['processed_image_path']
                processed_files = reply_content.get("processed_image_path", [])

                if processed_files:
                    st.write("🖼️ 处理后的图像：")
                    for file_data in processed_files:
                        try:
                            if file_data.startswith('http'):
                                st.image(file_data, caption="处理后的图像", width=300)
                            else:
                                st.image(file_data, caption="处理后的图像", width=300)

                            # Save processed images to conversation history
                            if "processed_images" not in current_conv:
                                current_conv["processed_images"] = []
                            current_conv["processed_images"].append(file_data)
                        except Exception as e:
                            st.error(f"无法显示图像: {e}")
                            
                processed_docs = reply_content.get("processed_doc_path", [])
                if processed_docs:
                    st.write("📄 处理后的文档：")
                    for doc_path in processed_docs:
                        try:
                            # 支持 URL 或本地路径两种情况
                            if doc_path.startswith("http"):
                                doc_name = doc_path.split("/")[-1]
                                st.markdown(f"**{doc_name}**")
                                st.download_button(
                                    label="⬇️ 下载文档",
                                    data=requests.get(doc_path).content,
                                    file_name=doc_name,
                                    mime="application/octet-stream"
                                )
                            else:
                                doc_name = doc_path.split("/")[-1]
                                with open(doc_path, "rb") as f:
                                    file_bytes = f.read()
                                st.markdown(f"**{doc_name}**")
                                st.download_button(
                                    label="⬇️ 下载文档",
                                    data=file_bytes,
                                    file_name=doc_name,
                                    mime="application/octet-stream"
                                )

                            # 简单预览文本文件内容（可选）
                            if doc_name.lower().endswith((".txt", ".md")):
                                st.text(file_bytes.decode("utf-8")[:1000])
                            elif doc_name.lower().endswith(".docx"):
                                st.info("📘 该文档为 Word 文件，可下载查看内容。")
                            elif doc_name.lower().endswith(".pdf"):
                                st.info("📕 该文档为 PDF 文件，可下载查看内容。")

                            # 保存历史记录
                            if "processed_docs" not in current_conv:
                                current_conv["processed_docs"] = []
                            current_conv["processed_docs"].append(doc_path)

                        except Exception as e:
                            st.error(f"无法显示或下载文档: {e}")   

            # Reload processed images from conversation history
            # if "processed_images" in current_conv:
            #     st.write("🖼️ 历史处理图像：")
            #     for img_data in current_conv["processed_images"]:
            #         try:
            #             if img_data.startswith('http'):
            #                 st.image(img_data, caption="历史处理图像", use_column_width=True)
            #             else:
            #                 st.image(img_data, caption="历史处理图像", use_column_width=True)
            #         except Exception as e:
            #             st.error(f"无法显示历史图像: {e}")
