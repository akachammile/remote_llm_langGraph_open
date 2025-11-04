import streamlit as st
import streamlit as st
import requests
import json
import os
from typing import List, Optional, Dict
from datetime import datetime

# --- 页面配置 & 主题设置 ---
st.set_page_config(
    page_title="🤖 遥感智能体", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "### 🤖 遥感智能体支持多模态问答、文档解析和图像处理"}
)

# --- 自定义 CSS 样式 ---
st.markdown("""
<style>
    /* 全局样式 */
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
    }
    
    /* 主容器 - 居中对齐，固定宽度 */
    [data-testid="stMainBlockContainer"] {
        max-width: 800px;
        margin: 0 auto;
        padding: 1.5rem 2rem !important;
    }
    
    /* 侧边栏 - 清洁风格 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3561 0%, #1a1f3a 100%);
        padding: 1.5rem 1rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* 页面标题 */
    [data-testid="stAppViewContainer"] > section > div:first-child h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.3rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stCaptionContainer {
        margin-bottom: 1.2rem !important;
    }
    
    /* 对话标题 - 简洁风格 */
    .chat-title-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    .chat-title-section h2 {
        color: white !important;
        font-size: 1rem !important;
        margin: 0 !important;
        -webkit-text-fill-color: white !important;
        font-weight: 600;
    }
    
    /* 聊天消息 */
    .stChatMessage {
        margin: 0.6rem 0 !important;
        padding: 0.75rem 0.9rem !important;
        border-radius: 8px;
    }
    
    .stChatMessage[aria-label*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: auto;
        max-width: 85%;
    }
    
    .stChatMessage[aria-label*="user"] p {
        color: white !important;
        font-size: 0.95rem;
    }
    
    .stChatMessage[aria-label*="assistant"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        margin-right: auto;
        max-width: 85%;
    }
    
    .stChatMessage[aria-label*="assistant"] p {
        font-size: 0.95rem;
    }
    
    /* 侧边栏元素对齐 */
    [data-testid="stSidebar"] .stMarkdown {
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 1.2rem !important;
        -webkit-text-fill-color: white !important;
        margin: 0 0 0.8rem 0 !important;
        text-align: center;
    }
    
    /* 侧边栏按钮统一风格 */
    [data-testid="stSidebar"] .stButton {
        width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        padding: 0.6rem 0.8rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        border-color: #667eea !important;
    }
    
    /* 主按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
    }
    
    /* 下载按钮 */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* 分隔线 */
    [data-testid="stSidebar"] hr {
        margin: 0.8rem 0 !important;
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* 侧边栏底部信息 */
    [data-testid="stSidebar"] .sidebar-footer {
        text-align: center;
        margin-top: 2rem;
        padding: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        line-height: 1.4;
    }
    
    /* 输入框 - 关键！保持宽度一致 */
    .stChatInputContainer {
        max-width: 600px !important;
        margin: 0 auto !important;
        padding: 0.6rem 0 !important;
        width: 100% !important;
    }
    
    .stChatInput {
        max-width: 600px !important;
        margin: 0 auto !important;
        width: 100% !important;
    }
    
    .stChatInput input {
        border-radius: 20px !important;
        border: 1px solid #ddd !important;
        padding: 0.75rem 1.2rem !important;
        font-size: 0.9rem !important;
        background-color: #f8f9fa !important;
        width: 100% !important;
    }
    
    .stChatInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
        background-color: white !important;
    }
    
    /* 消息样式 */
    .stSuccess {
        padding: 0.6rem 0.8rem !important;
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border-left: 3px solid #10b981 !important;
        border-radius: 4px !important;
        font-size: 0.9rem !important;
        margin: 0.4rem 0 !important;
    }
    
    .stError {
        padding: 0.6rem 0.8rem !important;
        background-color: #fee2e2 !important;
        color: #7f1d1d !important;
        border-left: 3px solid #ef4444 !important;
        border-radius: 4px !important;
        font-size: 0.9rem !important;
        margin: 0.4rem 0 !important;
    }
    
    .stWarning {
        padding: 0.6rem 0.8rem !important;
        background-color: #fef3c7 !important;
        color: #78350f !important;
        border-left: 3px solid #f59e0b !important;
        border-radius: 4px !important;
        font-size: 0.9rem !important;
        margin: 0.4rem 0 !important;
    }
    
    .stInfo {
        padding: 0.6rem 0.8rem !important;
        background-color: #dbeafe !important;
        color: #1e40af !important;
        border-left: 3px solid #3b82f6 !important;
        border-radius: 4px !important;
        font-size: 0.9rem !important;
        margin: 0.4rem 0 !important;
    }
    
    /* 响应式 */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem !important;
        }
        
        .stChatMessage[aria-label*="user"],
        .stChatMessage[aria-label*="assistant"] {
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 配置 & 常量 ---
CHAT_BACKEND_URL = "http://127.0.0.1:7861/api/v1/chat/chat"
UPLOAD_BACKEND_URL = "http://127.0.0.1:7861/api/v1/chat/upload"

st.set_page_config(page_title="🌏遥感智能体", layout="wide")
st.title("🌏遥感智能体")
st.caption("🚀 Qwen多模态智能体, 支持问答、文档解析！")

# --- API 调用函数 ---

# 1. 用于上传文件的函数
def call_upload_api(files: List) -> Optional[List[str]]:
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

# --- Session State 和侧边栏 ---
if "conversations" not in st.session_state:
    st.session_state.conversations = [{"title": "对话 1", "messages": []}]
    st.session_state.current_chat_index = 0

if "sample_triggered" not in st.session_state:
    st.session_state.sample_triggered = None

if "prefilled_query" not in st.session_state:
    st.session_state.prefilled_query = ""

if "prefilled_files" not in st.session_state:
    st.session_state.prefilled_files = []

# 定义模拟上传文件的辅助类
class MockUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data
    def getvalue(self):
        return self._data
    @property
    def type(self):
        ext = self.name.split('.')[-1].lower()
        mime_types = {
            'tif': 'image/tiff',
            'tiff': 'image/tiff',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
        }
        return mime_types.get(ext, 'application/octet-stream')

with st.sidebar:
    # 侧边栏标题
    st.markdown("""
    <h1 style="text-align: center; margin-bottom: 0.8rem;">💬</h1>
    <h3 style="text-align: center; color: white; margin: 0 0 1rem 0; -webkit-text-fill-color: white; font-size: 1rem;">AI 对话</h3>
    """, unsafe_allow_html=True)
    
    # 新建对话按钮
    if st.button("📝 新建对话", use_container_width=True):
        new_chat_index = len(st.session_state.conversations)
        st.session_state.conversations.append({"title": f"对话 {new_chat_index + 1}", "messages": []})
        st.session_state.current_chat_index = new_chat_index
        st.rerun()
    
    st.divider()
    
    # 最近对话列表
    st.markdown("""
    <h4 style="color: rgba(255, 255, 255, 0.6); margin: 0.4rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">
        最近对话
    </h4>
    """, unsafe_allow_html=True)
    
    for i, conv in enumerate(st.session_state.conversations):
        if st.button(
            f"💭 {conv['title']}", 
            key=f"conv_{i}", 
            use_container_width=True
        ):
            st.session_state.current_chat_index = i
            st.rerun()
    
    # 侧边栏底部
    st.divider()
    st.markdown("""
    <div class="sidebar-footer">
        <p>🚀 Qwen2-VL</p>
        <p>模型赋能</p>
    </div>
    """, unsafe_allow_html=True)

# --- 主聊天界面 ---
current_conv = st.session_state.conversations[st.session_state.current_chat_index]

# 对话标题
st.markdown(f"""
<div class="chat-title-section">
    <h2>💬 {current_conv['title']}</h2>
</div>
""", unsafe_allow_html=True)

# --- 样例按钮区域 (固定显示) ---
st.markdown("""
<style>
    .sample-container {
        margin: 1rem auto 1.5rem auto;
        max-width: 720px;
        text-align: center;
    }
    .sample-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .sample-button-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    .sample-description {
        font-size: 0.75rem;
        color: #6b7280;
        line-height: 1.3;
        margin-top: 0.3rem;
        min-height: 2.6rem;
    }
    [data-testid="column"] .stButton > button {
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.25) !important;
    }
    [data-testid="column"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sample-container">', unsafe_allow_html=True)
st.markdown('<h4 class="sample-title">✨ 快速开始 - 试试这些功能</h4>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 图像解读", key="sample_seg", use_container_width=True):
        st.session_state.prefilled_query = "请解读该图像内容后填入内置的doc文档"
        
        # 加载本地文件
        local_file_path = "/home/lmc_workspace/remote_llm_langGraph/test/image/airport_301.png"
        if os.path.exists(local_file_path):
            try:
                with open(local_file_path, "rb") as f:
                    file_data = f.read()
                file_name = os.path.basename(local_file_path)
                st.session_state.prefilled_files = [MockUploadedFile(file_name, file_data)]
                st.info(f"💡 已自动加载样例文件: {file_name}，请点击发送按钮提交")
            except Exception as e:
                st.warning(f"⚠️ 加载本地文件失败: {e}")
        st.rerun()
    st.markdown('<p class="sample-description">选取上传遥感图像给模型后，模型读取图片内容并写入，附带的专业文档报告，提供下载功能</p>', unsafe_allow_html=True)

with col2:
    if st.button("🛰️ 遥感图像处理", key="sample_image", use_container_width=True):
        st.session_state.prefilled_query = "请对这张遥感图像进行分割处理"
        
        # 加载本地文件
        local_file_path = "/home/lmc_workspace/remote_llm_langGraph/test/image/airport_301.png"
        if os.path.exists(local_file_path):
            try:
                with open(local_file_path, "rb") as f:
                    file_data = f.read()
                file_name = os.path.basename(local_file_path)
                st.session_state.prefilled_files = [MockUploadedFile(file_name, file_data)]
                st.info(f"💡 已自动加载样例文件: {file_name}，请点击发送按钮提交")
            except Exception as e:
                st.warning(f"⚠️ 加载本地文件失败: {e}")
        st.rerun()
    st.markdown('<p class="sample-description">针对提供的图像进行分割处理</p>', unsafe_allow_html=True)

with col3:
    if st.button("💬 智能问答", key="sample_qa", use_container_width=True):
        st.session_state.prefilled_query = "你好!请介绍一下你的功能和能力"
        st.session_state.prefilled_files = []
        st.info(f"💡 已填充示例问题，请点击发送按钮提交")
        st.rerun()
    st.markdown('<p class="sample-description">多轮对话理解需求,提供专业的遥感知识问答服务</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

# 聊天历史消息
for msg in current_conv["messages"]:
    with st.chat_message(msg["role"]):
        if "content" in msg:
            st.write(msg["content"])
        if "files" in msg:
            for file_info in msg["files"]:
                file_name = file_info["name"]
                file_ext = file_name.split('.')[-1].lower()
                # 只显示图片文件
                if file_ext in ["tif", "png", "jpeg", "jpg", "gif"]:
                    try:
                        st.image(file_info["data"], caption=file_name, width=200)
                    except Exception as e:
                        st.warning(f"无法显示图像 {file_name}: {e}")
                else:
                    # 其他文件类型显示文件名
                    st.markdown(f"📎 **{file_name}**")

# 显示预填充的问题和文件预览（输入框上方的待发送区域）
if st.session_state.prefilled_query or st.session_state.prefilled_files:
    st.markdown("""<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 16px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                <h4 style='color: white; margin: 0 0 12px 0; font-size: 1rem;'>✨ 待发送内容</h4>
                </div>""", unsafe_allow_html=True)
    
    with st.container():
        # 显示问题文本
        if st.session_state.prefilled_query:
            st.markdown(f"""<div style='background: white; padding: 14px 16px; border-radius: 8px; 
                        margin-bottom: 12px; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                        <strong style='color: #667eea;'>💬 问题:</strong><br/>
                        <span style='color: #374151; font-size: 0.95rem;'>{st.session_state.prefilled_query}</span>
                        </div>""", unsafe_allow_html=True)
        
        # 显示文件
        if st.session_state.prefilled_files:
            st.markdown("""<div style='background: white; padding: 14px 16px; border-radius: 8px; 
                        margin-bottom: 12px; border-left: 4px solid #10b981; box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                        <strong style='color: #10b981;'>📎 已加载文件:</strong></div>""", unsafe_allow_html=True)
            
            cols = st.columns(min(3, len(st.session_state.prefilled_files)))
            for idx, file in enumerate(st.session_state.prefilled_files):
                with cols[idx % len(cols)]:
                    file_ext = file.name.split('.')[-1].lower()
                    if file_ext in ["tif", "png", "jpeg", "jpg", "gif"]:
                        st.image(file.getvalue(), caption=file.name, use_column_width=True)
                    else:
                        st.markdown(f"""<div style='background: #f3f4f6; padding: 12px; border-radius: 6px; text-align: center;'>
                                    📄 <strong>{file.name}</strong></div>""", unsafe_allow_html=True)
        
        # 发送和取消按钮
        col_send, col_cancel = st.columns([1, 1])
        with col_send:
            send_prefilled = st.button("📤 发送此内容", key="send_prefilled", use_container_width=True, type="primary")
        with col_cancel:
            cancel_prefilled = st.button("❌ 取消", key="cancel_prefilled", use_container_width=True)
        
        if cancel_prefilled:
            st.session_state.prefilled_query = ""
            st.session_state.prefilled_files = []
            st.rerun()
        
        if send_prefilled:
            # 处理发送逻辑
            user_text = st.session_state.prefilled_query
            uploaded_files = st.session_state.prefilled_files
            
            # 清除预填充状态
            st.session_state.prefilled_query = ""
            st.session_state.prefilled_files = []
            
            # 显示用户消息
            st.chat_message("user").write(f"🔊 {user_text}")
            user_message = {"role": "user", "content": user_text, "files": []}
            
            if uploaded_files:
                with st.chat_message("user"):
                    for file in uploaded_files:
                        bytes_data = file.getvalue()
                        file_ext = file.name.split('.')[-1].lower()
                        if file_ext in ["tif", "png", "jpeg", "jpg", "gif"]:
                            st.image(bytes_data, caption=file.name, width=200)
                        else:
                            st.markdown(f"📎 **{file.name}**")
                        if isinstance(user_message.get("files"), list):
                            user_message["files"].append({"name": file.name, "data": bytes_data})
            
            current_conv["messages"].append(user_message)
            
            # 上传文件
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
            
            # 调用后端API
            if upload_ok:
                metadata_dict = {}
                if server_filenames:
                    metadata_dict["files"] = [{"saved_path": name} for name in server_filenames]
                
                with st.chat_message("assistant"):
                    with st.spinner("AI 正在思考中..."):
                        reply_content = call_chat_api(user_text, metadata_dict)
                        if reply_content is not None:
                            reply_content = json.loads(reply_content)
                        if reply_content and reply_content.get("messages"):
                            first_message_content = reply_content["messages"][0].get("content", "")
                            st.write(first_message_content)
                            assistant_message = {"role": "assistant", "content": first_message_content}
                            current_conv["messages"].append(assistant_message)
                        
                        # 处理图像
                        processed_files = reply_content.get("processed_image_path", []) if reply_content else []
                        if processed_files:
                            st.markdown("""
                            <div style="margin-top: 1rem;">
                                <h4 style="color: #667eea; font-weight: 600;">🖼️ 处理后的图像</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            cols = st.columns(min(3, len(processed_files)))
                            for idx, file_path in enumerate(processed_files):
                                try:
                                    with cols[idx % len(cols)]:
                                        if file_path.startswith('http'):
                                            st.image(file_path, use_container_width=True)
                                        elif os.path.exists(file_path):
                                            with open(file_path, "rb") as f:
                                                image_data = f.read()
                                            st.image(image_data, use_container_width=True)
                                        else:
                                            st.warning(f"⚠️ 图像文件不存在: {file_path}")
                                except Exception as e:
                                    st.error(f"无法显示图像 {file_path}: {e}")
                        
                        # 处理文档
                        processed_docs = reply_content.get("processed_doc_path", []) if reply_content else []
                        if processed_docs:
                            st.markdown("""
                            <div style="margin-top: 1rem;">
                                <h4 style="color: #667eea; font-weight: 600;">📄 处理后的文档</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for doc_path in processed_docs:
                                try:
                                    file_bytes = None
                                    doc_name = doc_path.split("/")[-1]
                                    mime_type = "application/octet-stream"
                                    
                                    if doc_path.startswith("http"):
                                        file_bytes = requests.get(doc_path).content
                                    else:
                                        if os.path.exists(doc_path):
                                            with open(doc_path, "rb") as f:
                                                file_bytes = f.read()
                                    
                                    if file_bytes:
                                        if doc_name.lower().endswith(".docx"):
                                            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                        elif doc_name.lower().endswith(".doc"):
                                            mime_type = "application/msword"
                                        elif doc_name.lower().endswith(".pdf"):
                                            mime_type = "application/pdf"
                                        elif doc_name.lower().endswith((".txt", ".md")):
                                            mime_type = "text/plain"
                                        
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.markdown(f"""
                                            <div style="background: white; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.06);">
                                                <strong>📄 {doc_name}</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with col2:
                                            st.download_button(
                                                label="⬇️ 下载",
                                                data=file_bytes,
                                                file_name=doc_name,
                                                mime=mime_type,
                                                key=f"doc_prefilled_{hash(doc_path)}",
                                                use_container_width=True
                                            )
                                        
                                        if doc_name.lower().endswith((".txt", ".md")):
                                            try:
                                                preview_text = file_bytes.decode("utf-8")[:500]
                                                st.text_area("", value=preview_text, height=120, disabled=True, label_visibility="collapsed")
                                            except:
                                                st.info("📄 文本文件，可下载查看完整内容。")
                                        elif doc_name.lower().endswith(".docx"):
                                            st.success("✅ Word 文档已处理，点击上方下载按钮获取。")
                                        elif doc_name.lower().endswith(".pdf"):
                                            st.success("✅ PDF 文档已处理，点击上方下载按钮获取。")
                                        
                                        if "processed_docs" not in current_conv:
                                            current_conv["processed_docs"] = []
                                        current_conv["processed_docs"].append(doc_path)
                                
                                except Exception as e:
                                    st.error(f"无法显示或下载文档: {e}")
    
    st.markdown("<div style='margin: 16px 0; border-top: 2px dashed #e5e7eb;'></div>", unsafe_allow_html=True)

# 普通输入框（用户自己输入）
if prompt_data := st.chat_input(
    "💬 输入消息或上传文件...", 
    accept_file="multiple", 
    file_type=["tif", "png", "jpeg", "jpg", "docx", "doc", "pdf", "txt"]
):
    user_text = prompt_data.text
    uploaded_files = list(prompt_data.files) if prompt_data.files else []

    st.chat_message("user").write(f"🔊 {user_text}")
    user_message = {"role": "user", "content": user_text, "files": []}
    if uploaded_files:
        with st.chat_message("user"):
            for file in uploaded_files:
                bytes_data = file.getvalue()
                file_ext = file.name.split('.')[-1].lower()
                # 只展示图片文件
                if file_ext in ["tif", "png", "jpeg", "jpg", "gif"]:
                    st.image(bytes_data, caption=file.name, width=200)
                else:
                    # 其他文件类型显示文件名
                    st.markdown(f"📎 **{file.name}**")
                if isinstance(user_message.get("files"), list):
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
                if reply_content is not None:
                    reply_content = json.loads(reply_content)
                if reply_content and reply_content.get("messages"):
                    first_message_content = reply_content["messages"][0].get("content", "")
                    st.write(first_message_content)
                    assistant_message = {"role": "assistant", "content": first_message_content}
                    current_conv["messages"].append(assistant_message)

                # 处理后的图像
                # processed_files = reply_content['processed_image_path']
                processed_files = reply_content.get("processed_image_path", []) if reply_content else []

                if processed_files:
                    st.markdown("""
                    <div style="margin-top: 1rem;">
                        <h4 style="color: #667eea; font-weight: 600;">🖼️ 处理后的图像</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(min(3, len(processed_files)))
                    for idx, file_path in enumerate(processed_files):
                        try:
                            with cols[idx % len(cols)]:
                                # 判断是URL还是本地路径
                                if file_path.startswith('http'):
                                    st.image(file_path, use_container_width=True)
                                elif os.path.exists(file_path):
                                    with open(file_path, "rb") as f:
                                        image_data = f.read()
                                    st.image(image_data, use_container_width=True)
                                else:
                                    st.warning(f"⚠️ 图像文件不存在: {file_path}")
                        except Exception as e:
                            st.error(f"无法显示图像 {file_path}: {e}")
                            
                processed_docs = reply_content.get("processed_doc_path", []) if reply_content else []
                if processed_docs:
                    st.markdown("""
                    <div style="margin-top: 1rem;">
                        <h4 style="color: #667eea; font-weight: 600;">📄 处理后的文档</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for doc_path in processed_docs:
                        try:
                            file_bytes = None
                            doc_name = doc_path.split("/")[-1]
                            mime_type = "application/octet-stream"
                            
                            # 支持 URL 或本地路径两种情况
                            if doc_path.startswith("http"):
                                file_bytes = requests.get(doc_path).content
                            else:
                                if os.path.exists(doc_path):
                                    with open(doc_path, "rb") as f:
                                        file_bytes = f.read()
                                else:
                                    st.warning(f"⚠️ 文档文件不存在: {doc_path}")
                            
                            if file_bytes:
                                # 根据文件类型设置 MIME 类型
                                if doc_name.lower().endswith(".docx"):
                                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                elif doc_name.lower().endswith(".doc"):
                                    mime_type = "application/msword"
                                elif doc_name.lower().endswith(".pdf"):
                                    mime_type = "application/pdf"
                                elif doc_name.lower().endswith((".txt", ".md")):
                                    mime_type = "text/plain"
                                
                                # 洋气卡片样式
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"""
                                    <div style="background: white; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.06);">
                                        <strong>📄 {doc_name}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    st.download_button(
                                        label="⬇️ 下载",
                                        data=file_bytes,
                                        file_name=doc_name,
                                        mime=mime_type,
                                        key=f"doc_{doc_path}",
                                        use_container_width=True
                                    )
                                
                                # 文本文件预览
                                if doc_name.lower().endswith((".txt", ".md")):
                                    try:
                                        preview_text = file_bytes.decode("utf-8")[:500]
                                        st.markdown("""
                                        <details style="background: #f5f5f5; padding: 12px; border-radius: 6px; margin-top: 8px;">
                                        <summary style="cursor: pointer; font-weight: 500; color: #667eea;">🔍 预览内容</summary>
                                        </details>
                                        """, unsafe_allow_html=True)
                                        st.text_area("", value=preview_text, height=120, disabled=True, label_visibility="collapsed")
                                    except:
                                        st.info("📄 文本文件，可下载查看完整内容。")
                                elif doc_name.lower().endswith(".docx"):
                                    st.success("✅ Word 文档已处理，点击上方下载按钮获取。")
                                elif doc_name.lower().endswith(".pdf"):
                                    st.success("✅ PDF 文档已处理，点击上方下载按钮获取。")
                                
                                # 保存历史记录
                                if "processed_docs" not in current_conv:
                                    current_conv["processed_docs"] = []
                                current_conv["processed_docs"].append(doc_path)
                        
                        except Exception as e:
                            st.error(f"无法显示或下载文档: {e}")   
