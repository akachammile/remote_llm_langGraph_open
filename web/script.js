// ========== DOM 元素 ==========
const toggleBtn = document.getElementById('toggleBtn');
const sidebarContainer = document.querySelector('.sidebar-container');
const newChatBtn = document.getElementById('newChatBtn');
const sendBtn = document.getElementById('sendBtn');
const inputField = document.getElementById('inputField');
const plusMenuBtn = document.getElementById('plusMenuBtn');
const plusMenuContent = document.getElementById('plusMenuContent');
const uploadFileMenuBtn = document.getElementById('uploadFileMenuBtn');
const uploadImageMenuBtn = document.getElementById('uploadImageMenuBtn');
const fileInput = document.getElementById('fileInput');
const imageInput = document.getElementById('imageInput');
const conversationContainer = document.getElementById('conversationContainer');
const chatList = document.getElementById('chatList');
const chatTitle = document.getElementById('chatTitle');

// ========== 状态管理 ==========
const state = {
  currentChatId: null,
  chats: [],
  messages: {},
  attachments: []
};

// 创建轮消板 (API 上传)
const API_BASE = 'http://127.0.0.1:7861/api/v1/chat/chat';

// ========== 事件监听器 ==========

// 侧边栏折叠按钮
toggleBtn.addEventListener('click', () => {
  sidebarContainer.classList.toggle('collapsed');
});

// 加号菜单切换
plusMenuBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  plusMenuContent.classList.toggle('show');
});

// 点击其他地方关闭plus菜单
document.addEventListener('click', (e) => {
  if (!plusMenuBtn.contains(e.target) && !plusMenuContent.contains(e.target)) {
    plusMenuContent.classList.remove('show');
  }
});

// 文件上传按钮
uploadFileMenuBtn.addEventListener('click', () => {
  fileInput.click();
  plusMenuContent.classList.remove('show');
});

// 图片上传按钮
uploadImageMenuBtn.addEventListener('click', () => {
  imageInput.click();
  plusMenuContent.classList.remove('show');
});

// 新对话
newChatBtn.addEventListener('click', () => {
  const chatId = Date.now().toString();
  const newChat = {
    id: chatId,
    title: `对话 ${state.chats.length + 1}`,
    createdAt: new Date()
  };
  state.chats.push(newChat);
  state.messages[chatId] = [];
  selectChat(chatId);
  renderChatList();
});

// 发送消息
sendBtn.addEventListener('click', sendMessage);
inputField.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 图片选择变化
imageInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files);
  files.forEach(file => {
    const reader = new FileReader();
    reader.onload = (event) => {
      addAttachment(file, event.target.result, 'image');
    };
    reader.readAsDataURL(file);
  });
  e.target.value = ''; // 清空input,允许重复选择相同文件
});

// 文件选择变化
fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files);
  files.forEach(file => {
    addAttachment(file, null, 'file');
  });
  e.target.value = ''; // 清空input
});

// ========== 函数 ==========

function selectChat(chatId) {
  state.currentChatId = chatId;
  const chat = state.chats.find(c => c.id === chatId);
  chatTitle.textContent = chat.title;
  renderMessages();
  document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active'));
  const activeItem = document.querySelector(`[data-chat-id="${chatId}"]`);
  if (activeItem) activeItem.classList.add('active');
}

function renderChatList() {
  chatList.innerHTML = '';
  state.chats.forEach(chat => {
    const li = document.createElement('li');
    li.className = 'chat-item';
    li.textContent = chat.title;
    li.dataset.chatId = chat.id;
    if (chat.id === state.currentChatId) li.classList.add('active');
    li.addEventListener('click', () => selectChat(chat.id));
    chatList.appendChild(li);
  });
}

function renderMessages() {
  const messages = state.messages[state.currentChatId] || [];
  const hasMessages = messages.length > 0;
  const inputSection = document.getElementById('inputSection');
  
  // 动态切换输入框位置
  if (hasMessages) {
    conversationContainer.classList.add('has-messages');
    inputSection.classList.add('fixed-bottom');
  } else {
    conversationContainer.classList.remove('has-messages');
    inputSection.classList.remove('fixed-bottom');
  }

  if (messages.length === 0) {
    document.getElementById('conversationInner').innerHTML = '<div class="empty-state"><div class="empty-state-icon">💬</div><div class="empty-state-title">开始新对话</div><div class="empty-state-desc">发送消息或上传图片/文件来开始与 AI 的对话</div></div>';
    return;
  }

  const innerDiv = document.getElementById('conversationInner');
  innerDiv.innerHTML = '';
  messages.forEach(msg => {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${msg.role}`;
    
    // 构建消息内容容器
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 如果有附件，先显示附件
    if (msg.attachments && msg.attachments.length > 0) {
      const attachmentsDiv = document.createElement('div');
      attachmentsDiv.className = 'message-attachments';
      msg.attachments.forEach(att => {
        if (att.type === 'image') {
          const img = document.createElement('img');
          img.src = att.data;
          img.className = 'attachment';
          img.alt = att.name;
          attachmentsDiv.appendChild(img);
        } else {
          const fileDiv = document.createElement('div');
          fileDiv.className = 'attachment-file';
          fileDiv.textContent = `📎 ${att.name}`;
          attachmentsDiv.appendChild(fileDiv);
        }
      });
      contentDiv.appendChild(attachmentsDiv);
    }
    
    // 如果有文本，显示文本
    if (msg.text) {
      const textDiv = document.createElement('div');
      textDiv.textContent = msg.text;
      if (msg.attachments && msg.attachments.length > 0) {
        textDiv.style.marginTop = '8px';
      }
      contentDiv.appendChild(textDiv);
    }
    
    msgEl.appendChild(contentDiv);
    innerDiv.appendChild(msgEl);
  });
  conversationContainer.scrollTop = conversationContainer.scrollHeight;
}

async function sendMessage() {
  const text = inputField.value.trim();
  if (!text && state.attachments.length === 0) return;

  if (!state.currentChatId) {
    newChatBtn.click();
  }

  // 添加用户消息
  const userMsg = {
    role: 'user',
    text: text,
    attachments: state.attachments.map(att => ({
      name: att.name,
      type: att.type,
      data: att.data
    }))
  };

  state.messages[state.currentChatId].push(userMsg);
  renderMessages();

  // 清编辑區
  inputField.value = '';
  const currentAttachments = [...state.attachments];
  state.attachments = [];
  renderAttachmentsPreview(); // 清空预览
  inputField.focus();

  // 调用后端 API
  await callBackendAPI(text, currentAttachments);
}

// 调用后端 API
async function callBackendAPI(text, attachments) {
  try {
    // 显示加载状态
    const loadingMsg = {
      role: 'assistant',
      text: '正在思考中...',
      attachments: [],
      isLoading: true
    };
    state.messages[state.currentChatId].push(loadingMsg);
    renderMessages();

    // 准备metadata（如果有附件需要先上传）
    let metadata = {};
    if (attachments.length > 0) {
      // 先上传文件，获取文件信息
      const uploadResult = await uploadFiles(attachments);
      // 将上传结果转换为后端期望的格式
      if (uploadResult.uploaded_files && Array.isArray(uploadResult.uploaded_files)) {
        metadata = {
          files: uploadResult.uploaded_files.map(filename => ({
            saved_path: filename
          }))
        };
      }
      console.log('转换后的metadata:', metadata);
    }

    // 准备请求数据
    const requestBody = {
      query: text,
      metadata: JSON.stringify(metadata),
      stream: false
    };

    console.log('发送请求:', requestBody);

    // 发送请求到后端
    const response = await fetch(API_BASE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    // 移除加载消息
    state.messages[state.currentChatId].pop();

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('API响应:', result);

    // 解析后端返回的数据
    let replyText = '';
    let replyAttachments = [];

    // 从 messages 数组中提取最后一条消息作为回复
    if (result.messages && Array.isArray(result.messages) && result.messages.length > 0) {
      const lastMessage = result.messages[result.messages.length - 1];
      if (lastMessage && lastMessage.content) {
        replyText = lastMessage.content;
      }
    } else if (result.reply) {
      replyText = result.reply;
    } else if (result.message) {
      replyText = result.message;
    } else {
      replyText = '我已收到你的消息。';
    }

    // 处理图片路径（数组格式）
    // if (result.processed_image_path && Array.isArray(result.processed_image_path)) {
    //   result.processed_image_path.forEach(imgPath => {
    //     if (imgPath && typeof imgPath === 'string') {
    //       replyAttachments.push({
    //         name: imgPath.split('/').pop(),
    //         type: 'image',
    //         data: imgPath
    //       });
    //     }
    //   });
    // }

    // // 处理文档路径（数组格式）
    // if (result.processed_doc_path && Array.isArray(result.processed_doc_path)) {
    //   result.processed_doc_path.forEach(docPath => {
    //     if (docPath && typeof docPath === 'string') {
    //       replyAttachments.push({
    //         name: docPath.split('/').pop(),
    //         type: 'file',
    //         data: docPath
    //       });
    //     }
    //   });
    // }

    // 添加 AI 响应消息
    const aiMsg = {
      role: 'assistant',
      text: replyText,
      attachments: replyAttachments
    };

    state.messages[state.currentChatId].push(aiMsg);
    renderMessages();

  } catch (error) {
    console.error('API 调用失败:', error);
    
    // 移除加载消息
    const lastMsg = state.messages[state.currentChatId][state.messages[state.currentChatId].length - 1];
    if (lastMsg && lastMsg.isLoading) {
      state.messages[state.currentChatId].pop();
    }

    // 显示错误消息
    const errorMsg = {
      role: 'assistant',
      text: `抱歉，连接服务器失败: ${error.message}`,
      attachments: [],
      isError: true
    };
    state.messages[state.currentChatId].push(errorMsg);
    renderMessages();
  }
}

// 上传文件到服务器
async function uploadFiles(attachments) {
  const formData = new FormData();
  
  attachments.forEach((att) => {
    if (att.file) {
      formData.append('files', att.file);
    }
  });

  console.log('上传文件...');
  const uploadResponse = await fetch('http://127.0.0.1:7861/api/v1/chat/upload', {
    method: 'POST',
    body: formData
  });

  if (!uploadResponse.ok) {
    throw new Error('文件上传失败');
  }

  const uploadResult = await uploadResponse.json();
  console.log('文件上传结果:', uploadResult);
  
  return uploadResult;
}

// 添加附件并显示预览
function addAttachment(file, dataUrl, type) {
  const attachment = {
    file: file,
    name: file.name,
    type: type,
    data: dataUrl,
    id: Date.now() + Math.random()
  };
  state.attachments.push(attachment);
  renderAttachmentsPreview();
}

// 渲染附件预览
function renderAttachmentsPreview() {
  const previewContainer = document.getElementById('attachmentsPreview');
  
  if (state.attachments.length === 0) {
    previewContainer.classList.remove('show');
    previewContainer.innerHTML = '';
    return;
  }

  previewContainer.classList.add('show');
  previewContainer.innerHTML = '';

  state.attachments.forEach((att, index) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = `attachment-preview-item ${att.type}`;

    if (att.type === 'image') {
      itemDiv.innerHTML = `
        <img src="${att.data}" class="attachment-preview-img" alt="${att.name}">
        <div class="attachment-preview-name">${att.name}</div>
        <div class="attachment-remove-btn" data-index="${index}">×</div>
      `;
    } else {
      itemDiv.innerHTML = `
        <svg width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
          <path d="M4.5 3a2.5 2.5 0 0 1 5 0v9a1.5 1.5 0 0 1-3 0V5a.5.5 0 0 1 1 0v7a.5.5 0 0 0 1 0V3a1.5 1.5 0 1 0-3 0v9a2.5 2.5 0 0 0 5 0V5a.5.5 0 0 1 1 0v7a3.5 3.5 0 1 1-7 0V3z"/>
        </svg>
        <div class="attachment-preview-name">${att.name}</div>
        <div class="attachment-remove-btn" data-index="${index}">×</div>
      `;
    }

    previewContainer.appendChild(itemDiv);
  });

  // 绑定删除按钮事件
  document.querySelectorAll('.attachment-remove-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const index = parseInt(e.target.dataset.index);
      removeAttachment(index);
    });
  });
}

// 删除附件
function removeAttachment(index) {
  state.attachments.splice(index, 1);
  renderAttachmentsPreview();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ========== 初始化 ==========
// 上敳来我的对话
const initialChat = {
  id: '1',
  title: '新对话',
  createdAt: new Date()
};
state.chats.push(initialChat);
state.messages['1'] = [];
selectChat('1');
