# 1. 使用官方 Python 镜像作为基础镜像
FROM python:3.13-slim

# 2. 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. 设置工作目录
WORKDIR /app

# 4. 安装 uv
RUN pip install uv

# 5. 复制依赖文件并安装依赖
# 使用 uv sync 来利用 uv.lock 文件，可以更快、更确定地安装依赖
COPY pyproject.toml uv.lock* ./
RUN uv sync --system

# 6. 复制所有项目文件到工作目录
COPY . .

# 7. 暴露应用程序运行的端口
EXPOSE 7861

# 8. 设置容器启动时执行的命令
# 使用 0.0.0.0 作为主机以允许从外部访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7861"]
