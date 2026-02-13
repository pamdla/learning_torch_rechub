FROM python:3.12-slim
LABEL maintainer="agent1237@datawhalechina.cn"
LABEL team="智能推荐①②③⑦"
LABEL project="learning_torch_rechub"
LABEL git.repo="https://github.com/pamdla/learning_torch_rechub"

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# # 安装 uv (用于开发环境)
# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
#     && pip install --no-cache-dir uv

# 安装 PyTorch (CPU版本，如需GPU请使用 nvidia/cuda 基础镜像)
RUN pip install --no-cache-dir torch~=2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 torch-rechub 稳定版
RUN pip install --no-cache-dir torch-rechub~=0.3.0

# 安装其他常用数据科学依赖
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    jupyter \
    matplotlib \
    seaborn

RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    onnx onnxruntime

# 验证安装
RUN python -c "import torch_rechub; print(f'Torch-RecHub version: {torch_rechub.__version__}')"

# 暴露 Jupyter 端口
EXPOSE 8899

# 默认命令
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8899", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
