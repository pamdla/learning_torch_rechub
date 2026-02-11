# Torch-RecHub Docker 环境

基于 [官方安装指南](https://datawhalechina.github.io/torch-rechub/guide/install.html) 构建的 Docker 环境。

## 快速开始

### 使用 CPU 版本

```bash
docker-compose up torch-rechub-cpu
```

访问: http://localhost:8888

### 使用 GPU 版本 (需要 NVIDIA Docker Runtime)

```bash
docker-compose up torch-rechub-gpu
```

访问: http://localhost:8889

### 开发环境

```bash
# 克隆源码
git clone https://github.com/datawhalechina/torch-rechub.git

# 启动开发环境
docker-compose up torch-rechub-dev

# 进入容器安装依赖
docker exec -it torch-rechub-dev bash
uv sync
uv pip install -e .
```

访问: http://localhost:8890

## 目录结构

- `workspace/` - 工作目录，会被挂载到容器中
- `data/` - 数据目录，会被挂载到容器中

## 构建镜像

```bash
# CPU 版本
docker build -t torch-rechub:cpu -f Dockerfile .

# GPU 版本
docker build -t torch-rechub:gpu -f Dockerfile.gpu .

# 开发版本
docker build -t torch-rechub:dev -f Dockerfile.dev .
```

## 系统要求

- Docker 20.10+
- Docker Compose 2.0+
- (GPU 版本) NVIDIA Docker Runtime

## 包含的依赖

- Python 3.12
- PyTorch + CUDA (GPU 版本)
- torch-rechub
- NumPy, Pandas, SciPy, Scikit-learn
- Jupyter Notebook
- matplotlib, seaborn
