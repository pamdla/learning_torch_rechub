> 声明：本仓库为Datawhale学习小组打卡所用，记录学习进度、问题、心得等。
> 
> 也欢迎star、fork等，谢谢关注。
> 

# Torch-RecHub的个人学习项目

基于 [Datawhale Torch-RecHub](https://datawhalechina.github.io/torch-rechub/) 的推荐系统算法学习仓库。

## 项目简介

本项目用于系统学习推荐系统算法，基于Torch-RecHub框架进行实践。Torch-RecHub是一个用于推荐系统的PyTorch库，支持CTR预估、召回、排序等多种推荐任务。

**官方资源:**
- 文档: https://datawhalechina.github.io/torch-rechub/
- GitHub: https://github.com/datawhalechina/torch-rechub
- 安装指南: https://datawhalechina.github.io/torch-rechub/guide/install.html

## 学习目标

下面是官方Datawhale建议的[学习任务清单](./assets/tasks.png)

可以拆分成如下部分，逐个完成：

- [x] 掌握推荐系统基础概念
- [x] 学习CTR预估模型(DeepFM,Wide&Deep,xDeepFM等)
- [x] 学习召回模型 (DSSM,YouTubeDNN,MIND等)
- [x] 学习排序模型
- [x] 掌握多任务学习 (MMoE,PLE,ESMM等)
- [ ] 实践完整推荐系统流程

## 快速开始

### 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- (可选) NVIDIA Docker Runtime (用于 GPU 版本)

### 启动环境

```bash
# 1. 克隆本仓库
git clone <your-repo-url>
cd learning_torch_rechub

# 2. 创建必要的目录
mkdir -p workspace data

# 3. 启动服务
docker-compose up -d

# 4. 访问 Jupyter
# 打开浏览器访问: http://localhost:8888
```

### 验证安装

在 Jupyter Notebook 中运行:

```python
import torch_rechub
import torch

print(f"Torch-RecHub 版本: {torch_rechub.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
```

## 项目结构

```
learning_torch_rechub/
├── 📁 workspace/           # 工作目录 (你的代码和笔记)
│   ├── 📁 notebooks/       # Jupyter 笔记本
│   ├── 📁 src/            # Python 源码
│   └── 📄 README.md       # 个人学习笔记
│
├── 📁 assets/             # 资源物料目录
│   └── 📄 tasks.png       # 本课程表
│
├── 📁 data/               # 数据集目录
│   ├── 📁 raw/            # 原始数据
│   └── 📁 processed/      # 处理后数据
│
├── 📁 task1/              # 各任务目录
│   └── 📄 README.md       # 本任务学习笔记
├── 📁 task2/              # 各任务目录
│   ├── 📄 Day2-RecallSys.ipynb  # 代码笔记本
│   └── 📄 README.md       # 本任务学习笔记
├── 📁 task3/              # 各任务目录
│   ├── 📄 Day3-ranking-model.ipynb  # 代码笔记本
│   └── 📄 README.md       # 本任务学习笔记
├── 📁 task4/              # 多任务学习 (MMoE)
│   ├── 📄 demo.py        # MMoE多任务学习demo
│   ├── 📄 README.md       # 本任务学习笔记
│   └── 📄 CHANGELOG.md   # 版本变更记录
│
├── 📄 Dockerfile          # CPU 版本镜像
├── 📄 Dockerfile.gpu      # GPU 版本镜像
├── 📄 Dockerfile.dev      # 开发环境镜像
├── 📄 docker-compose.yml  # 服务编排
├── 📄 .dockerignore       # Docker 忽略文件
└── 📄 README.md           # 本文件
```

## 学习路线

### 阶段〇: 环境搭建与基础 (1天)

1. 搭建 Docker 环境
2. 熟悉 Torch-RecHub API
3. 跑通第一个示例

### 阶段一: CTR 预估 (1-2 周)

- [ ] LR (逻辑回归)
- [ ] FM (因子分解机)
- [x] DeepFM
- [x] Wide & Deep
- [ ] DCN (Deep Cross Network)
- [ ] xDeepFM
- [ ] AutoInt

### 阶段二: 召回模型 (1-2 周)

- [x] DSSM (双塔模型)
- [ ] YouTubeDNN
- [ ] MIND (多兴趣网络)
- [ ] SINE
- [ ] SDM

### 阶段三: 多任务学习 (1 周)

- [ ] Shared Bottom
- [x] MMoE (多门混合专家)
- [ ] PLE (渐进式分层提取)
- [ ] ESMM (多任务样本加权)

### 阶段四: 项目实战 (2 周)

- [ ] 完整推荐系统流程
- [ ] 特征工程实践
- [ ] 模型训练与评估
- [ ] 模型部署

## 推荐数据集

| 数据集 | 描述 | 适用任务 |
|--------|------|----------|
| Criteo | 广告点击率数据集 | CTR 预估 |
| MovieLens | 电影评分数据 | 召回/排序 |
| Amazon | 商品评论数据 | 召回/排序 |
| Avazu | 移动广告数据 | CTR 预估 |

数据集下载:
```bash
# 创建数据目录（可参考）
mkdir -p data/criteo data/movielens

# 在 Jupyter 中下载数据
# 或使用 wget/curl 下载到 data/ 目录
```

## 常用命令

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 进入容器
docker exec -it torch-rechub-cpu bash

# 停止服务
docker-compose down

# 重新构建镜像
docker-compose build

# 查看运行状态
docker-compose ps
```

## 学习资源

### 官方文档
- [安装指南](https://datawhalechina.github.io/torch-rechub/guide/install.html)
- [快速开始](https://datawhalechina.github.io/torch-rechub/guide/quick_start.html)
- [核心概念](https://datawhalechina.github.io/torch-rechub/core/intro.html)
- [模型介绍](https://datawhalechina.github.io/torch-rechub/models/intro.html)
- [API 文档](https://datawhalechina.github.io/torch-rechub/api/api.html)

### 推荐论文

**CTR 预估:**
- Wide & Deep (Google, 2016)
- DeepFM (Huawei, 2017)
- DCN (Google, 2017)
- xDeepFM (Microsoft, 2018)

**召回:**
- DSSM (Microsoft, 2013)
- YouTubeDNN (Google, 2016)
- MIND (Alibaba, 2019)

**多任务:**
- MMoE (Google, 2018)
- PLE (Tencent, 2020)
- ESMM (Alibaba, 2018)

### 推荐书籍
- 《深度学习推荐系统》(王喆)
- 《推荐系统实践》(项亮)

## 学习笔记模板

在 `workspace/` 目录下创建你的学习笔记:

```markdown
# 日期: YYYY-MM-DD

## 学习内容

### 模型名称
- 论文链接:
- 核心思想:
- 创新点:

### 代码实现
```python
# 你的代码
```

### 实验结果
- 数据集:
- 评估指标:
- 结果记录:

### 总结与思考
- 收获:
- 疑问:
- 下一步计划:
```

## 贡献

欢迎分享你的学习笔记和代码!

1. Fork 本仓库
2. 创建你的学习分支
3. 提交学习笔记
4. 推送到你的仓库

## 许可证

本项目仅用于学习目的。

Torch-RecHub 遵循其自身的许可证。

## 联系方式

如有问题，请参考:
- [Torch-RecHub GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
- [Datawhale 社区](https://github.com/datawhalechina)

---

**Happy Learning! 🚀**
