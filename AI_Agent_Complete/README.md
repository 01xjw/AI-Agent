# AI Agent LLM性能分析器 - 完整部署包

## 📦 这是什么？

这是一个**完整的、开箱即用的** AI Agent LLM性能分析器部署包。

所有文件已经整理好，路径已经配置正确，可以直接运行。

---

## 🚀 快速开始（3步）

### 第1步：安装依赖

```bash
pip install -r requirements.txt
```

### 第2步：配置必要的路径

打开 `config.yaml` 文件，修改以下内容：

```yaml
# 修改这两个路径为你的实际路径
sglang_path: "D:/Code/sglang"        # ← 你的SGlang代码路径
models_path: "D:/Models"              # ← 你的模型文件路径
```

### 第3步：启动服务

```bash
python start.py
```

然后浏览器打开：**http://localhost:8000/chat**

---

## 📁 目录结构

```
AI_Agent_Complete/
├── start.py                  # 启动脚本（运行这个）
├── config.yaml              # 配置文件（修改这个）
├── requirements.txt         # 依赖列表
├── README.md               # 本文件
├── backend/                # 后端服务
│   ├── web_server.py       # Web服务器
│   ├── agent_core.py       # AI Agent核心
│   └── utils/
│       ├── nsys_parser.py  # NSys解析器
│       └── ncu_parser.py   # NCU解析器
└── frontend/               # 前端界面
    └── chat.html           # 聊天界面
```

---

## ⚙️ 配置说明

### 必须配置的项：

1. **SGlang路径** - `config.yaml` 中的 `sglang_path`
   ```yaml
   sglang_path: "D:/Code/sglang"  # 改为你的路径
   ```

2. **模型路径** - `config.yaml` 中的 `models_path`
   ```yaml
   models_path: "D:/Models"  # 改为你的路径
   ```

### 可选配置：

- **服务器端口**：默认8000，可在 `config.yaml` 修改
- **模型映射**：在 `config.yaml` 中的 `model_mappings` 部分配置

---

## ✅ 运行前检查

运行以下命令检查环境：

```bash
# 检查Python
python --version          # 需要 Python 3.8+

# 检查NVIDIA工具
nvidia-smi               # 检查GPU
nsys --version           # 检查NSight Systems
ncu --version            # 检查NSight Compute

# 检查依赖
pip list | findstr "fastapi pandas"
```

---

## 🎯 使用示例

### 启动服务后：

1. 浏览器打开 http://localhost:8000/chat
2. 在对话框输入：
   ```
   分析 llama-7b 模型，batch_size=8
   ```
3. AI会自动解析并开始分析

### 支持的命令格式：

```
分析 llama-7b，batch_size=8,16
对 qwen-14b 进行 nsys 全局分析
综合分析 chatglm-6b 的性能瓶颈
使用 ncu 深度分析 vicuna-7b
```

---

## 🐛 常见问题

### 问题1：启动失败

**检查**：
```bash
# 查看详细错误
python start.py
```

**可能原因**：
- 依赖未安装：运行 `pip install -r requirements.txt`
- 端口被占用：修改 `config.yaml` 中的 `port`

### 问题2：找不到模型

**检查**：
- `config.yaml` 中的 `models_path` 是否正确
- 模型文件是否存在
- `model_mappings` 配置是否正确

### 问题3：SGlang命令执行失败

**检查**：
- `config.yaml` 中的 `sglang_path` 是否正确
- SGlang是否已安装：`cd <sglang_path> && python -m sglang.launch_server --help`

---

## 📞 获取帮助

1. 查看日志：运行 `start.py` 时的输出
2. 检查配置：`cat config.yaml`
3. 测试连接：`curl http://localhost:8000/health`

---

## 🔄 更新日志

- v1.0.0: 初始版本，整理完整部署包
- 包含路径修复
- 统一配置文件
- 简化启动流程

---

## 💡 下一步

成功运行后，你可以：

1. 上传配置文件进行分析
2. 查看生成的性能报告
3. 根据建议优化模型性能

祝使用愉快！🎉

