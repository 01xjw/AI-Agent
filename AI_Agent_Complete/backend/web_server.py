#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent Web服务器
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# 导入AI Agent核心
try:
    from agent_core import AIAgent
except ImportError:
    print("警告: 无法导入agent_core，使用简化模式")
    AIAgent = None

# 加载配置
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(
    title="AI Agent LLM性能分析器",
    description="智能的大语言模型性能分析Web服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# 全局变量
agent = None
active_connections: Dict[str, WebSocket] = {}

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"🔗 连接建立: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"❌ 连接断开: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"发送消息失败: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global agent
    if AIAgent:
        try:
            agent = AIAgent(CONFIG)
            print("✅ AI Agent初始化成功")
        except Exception as e:
            print(f"⚠️ AI Agent初始化失败: {e}")
    
    print("🤖 AI Agent Web服务器启动完成")
    print(f"📡 服务地址: http://{CONFIG['server']['host']}:{CONFIG['server']['port']}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Agent LLM性能分析器</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>🤖 AI Agent LLM性能分析器</h1>
        <p>请访问 <a href="/chat">/chat</a> 开始使用</p>
        <p>API文档: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """聊天页面"""
    chat_file = Path(__file__).parent.parent / "frontend" / "chat.html"
    if chat_file.exists():
        return chat_file.read_text(encoding='utf-8')
    else:
        return HTMLResponse(
            content="<h1>聊天页面未找到</h1>",
            status_code=404
        )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket连接端点"""
    await manager.connect(websocket, session_id)
    
    # 发送欢迎消息
    await manager.send_message(session_id, {
        "type": "assistant_message",
        "content": """🤖 **欢迎使用AI Agent LLM性能分析器！**

我可以帮您：
• 🔍 分析各种LLM模型的性能
• 📊 进行NSys全局性能分析
• 🔬 执行NCU深度kernel分析
• 💡 提供性能优化建议

请告诉我您的分析需求！例如：
"分析 llama-7b 模型，batch_size=8"
""",
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            await handle_websocket_message(session_id, message_data)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        manager.disconnect(session_id)

async def handle_websocket_message(session_id: str, message_data: dict):
    """处理WebSocket消息"""
    
    message_type = message_data.get("type", "")
    content = message_data.get("content", "")
    
    if message_type == "user_message":
        await process_user_message(session_id, content)
    
    elif message_type == "ping":
        await manager.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

async def process_user_message(session_id: str, message: str):
    """处理用户消息"""
    
    try:
        await manager.send_message(session_id, {
            "type": "assistant_message",
            "content": f"🔄 正在处理您的请求: {message}",
            "timestamp": datetime.now().isoformat()
        })
        
        # 简化版响应（实际应调用AIAgent）
        if agent:
            response = await agent.process_message(message)
        else:
            response = f"""✅ 已接收到您的请求

**请求内容**: {message}

📋 **解析结果**:
• 这是一个模拟响应
• 完整功能需要配置SGlang和模型路径
• 请查看 config.yaml 进行配置

💡 **下一步**:
1. 配置 config.yaml 中的路径
2. 确保SGlang已安装
3. 准备好模型文件
4. 重新启动服务

详细配置说明请查看 README.md
"""
        
        await manager.send_message(session_id, {
            "type": "assistant_message",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "content": f"❌ 处理失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

@app.post("/upload_config")
async def upload_config(file: UploadFile = File(...)):
    """上传配置文件"""
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # 解析配置
        if file.filename.endswith('.json'):
            config_data = json.loads(content_str)
        elif file.filename.endswith(('.yaml', '.yml')):
            config_data = yaml.safe_load(content_str)
        else:
            return {"error": "不支持的文件格式"}
        
        return {
            "filename": file.filename,
            "message": "配置文件上传成功",
            "config": config_data
        }
        
    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "agent_ready": agent is not None,
        "config_loaded": CONFIG is not None
    }

@app.get("/config")
async def get_config():
    """获取配置信息"""
    return {
        "sglang_path": CONFIG.get('sglang_path'),
        "models_path": CONFIG.get('models_path'),
        "server": CONFIG.get('server'),
        "model_mappings": CONFIG.get('model_mappings', {})
    }

if __name__ == "__main__":
    # 获取配置
    host = CONFIG.get('server', {}).get('host', '0.0.0.0')
    port = CONFIG.get('server', {}).get('port', 8000)
    
    # 启动服务
    uvicorn.run(
        "web_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

