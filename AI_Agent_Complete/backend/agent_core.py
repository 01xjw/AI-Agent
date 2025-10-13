#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent核心模块
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

class AIAgent:
    """AI Agent核心类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sglang_path = Path(config.get('sglang_path', 'SGlang'))
        self.models_path = Path(config.get('models_path', 'models'))
        self.model_mappings = config.get('model_mappings', {})
    
    async def process_message(self, message: str) -> str:
        """处理用户消息"""
        
        # 提取模型名称
        model_name = self._extract_model_name(message)
        
        # 提取分析类型
        analysis_type = self._extract_analysis_type(message)
        
        # 提取参数
        params = self._extract_parameters(message)
        
        # 生成响应
        response = f"""✅ **已解析您的请求**

🤖 **模型**: {model_name or '未指定'}
🔬 **分析类型**: {analysis_type}
📊 **参数**:
"""
        
        if params.get('batch_size'):
            response += f"  • batch_size: {params['batch_size']}\n"
        if params.get('input_len'):
            response += f"  • input_len: {params['input_len']}\n"
        if params.get('output_len'):
            response += f"  • output_len: {params['output_len']}\n"
        
        response += """

💡 **下一步**:
• 确认配置正确
• 准备模型文件
• 启动分析任务

⚠️ **注意**: 当前为演示模式，完整功能需要：
1. 配置SGlang路径
2. 准备模型文件
3. 安装分析工具(nsys, ncu)
"""
        
        return response
    
    def _extract_model_name(self, prompt: str) -> Optional[str]:
        """提取模型名称"""
        # 模型名称模式
        patterns = [
            r'llama[^/]*-?\d*[^/]*-?\d+[bB]?',
            r'qwen[^/]*-?\d*[^/]*-?\d+[bB]?',
            r'chatglm[^/]*-?\d+[bB]?',
            r'baichuan[^/]*-?\d+[bB]?',
            r'vicuna[^/]*-?\d+[bB]?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_analysis_type(self, prompt: str) -> str:
        """提取分析类型"""
        prompt_lower = prompt.lower()
        
        if 'ncu' in prompt_lower or 'kernel' in prompt_lower or '深度' in prompt_lower:
            return 'ncu'
        elif 'nsys' in prompt_lower or '全局' in prompt_lower:
            return 'nsys'
        elif '集成' in prompt_lower or '综合' in prompt_lower:
            return 'auto'
        else:
            return 'auto'
    
    def _extract_parameters(self, prompt: str) -> Dict:
        """提取参数"""
        params = {}
        
        # 提取batch_size
        batch_match = re.search(r'batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if batch_match:
            batch_sizes = [int(x.strip()) for x in re.split(r'[,，\s]+', batch_match.group(1))]
            params['batch_size'] = batch_sizes
        
        # 提取input_len
        input_match = re.search(r'input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if input_match:
            input_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', input_match.group(1))]
            params['input_len'] = input_lens
        
        # 提取output_len
        output_match = re.search(r'output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if output_match:
            output_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', output_match.group(1))]
            params['output_len'] = output_lens
        
        return params

