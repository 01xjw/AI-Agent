#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent核心模块 - 集成NSys和NCU性能分析
"""

import re
import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 导入分析工具
from utils.nsys_to_ncu_analyzer import NSysToNCUAnalyzer, create_sglang_analysis_workflow

class AIAgent:
    """AI Agent核心类 - 自动化性能分析"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sglang_path = Path(config.get('sglang_path', 'SGlang'))
        self.models_path = Path(config.get('models_path', 'models'))
        self.model_mappings = config.get('model_mappings', {})
        self.results_dir = Path(config.get('output', {}).get('results_dir', 'analysis_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # 分析工具配置
        self.profiling_config = config.get('profiling_tools', {})
        self.analysis_defaults = config.get('analysis_defaults', {})
        
    async def process_message(self, message: str) -> str:
        """处理用户消息并执行分析"""
        
        # 提取模型名称
        model_name = self._extract_model_name(message)
        
        # 提取分析类型
        analysis_type = self._extract_analysis_type(message)
        
        # 提取参数
        params = self._extract_parameters(message)
        
        # 如果没有提供参数，使用默认值
        if not params.get('batch_size'):
            params['batch_size'] = self.analysis_defaults.get('batch_size', [8])
        if not params.get('input_len'):
            params['input_len'] = self.analysis_defaults.get('input_len', [512])
        if not params.get('output_len'):
            params['output_len'] = self.analysis_defaults.get('output_len', [64])
        
        # 生成初始响应
        response = f"""✅ **已解析您的请求**

🤖 **模型**: {model_name or '未指定'}
🔬 **分析类型**: {analysis_type}
📊 **参数**:
  • batch_size: {params.get('batch_size', [])}
  • input_len: {params.get('input_len', [])}
  • output_len: {params.get('output_len', [])}

"""
        
        # 如果模型名称明确，执行实际分析
        if model_name:
            # 获取模型路径
            model_path = self._resolve_model_path(model_name)
            
            if not model_path:
                response += f"""
❌ **错误**: 未找到模型 '{model_name}'
📋 可用模型: {', '.join(self.model_mappings.keys())}

💡 **提示**: 请在 config.yaml 中配置模型路径
"""
                return response
            
            response += f"""🚀 **开始分析...**

📁 模型路径: {model_path}
⏳ 预计时间: 3-10分钟（取决于参数组合数量）

"""
            
            # 执行分析（异步）
            try:
                analysis_results = await self._run_analysis(
                    model_path=model_path,
                    analysis_type=analysis_type,
                    params=params
                )
                
                response += analysis_results
                
            except Exception as e:
                response += f"""
❌ **分析失败**: {str(e)}

💡 **可能原因**:
1. NSys/NCU工具未安装或未在PATH中
2. 模型路径不正确
3. GPU不可用或驱动问题
4. 参数配置错误

🔧 **调试步骤**:
1. 运行 `nsys --version` 和 `ncu --version` 检查工具
2. 运行 `nvidia-smi` 检查GPU
3. 检查模型路径是否存在
"""
        else:
            response += """
💡 **下一步**:
请指定要分析的模型名称，例如：
• "分析 llama-7b"
• "对 qwen-14b 进行性能分析"
• "使用 ncu 深度分析 chatglm-6b"

📋 **可用模型**: """ + ', '.join(self.model_mappings.keys())
        
        return response
    
    async def _run_analysis(self, model_path: str, analysis_type: str, params: Dict) -> str:
        """执行实际的性能分析"""
        
        results = []
        
        # 获取参数组合
        batch_sizes = params.get('batch_size', [8])
        input_lens = params.get('input_len', [512])
        output_lens = params.get('output_len', [64])
        
        # 只分析第一组参数（避免时间过长）
        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens
        
        try:
            # 创建分析工作流
            analysis_workflow = create_sglang_analysis_workflow()
            
            # 执行分析
            result_dir = await asyncio.get_event_loop().run_in_executor(
                None,
                analysis_workflow,
                str(model_path),
                batch_size,
                input_len,
                output_len
            )
            
            # 读取生成的报告
            report_file = Path(result_dir) / "integrated_performance_report.md"
            
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                # 提取关键信息
                summary = self._extract_report_summary(report_content)
                
                results.append(f"""
✅ **分析完成!**

📁 **结果目录**: {result_dir}
📄 **报告文件**: {report_file}

{summary}

🔍 **详细报告**: 请查看 {report_file}
📊 **可视化图表**: 请查看结果目录中的图片文件
""")
            else:
                results.append(f"""
⚠️ **分析已完成，但未生成报告文件**

📁 结果目录: {result_dir}
💡 请检查目录中的其他输出文件
""")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            results.append(f"""
❌ **分析执行失败**

错误信息: {str(e)}

详细错误:
```
{error_detail}
```

💡 **常见问题解决**:
1. 确保已安装 nsys 和 ncu 工具
2. 确保 SGlang 已正确安装
3. 确保模型文件路径正确
4. 确保有足够的 GPU 内存
""")
        
        return '\n'.join(results)
    
    def _extract_report_summary(self, report_content: str) -> str:
        """从报告中提取关键摘要信息"""
        
        lines = report_content.split('\n')
        summary_lines = []
        
        # 提取关键统计信息
        for i, line in enumerate(lines):
            if '总kernels数量' in line or '总kernel执行时间' in line:
                summary_lines.append(line)
            elif '🔥 识别的热点Kernels' in line:
                # 提取前3个热点kernel
                summary_lines.append("\n**🔥 热点Kernels (Top 3):**")
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and lines[j].startswith(('1.', '2.', '3.')):
                        summary_lines.append(lines[j][:100])
                break
        
        if summary_lines:
            return '\n'.join(summary_lines)
        else:
            return "**📊 分析报告已生成，请查看详细文件**"
    
    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        """解析模型路径"""
        
        # 检查是否在映射表中
        if model_name in self.model_mappings:
            mapped_path = self.model_mappings[model_name]
            
            # 如果是绝对路径，直接返回
            if Path(mapped_path).is_absolute():
                return mapped_path
            
            # 否则，相对于 models_path
            full_path = self.models_path / mapped_path
            return str(full_path)
        
        # 如果不在映射表中，尝试直接作为路径
        if Path(model_name).exists():
            return model_name
        
        # 尝试相对于 models_path
        potential_path = self.models_path / model_name
        if potential_path.exists():
            return str(potential_path)
        
        return None
    
    def _extract_model_name(self, prompt: str) -> Optional[str]:
        """提取模型名称"""
        
        # 首先检查已知的模型别名
        for model_name in self.model_mappings.keys():
            if model_name.lower() in prompt.lower():
                return model_name
        
        # 然后使用正则表达式匹配通用模型名称模式
        patterns = [
            r'llama[^/\s]*-?\d*[^/\s]*-?\d+[bB]?',
            r'qwen[^/\s]*-?\d*[^/\s]*-?\d+[bB]?',
            r'chatglm[^/\s]*-?\d+[bB]?',
            r'baichuan[^/\s]*-?\d+[bB]?',
            r'vicuna[^/\s]*-?\d+[bB]?',
            r'mistral[^/\s]*-?\d+[bB]?',
            r'mixtral[^/\s]*-?\d+[bB]?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_analysis_type(self, prompt: str) -> str:
        """提取分析类型"""
        prompt_lower = prompt.lower()
        
        if 'ncu' in prompt_lower or 'kernel' in prompt_lower or '深度' in prompt_lower or 'nsight compute' in prompt_lower:
            return 'ncu (深度kernel分析)'
        elif 'nsys' in prompt_lower or '全局' in prompt_lower or 'nsight systems' in prompt_lower:
            return 'nsys (全局性能分析)'
        elif '集成' in prompt_lower or '综合' in prompt_lower or '完整' in prompt_lower:
            return 'auto (集成分析: nsys + ncu)'
        else:
            return 'auto (集成分析: nsys + ncu)'
    
    def _extract_parameters(self, prompt: str) -> Dict:
        """提取参数"""
        params = {}
        
        # 提取batch_size
        batch_match = re.search(r'batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if batch_match:
            batch_sizes = [int(x.strip()) for x in re.split(r'[,，\s]+', batch_match.group(1)) if x.strip()]
            params['batch_size'] = batch_sizes
        
        # 提取input_len
        input_match = re.search(r'input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if input_match:
            input_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', input_match.group(1)) if x.strip()]
            params['input_len'] = input_lens
        
        # 提取output_len
        output_match = re.search(r'output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)', prompt, re.IGNORECASE)
        if output_match:
            output_lens = [int(x.strip()) for x in re.split(r'[,，\s]+', output_match.group(1)) if x.strip()]
            params['output_len'] = output_lens
        
        return params

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.model_mappings.keys())
    
    def get_analysis_status(self) -> Dict:
        """获取当前分析状态"""
        return {
            'available_models': self.get_available_models(),
            'results_directory': str(self.results_dir),
            'nsys_enabled': self.profiling_config.get('nsys', {}).get('enabled', True),
            'ncu_enabled': self.profiling_config.get('ncu', {}).get('enabled', True),
        }
