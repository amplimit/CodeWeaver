import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import subprocess
import sys
from datetime import datetime
import uuid
import re

# 导入代码分析模块
from analyze_codebase import analyze_codebase, CodebaseInfo
from query import (
    load_storage,
    init_vectorizer,
    search_similar_functions,
    print_function_info
)

def prepare_source_path(code_source: str) -> str:
    """准备源代码路径，支持远程代码克隆"""
    if code_source.startswith(("http://", "https://", "git@")):
        try:
            tmp_dir = Path("./tmp")
            tmp_dir.mkdir(exist_ok=True)
            clone_path = tmp_dir / str(uuid.uuid4())
            
            print("\nCloning repository...")
            # 使用subprocess替代os.system，这样我们可以获取更详细的输出
            process = subprocess.Popen(
                ["git", "clone", "--depth", "1", code_source, str(clone_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时显示克隆进度
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    
            # 检查是否成功
            if process.returncode != 0:
                raise Exception("Failed to clone repository")
                
            print("Repository cloned successfully!")
            return str(clone_path)
        except Exception as e:
            print(f"\nError during repository cloning: {str(e)}")
            print("Please make sure git is installed and the repository URL is correct.")
            print("You can also try cloning the repository manually and providing the local path.")
            sys.exit(1)
    return code_source

@dataclass
class ProjectContext:
    """项目上下文信息"""
    codebase_info: CodebaseInfo
    current_focus: Optional[str] = None
    conversation_history: List[Dict] = None
    search_cache: Dict[str, Any] = None  # 新增：搜索缓存
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.search_cache is None:
            self.search_cache = {}

class CodeUnderstandingAI:
    def __init__(
        self,
        model_name: str = "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1",
        device: Optional[str] = None,
        load_in_8bit: bool = True
    ):
        self.setup_logging()
        self.initialize_components(model_name, device, load_in_8bit)
        self.project_context = None
        self.storage = None
        self.vectorizer = None
        
        # 新增：关键词处理器
        self.keyword_handlers = {
            r"\[SEARCH\]": self._handle_search,
            r"\[RAG\]": self._handle_rag,
            r"\[UPDATE\]": self._handle_update,
            r"\[CODE\]": self._handle_code,
            r"\[FOCUS\]": self._handle_focus
        }

    def setup_logging(self):
        """设置增强的日志系统"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'ai_assistant_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = True
    ):
        """增强的模型初始化"""
        self.logger.info(f"Initializing AI with model: {model_name}")
        
        # 智能设备选择
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
                self.logger.info(f"CUDA available, using device: {device}")
            else:
                device = "cpu"
                self.logger.warning("CUDA not available, falling back to CPU")
        self.device = device
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 增强的模型加载
            load_kwargs = {
                "device_map": self.device,
                "trust_remote_code": True
            }
            if load_in_8bit and self.device != "cpu":
                load_kwargs["load_in_8bit"] = True
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            ).eval()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def _handle_search(self, query: str) -> Dict[str, Any]:
        """处理搜索关键词"""
        results = search_similar_functions(self.storage, self.vectorizer, query)
        return {
            'action': 'search',
            'query': query,
            'results': results
        }

    def _handle_rag(self, query: str) -> Dict[str, Any]:
        """处理RAG检索关键词"""
        results = search_similar_functions(self.storage, self.vectorizer, query, k=3)
        context = self._format_search_results(results)
        return {
            'action': 'rag',
            'context': context,
            'results': results
        }

    def _handle_update(self, data: str) -> Dict[str, Any]:
        """处理上下文更新关键词"""
        try:
            update_data = json.loads(data)
            self.project_context.current_focus = update_data.get('focus')
            return {'action': 'update', 'status': 'success'}
        except json.JSONDecodeError:
            return {'action': 'update', 'status': 'failed'}

    def _handle_code(self, code: str) -> Dict[str, Any]:
        """处理代码分析关键词"""
        analysis = self._analyze_code_segment(code)
        return {
            'action': 'code_analysis',
            'code': code,
            'analysis': analysis
        }

    def _handle_focus(self, focus: str) -> Dict[str, Any]:
        """处理焦点切换关键词"""
        self.project_context.current_focus = focus
        return {
            'action': 'focus_change',
            'new_focus': focus
        }

    def _analyze_code_segment(self, code: str) -> Dict[str, Any]:
        """增强的代码分析"""
        similar_patterns = self._find_similar_patterns(code)
        return {
            'similar_patterns': similar_patterns,
            'complexity': self._estimate_complexity(code),
            'suggestions': self._generate_suggestions(code)
        }

    def _find_similar_patterns(self, code: str) -> List[Dict[str, Any]]:
        """查找相似代码模式"""
        results = search_similar_functions(self.storage, self.vectorizer, code)
        return [
            {
                'function': r['info'].name,
                'similarity': 1/(1+r['distance']),
                'file': r['info'].file_path if hasattr(r['info'], 'file_path') else 'unknown'
            }
            for r in results
        ]

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化搜索结果"""
        formatted = []
        for result in results:
            func_info = result['info']
            formatted.append(
                f"Function: {func_info.name}\n"
                f"File: {getattr(func_info, 'file_path', 'unknown')}\n"
                f"Description: {func_info.docstring if func_info.docstring else 'No description'}\n"
                f"Similarity: {1/(1+result['distance']):.3f}\n"
            )
        return "\n".join(formatted)

    def generate_response(self, user_input: str) -> str:
        """增强的响应生成"""
        try:
            # 缓存检查
            cache_key = hash(user_input)
            if cache_key in self.project_context.search_cache:
                context = self.project_context.search_cache[cache_key]
            else:
                # 智能上下文构建
                if self._is_project_overview_query(user_input):
                    context = self._get_project_overview()
                else:
                    context = self._get_relevant_context(user_input)
                self.project_context.search_cache[cache_key] = context

            # 构建优化的prompt
            prompt = self._build_prompt(user_input, context)
            
            # 生成参数优化
            gen_kwargs = self._get_optimized_generation_params()
            
            # 响应生成
            response = self._generate_with_fallback(prompt, gen_kwargs)
            
            # 关键词处理
            response = self._process_keywords(response)
            
            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"抱歉，生成回复时出现错误：{str(e)}"

    def _is_project_overview_query(self, query: str) -> bool:
        """检查是否是项目概览查询"""
        overview_keywords = ['项目', '概述', '简介', '是什么', '干什么']
        return any(keyword in query for keyword in overview_keywords)

    def analyze_project(self, project_path: str) -> bool:
        """分析项目代码库"""
        try:
            print("\nStarting project analysis...")
            print("This may take a while for large repositories.")
            
            print("\nStep 1: Preparing source code...")
            prepared_path = prepare_source_path(project_path)
            
            print("\nStep 2: Analyzing codebase structure...")
            self.logger.info(f"Starting project analysis: {prepared_path}")
            self.storage, codebase_info = analyze_codebase(prepared_path)
            
            print("\nStep 3: Initializing vectorizer...")
            self.vectorizer = init_vectorizer()
            
            print("\nStep 4: Setting up project context...")
            self.project_context = ProjectContext(codebase_info=codebase_info)
            
            # 清理搜索缓存（新增）
            if hasattr(self.project_context, 'search_cache'):
                self.project_context.search_cache.clear()
            
            print("\nAnalysis completed successfully!")
            print(f"Found {codebase_info.file_count} Python files")
            print(f"Analyzed {codebase_info.function_count} functions")
            print(f"Detected {codebase_info.class_count} classes")
            
            # 添加分析完成的时间戳到上下文
            timestamp = datetime.now().isoformat()
            self.project_context.conversation_history.append({
                'system': 'analysis_completed',
                'timestamp': timestamp,
                'stats': {
                    'files': codebase_info.file_count,
                    'functions': codebase_info.function_count,
                    'classes': codebase_info.class_count
                }
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Project analysis failed: {e}")
            print(f"\nError during project analysis: {str(e)}")
            print("Please check the logs for more details.")
            return False

    def _get_project_overview(self) -> str:
        """获取项目概览信息
        
        返回一个格式化的字符串，包含项目的主要统计信息和重要组件的概述
        """
        codebase = self.project_context.codebase_info
        
        # 基础统计信息
        overview = [
            "项目概览:",
            f"- 总文件数: {codebase.file_count}",
            f"- 总函数数: {codebase.function_count}",
            f"- 总类数: {codebase.class_count}"
        ]
        
        # 获取主要模块信息
        if hasattr(codebase, 'modules') and codebase.modules:
            overview.append("\n主要模块:")
            for module in codebase.modules[:5]:  # 只显示前5个主要模块
                module_name = getattr(module, 'name', 'Unknown')
                module_desc = getattr(module, 'description', '无描述')
                overview.append(f"- {module_name}: {module_desc}")
        
        # 获取核心类和函数
        try:
            # 使用向量搜索找到最相关的函数
            core_functions = search_similar_functions(
                self.storage,
                self.vectorizer,
                "main core important",
                k=3
            )
            
            if core_functions:
                overview.append("\n核心功能:")
                for func in core_functions:
                    func_info = func['info']
                    func_name = getattr(func_info, 'name', 'Unknown')
                    func_path = getattr(func_info, 'file_path', 'Unknown')
                    overview.append(f"- {func_name} (位于 {func_path})")
        except Exception as e:
            self.logger.warning(f"获取核心功能时发生错误: {e}")
        
        # 获取依赖信息
        if hasattr(codebase, 'dependencies'):
            overview.append("\n主要依赖:")
            for dep in getattr(codebase, 'dependencies', [])[:5]:
                overview.append(f"- {dep}")
        
        # 代码复杂度评估
        if hasattr(codebase, 'complexity_metrics'):
            metrics = getattr(codebase, 'complexity_metrics', {})
            if metrics:
                overview.append("\n代码复杂度指标:")
                for metric, value in metrics.items():
                    overview.append(f"- {metric}: {value}")
        
        # 最近的更新信息
        recent_changes = self.project_context.conversation_history[-5:]
        if recent_changes:
            overview.append("\n最近的分析记录:")
            for change in recent_changes:
                if 'system' in change and change['system'] == 'analysis_completed':
                    overview.append(f"- 完成于: {change['timestamp']}")
                    if 'stats' in change:
                        stats = change['stats']
                        overview.append(f"  文件: {stats['files']}, 函数: {stats['functions']}, 类: {stats['classes']}")
        
        return "\n".join(overview)
    
    def load_existing_analysis(self, storage_path: str, info_path: str) -> bool:
        """加载已有的分析结果"""
        try:
            self.storage = load_storage(storage_path)
            self.vectorizer = init_vectorizer()
            with open(info_path, 'r') as f:
                info_dict = json.load(f)
            self.project_context = ProjectContext(
                codebase_info=CodebaseInfo(**info_dict)
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load existing analysis: {e}")
            return False
    
    def _get_relevant_context(self, query: str) -> str:
        """获取相关上下文"""
        results = search_similar_functions(self.storage, self.vectorizer, query)
        return self._format_search_results(results)

    def _build_prompt(self, user_input: str, context: str) -> str:
        """构建优化的prompt，更好地引导模型理解和解释代码"""
        
        # 获取项目的核心信息
        core_info = f"""你是一个专业的代码分析助手。你正在分析一个项目，以下是项目的关键信息：
    
    基础统计:
    - 文件数: {self.project_context.codebase_info.file_count}
    - 函数数: {self.project_context.codebase_info.function_count}
    - 类数: {self.project_context.codebase_info.class_count}
    
    你的任务是：
    1. 理解这个代码库的核心功能和架构
    2. 分析模块间的调用关系和数据流
    3. 总结项目的关键功能和实现原理
    4. 用通俗易懂的语言解释技术细节
    
    你具有以下能力：
    1. 你可以看到所有的源代码和函数定义
    2. 你理解函数之间的调用关系
    3. 你能分析代码的结构和设计模式
    4. 你可以推理出代码的目的和实现原理
    
    相关上下文信息:
    {context}
    
    用户问题: {user_input}
    
    请基于以上信息，结合你对代码库的全面理解，给出详细的解答。你的回答应该：
    1. 直接切入用户问题的核心
    2. 解释相关的代码实现原理
    3. 分析关键的函数调用流程
    4. 用清晰的语言总结技术细节
    
    如果需要补充信息，你可以：
    - 使用 [SEARCH] 搜索相关代码
    - 使用 [RAG] 检索增强生成
    - 使用 [CODE] 分析特定代码片段
    """
        return core_info

    def _get_optimized_generation_params(self) -> Dict[str, Any]:
        """获取优化的生成参数"""
        return {
            "max_length": 2048,
            "min_length": 50,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": True,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }

    def _generate_with_fallback(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """带有fallback的生成"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
        except RuntimeError as e:
            if "inf" in str(e) or "nan" in str(e):
                # Fallback到更保守的参数
                gen_kwargs.update({
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "do_sample": False
                })
                outputs = self.model.generate(**inputs, **gen_kwargs)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.replace(prompt, "").strip()
            raise

    def chat(self):
        """优化的交互接口"""
        welcome_message = """
欢迎使用AI代码助手！我已经分析完成项目结构，可以帮助你：
1. 理解代码结构和功能
2. 搜索相似代码片段
3. 分析代码质量
4. 提供改进建议

你可以直接提问，或使用以下关键词：
- [SEARCH] 搜索相关代码
- [RAG] 使用上下文增强回答
- [CODE] 分析代码片段
- [FOCUS] 切换关注点

输入 'exit' 或 'quit' 结束对话。
"""
        print(welcome_message)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nAI Assistant: 感谢使用！再见！")
                    break
                    
                response = self.generate_response(user_input)
                print(f"\nAI Assistant: {response}")
                
                # 保存对话记录
                self.project_context.conversation_history.append({
                    'user': user_input,
                    'assistant': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\nAI Assistant: 检测到中断信号，是否要退出？(y/n)")
                if input().lower() == 'y':
                    break
            except Exception as e:
                self.logger.error(f"Error in chat loop: {e}")
                print(f"\nAI Assistant: 抱歉，发生了错误：{str(e)}")
                print("是否继续？(y/n)")
                if input().lower() != 'y':
                    break
    def _get_optimized_generation_params(self) -> Dict[str, Any]:
        """获取优化的生成参数"""
        return {
            "max_length": 2048,
            "min_length": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,  # 启用采样
            "num_beams": 4,     # 增加beam数量
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }
    
    def _process_keywords(self, response: str) -> str:
        """处理响应中的关键词并执行相应操作
        
        Args:
            response: 模型生成的原始响应
        
        Returns:
            处理后的响应文本
        """
        processed_response = response
        
        for pattern, handler in self.keyword_handlers.items():
            matches = re.finditer(pattern + r'\s*([^[]*?)(?=\[|$)', response)
            for match in matches:
                query = match.group(1).strip()
                if query:
                    try:
                        result = handler(query)
                        if result:
                            # 根据不同的操作类型格式化结果
                            if result['action'] == 'search':
                                formatted_result = "\n搜索结果:\n" + self._format_search_results(result['results'])
                            elif result['action'] == 'rag':
                                formatted_result = "\n相关上下文:\n" + result['context']
                            elif result['action'] == 'code_analysis':
                                formatted_result = "\n代码分析:\n" + str(result['analysis'])
                            elif result['action'] == 'focus_change':
                                formatted_result = f"\n已切换关注点到: {result['new_focus']}"
                            else:
                                formatted_result = str(result)
                                
                            # 替换原始关键词和查询为处理结果
                            original = match.group(0)
                            processed_response = processed_response.replace(original, formatted_result)
                    except Exception as e:
                        self.logger.error(f"处理关键词 {pattern} 时出错: {e}")
                        error_message = f"\n处理 {pattern} 时出现错误: {str(e)}"
                        processed_response = processed_response.replace(match.group(0), error_message)
        
        return processed_response

    def _estimate_complexity(self, code: str) -> Dict[str, Any]:
        """估算代码复杂度
        
        Args:
            code: 要分析的代码字符串
        
        Returns:
            包含各种复杂度指标的字典
        """
        metrics = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'lines_of_code': 0,
            'function_count': 0,
            'class_count': 0
        }
        
        try:
            # 计算代码行数
            lines = code.strip().split('\n')
            metrics['lines_of_code'] = len([line for line in lines if line.strip()])
            
            # 估算圈复杂度
            control_structures = ['if', 'for', 'while', 'and', 'or', 'except', 'with']
            for line in lines:
                line = line.strip()
                if any(s + ' ' in line for s in control_structures):
                    metrics['cyclomatic_complexity'] += 1
            
            # 计算函数和类的数量
            metrics['function_count'] = len(re.findall(r'\bdef\s+\w+\s*\(', code))
            metrics['class_count'] = len(re.findall(r'\bclass\s+\w+\s*[:\(]', code))
            
            # 估算认知复杂度
            indent_level = 0
            for line in lines:
                indent = len(line) - len(line.lstrip())
                new_level = indent // 4  # 假设使用4个空格缩进
                if new_level > indent_level:
                    metrics['cognitive_complexity'] += new_level - indent_level
                indent_level = new_level
                
        except Exception as e:
            self.logger.warning(f"复杂度估算出错: {e}")
        
        return metrics
    
    def _generate_suggestions(self, code: str) -> List[str]:
        """基于代码分析生成改进建议
        
        Args:
            code: 要分析的代码字符串
        
        Returns:
            改进建议列表
        """
        suggestions = []
        
        try:
            # 检查行长度
            max_line_length = 79  # PEP 8建议
            for i, line in enumerate(code.split('\n'), 1):
                if len(line.strip()) > max_line_length:
                    suggestions.append(f"第{i}行超过{max_line_length}个字符，建议拆分")
            
            # 检查函数长度
            if code.count('\n') > 50:  # 假设50行为合理长度
                suggestions.append("代码较长，考虑拆分为多个小函数")
            
            # 检查复杂度
            complexity = self._estimate_complexity(code)
            if complexity['cyclomatic_complexity'] > 10:
                suggestions.append("圈复杂度较高，建议简化控制流程")
            if complexity['cognitive_complexity'] > 15:
                suggestions.append("认知复杂度较高，建议重构以提高可读性")
                
            # 代码风格检查
            style_patterns = [
                (r'\s+$', "存在行尾空格"),
                (r'\t', "使用了制表符，建议使用空格"),
                (r'^\s*except\s*:', "except语句过于宽泛，建议指定具体异常类型"),
                (r'import \*', "不建议使用通配符导入"),
            ]
            
            for pattern, message in style_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    suggestions.append(message)
                    
        except Exception as e:
            self.logger.warning(f"生成建议时出错: {e}")
            suggestions.append(f"分析过程出现错误: {str(e)}")
        
        return suggestions if suggestions else ["代码结构良好，暂无具体改进建议"]

def main():
    try:
        ai = CodeUnderstandingAI()
        
        # 检查现有分析
        storage_dir = Path("storage")
        if storage_dir.exists():
            storages = list(storage_dir.glob("*_faiss"))
            if storages:
                print("\n发现已有的代码分析:")
                for i, storage_path in enumerate(storages, 1):
                    info_path = str(storage_path).replace('_faiss', '_info.json')
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    print(f"{i}. {storage_path.name}")
                    print(f"   - 文件数: {info['file_count']}")
                    print(f"   - 函数数: {info['function_count']}")
                    print(f"   - 类数: {info['class_count']}")
                
                choice = input("\n使用现有分析？(序号/n): ")
                if choice.lower() != 'n':
                    try:
                        idx = int(choice) - 1
                        storage_path = storages[idx]
                        info_path = str(storage_path).replace('_faiss', '_info.json')
                        if ai.load_existing_analysis(str(storage_path), info_path):
                            ai.chat()
                            return
                    except (ValueError, IndexError) as e:
                        print(f"\n无效的选择: {e}")
                        print("将进行新的分析...")
                        
        # 如果没有现有分析或选择重新分析
        while True:
            project_path = input("\n请输入项目路径或git仓库URL: ").strip()
            if project_path:
                break
            print("路径不能为空，请重新输入")
            
        if ai.analyze_project(project_path):
            # 保存分析完成的时间戳
            timestamp = datetime.now().isoformat()
            ai.project_context.conversation_history.append({
                'system': 'analysis_completed',
                'timestamp': timestamp
            })
            ai.chat()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        logging.error(f"程序发生错误: {e}", exc_info=True)
        print(f"\n程序发生错误: {e}\n请查看日志文件获取详细信息。")
        
        # 如果没有现有分析或选择重新分析
        while True:
            project_path = input("\n请输入项目路径或git仓库URL: ").strip()
            if project_path:
                break
            print("路径不能为空，请重新输入")
            
        if ai.analyze_project(project_path):
            # 保存分析完成的时间戳
            timestamp = datetime.now().isoformat()
            ai.project_context.conversation_history.append({
                'system': 'analysis_completed',
                'timestamp': timestamp
            })
            ai.chat()

if __name__ == "__main__":
    main()