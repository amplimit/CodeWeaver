"""
An AI-powered assistant that combines the analysis and query capabilities with large language models (LLMs) for enhanced code understanding. The assistant can:

- Process natural language queries about code
- Provide context-aware responses using RAG (Retrieval-Augmented Generation)
- Maintain conversation history for coherent interactions
- Automatically focus on relevant code sections
- Support both interactive chat and programmatic interfaces

"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import subprocess
import sys
from datetime import datetime
import uuid

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
    current_focus: Optional[str] = None  # 当前关注的代码部分
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

class CodeUnderstandingAI:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.setup_logging()
        self.initialize_components(model_name)
        self.project_context = None
        self.storage = None
        self.vectorizer = None

    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_assistant.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self, model_name: str):
        """初始化AI模型和必要组件"""
        self.logger.info(f"Initializing AI with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
    def analyze_project(self, project_path: str) -> bool:
        """分析项目代码库"""
        try:
            print("\nStarting project analysis...")
            print("This may take a while for large repositories.")
            
            print("\nStep 1: Preparing source code...")
            # 使用全局定义的prepare_source_path函数
            prepared_path = prepare_source_path(project_path)
            
            print("\nStep 2: Analyzing codebase structure...")
            self.logger.info(f"Starting project analysis: {prepared_path}")
            self.storage, codebase_info = analyze_codebase(prepared_path)
            
            print("\nStep 3: Initializing vectorizer...")
            self.vectorizer = init_vectorizer()
            
            print("\nStep 4: Setting up project context...")
            self.project_context = ProjectContext(codebase_info=codebase_info)
            
            print("\nAnalysis completed successfully!")
            print(f"Found {codebase_info.file_count} Python files")
            print(f"Analyzed {codebase_info.function_count} functions")
            print(f"Detected {codebase_info.class_count} classes")
            
            return True
        except Exception as e:
            self.logger.error(f"Project analysis failed: {e}")
            print(f"\nError during project analysis: {str(e)}")
            print("Please check the logs for more details.")
            return False

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

    def process_control_keywords(self, text: str) -> Dict[str, Any]:
        """处理输出中的控制关键词"""
        actions = {}
        
        # 处理[SEARCH]关键词
        if "[SEARCH]" in text:
            query = text[text.find("[SEARCH]")+8:text.find("[/SEARCH]")]
            results = search_similar_functions(self.storage, self.vectorizer, query)
            actions['search_results'] = results
            
        # 处理[RAG]关键词
        if "[RAG]" in text:
            query = text[text.find("[RAG]")+5:text.find("[/RAG]")]
            # 使用FAISS搜索相关代码片段
            results = search_similar_functions(self.storage, self.vectorizer, query, k=3)
            actions['rag_context'] = results
            
        # 处理[UPDATE]关键词
        if "[UPDATE]" in text:
            update_info = text[text.find("[UPDATE]")+8:text.find("[/UPDATE]")]
            try:
                update_data = json.loads(update_info)
                self.project_context.current_focus = update_data.get('focus')
                actions['update_status'] = 'success'
            except json.JSONDecodeError:
                actions['update_status'] = 'failed'
                
        return actions

    def generate_response(self, user_input: str) -> str:
        """生成回复并处理控制关键词"""
        # 构建提示
        prompt = f"""Based on the current project context:
Files: {self.project_context.codebase_info.file_count}
Functions: {self.project_context.codebase_info.function_count}
Classes: {self.project_context.codebase_info.class_count}
Current focus: {self.project_context.current_focus}

User question: {user_input}

Please provide a relevant response. Use [SEARCH], [RAG], or [UPDATE] keywords if needed.
"""
        # 生成回复
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **input_ids,
                max_length=2048,
                temperature=0.7,
                top_p=0.9
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 处理控制关键词并更新回复
        actions = self.process_control_keywords(response)
        
        # 根据actions调整响应
        final_response = response
        if 'search_results' in actions:
            final_response += "\n\nRelevant code found:"
            for result in actions['search_results']:
                final_response += f"\n- {result['info'].name}"
                
        if 'rag_context' in actions:
            final_response += "\n\nIncorporating context from similar code:"
            for result in actions['rag_context']:
                final_response += f"\n- {result['info'].name}"
                
        return final_response

    def chat(self):
        """交互式聊天接口"""
        print("\nAI Assistant: 我已经准备好了！我已经理解了项目的结构，你可以问我任何关于代码的问题。")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAI Assistant: 再见！")
                break
                
            response = self.generate_response(user_input)
            print(f"\nAI Assistant: {response}")
            
            # 保存对话历史
            self.project_context.conversation_history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })

def main():
    ai = CodeUnderstandingAI()
    
    # 检查是否有现有分析
    storage_dir = Path("storage")
    if storage_dir.exists():
        storages = list(storage_dir.glob("*_faiss"))
        if storages:
            print("\nFound existing analysis:")
            for i, storage_path in enumerate(storages):
                info_path = str(storage_path).replace('_faiss', '_info.json')
                print(f"{i+1}. {storage_path.name}")
                
            choice = input("\nUse existing analysis? (number/n): ")
            if choice.lower() != 'n':
                idx = int(choice) - 1
                storage_path = storages[idx]
                info_path = str(storage_path).replace('_faiss', '_info.json')
                if ai.load_existing_analysis(str(storage_path), info_path):
                    ai.chat()
                    return
    
    # 如果没有现有分析或选择重新分析
    project_path = input("\nEnter project path or git URL: ")
    if ai.analyze_project(project_path):
        ai.chat()

if __name__ == "__main__":
    main()