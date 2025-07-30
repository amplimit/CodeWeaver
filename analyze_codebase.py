"""
This file implements the core functionality for analyzing codebases. It can handle both local directories and Git repositories, extracting function definitions, class structures, and generating call graphs. The analyzer calculates SHA256 hashes for codebase versioning and supports incremental analysis by caching results. Key features include:

- Code file collection and filtering (Python, C++, Java)
- Code structure extraction (functions, classes)
- SHA256-based caching system
- Support for both local and remote Git repositories
- FAISS vector storage integration
"""
from CodeWeaver.parser.code_parser import CodeParser, get_parser_for_file
from CodeWeaver.vectorizer.code_vectorizer import CodeVectorizer, VectorizeConfig
from CodeWeaver.storage.code_storage import CodeStorage
import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging
import uuid
import argparse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodebaseInfo:
    """存储代码库分析结果的数据类"""
    uuid: str
    sha256: str
    file_count: int
    function_count: int
    class_count: int
    files_processed: List[str]
    
def calculate_sha256(files: List[Tuple[str, str]]) -> str:
    """计算所有代码文件的组合SHA256值"""
    sha256_hash = hashlib.sha256()
    
    for _, abs_path in sorted(files):  # 排序确保顺序一致性
        try:
            with open(abs_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
        except Exception as e:
            logger.error(f"Error reading file {abs_path} for SHA256 calculation: {e}")
            continue
            
    return sha256_hash.hexdigest()

def get_storage_path(code_uuid: str, use_faiss: bool = True) -> str:
    """为源代码生成对应的存储路径"""
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    if use_faiss:
        return str(storage_dir / f"{code_uuid}_faiss")
    else:
        return str(storage_dir / f"{code_uuid}_info.json")

def collect_code_files(base_path: str) -> List[Tuple[str, str]]:
    """收集所有支持的代码文件，返回(文件名, 绝对路径)的列表"""
    code_files = []
    base_path = Path(base_path).resolve()
    
    if base_path.is_file():
        ext = base_path.suffix
        if ext in ['.py', '.cpp', '.cc', '.h', '.hpp', '.c', '.java']:
            code_files.append((base_path.name, str(base_path)))
    else:
        # 搜索Python文件
        for py_file in base_path.rglob('*.py'):
            # 修复过滤逻辑：只排除以.开头的隐藏目录，__pycache__目录，和虚拟环境目录
            # 但保留__init__.py等重要文件
            excluded_dirs = {'.git', '.vscode', '.idea', '__pycache__', 'venv', 'env', '.env'}
            if not any(part in excluded_dirs or (part.startswith('.') and part != py_file.name) 
                      for part in py_file.parts):
                code_files.append((py_file.name, str(py_file)))
        
        # 搜索C++文件
        for ext in ['.cpp', '.cc', '.h', '.hpp', '.c']:
            for cpp_file in base_path.rglob(f'*{ext}'):
                excluded_dirs = {'.git', '.vscode', '.idea', '__pycache__', 'build', 'out', 'target'}
                if not any(part in excluded_dirs or (part.startswith('.') and part != cpp_file.name) 
                          for part in cpp_file.parts):
                    code_files.append((cpp_file.name, str(cpp_file)))
        
        # 搜索Java文件
        for java_file in base_path.rglob('*.java'):
            excluded_dirs = {'.git', '.vscode', '.idea', '__pycache__', 'target', 'build', 'out'}
            if not any(part in excluded_dirs or (part.startswith('.') and part != java_file.name) 
                      for part in java_file.parts):
                code_files.append((java_file.name, str(java_file)))
    
    return code_files

def analyze_codebase(code_source: str, use_faiss: bool = True) -> Tuple[Optional[CodeStorage], CodebaseInfo]:
    """分析代码库，支持单个文件或整个目录"""
    # 生成UUID
    code_uuid = str(uuid.uuid4())
    
    # 收集代码文件
    code_files = collect_code_files(code_source)
    if not code_files:
        raise ValueError(f"No code files found in {code_source}")
    
    # 计算SHA256
    code_sha256 = calculate_sha256(code_files)
    
    # 检查是否存在相同的SHA256
    storage_dir = Path("storage")
    if storage_dir.exists():
        for info_file in storage_dir.glob("*_info.json"):
            try:
                with open(info_file, 'r') as f:
                    existing_info = json.load(f)
                if existing_info.get('sha256') == code_sha256:
                    logger.info(f"Found existing analysis with same SHA256: {code_sha256}")
                    if use_faiss:
                        # 加载对应的FAISS存储
                        storage = CodeStorage(vector_dim=1024)
                        faiss_path = str(info_file).replace('_info.json', '_faiss')
                        if os.path.exists(faiss_path):
                            storage.load(faiss_path)
                            return storage, CodebaseInfo(**existing_info)
                    return None, CodebaseInfo(**existing_info)
            except Exception as e:
                logger.warning(f"Error reading existing info file {info_file}: {e}")
                continue
    
    storage = None
    if use_faiss:
        # 初始化组件
        unified_parser = CodeParser()  # 使用统一接口
        vectorizer = CodeVectorizer(VectorizeConfig(
            model_name='intfloat/multilingual-e5-large-instruct',
            max_length=512
        ))
        storage = CodeStorage(vector_dim=1024)
    
    # 用于统计的变量
    total_functions = 0
    total_classes = 0
    processed_files = []
    
    # 处理每个代码文件
    for rel_path, abs_path in code_files:
        logger.info(f"Processing file: {abs_path}")
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if use_faiss:
                try:
                    # 使用适当的语言解析器
                    language_parser = get_parser_for_file(abs_path)
                    functions, call_graph = language_parser.extract_function_info(code, abs_path)
                    
                    # 记录找到的类
                    classes_in_file = {
                        func_info.class_name 
                        for func_info in functions.values() 
                        if func_info.class_name is not None
                    }
                    total_classes += len(classes_in_file)
                    
                    # 处理每个函数
                    for func_id, func_info in functions.items():
                        vector = vectorizer.vectorize(func_info)
                        storage.add_function(func_id, vector, func_info, call_graph.get(func_id, []))
                    
                    total_functions += len(functions)
                except ValueError as e:
                    logger.warning(f"Skipping unsupported file {abs_path}: {e}")
                    continue
            else:
                # 简单统计，不进行向量化 (针对不同语言的基本统计)
                if abs_path.endswith('.py'):
                    total_functions += code.count('def ')
                    total_classes += code.count('class ')
                elif abs_path.endswith(('.cpp', '.cc', '.h', '.hpp', '.c')):
                    total_functions += code.count('void ') + code.count('int ') + code.count('bool ')
                    total_classes += code.count('class ') + code.count('struct ')
                elif abs_path.endswith('.java'):
                    total_functions += code.count('void ') + code.count('public ') + code.count('private ')
                    total_classes += code.count('class ') + code.count('interface ')
            
            processed_files.append(rel_path)
            logger.info(f"Successfully processed {rel_path}")
            
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            continue
    
    # 创建分析结果
    codebase_info = CodebaseInfo(
        uuid=code_uuid,
        sha256=code_sha256,
        file_count=len(processed_files),
        function_count=total_functions,
        class_count=total_classes,
        files_processed=processed_files
    )
    
    # 保存分析结果
    storage_path = get_storage_path(code_uuid, use_faiss)
    if use_faiss:
        logger.info(f"Saving FAISS storage to {storage_path}")
        storage.save(storage_path)
    
    # 保存信息文件
    info_path = get_storage_path(code_uuid, False)
    with open(info_path, 'w') as f:
        json.dump(vars(codebase_info), f, indent=2)
    
    return storage, codebase_info

def prepare_source_path(code_source: str) -> str:
    """准备源代码路径，支持远程代码克隆"""
    if code_source.startswith(("http://", "https://", "git@")):
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(exist_ok=True)
        clone_path = tmp_dir / str(uuid.uuid4())
        os.system(f"git clone {code_source} {clone_path}")
        return str(clone_path)
    return code_source

def main(code_source):
    """主函数"""    
    no_faiss=False
    try:
        # 准备源代码路径
        code_path = prepare_source_path(code_source)
        
        # 分析代码库
        storage, info = analyze_codebase(code_path, not no_faiss)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"UUID: {info.uuid}")
        logger.info(f"SHA256: {info.sha256}")
        logger.info(f"Files processed: {info.file_count}")
        logger.info(f"Total functions: {info.function_count}")
        logger.info(f"Total classes: {info.class_count}")
        if storage:
            logger.info(f"Total functions indexed: {storage.index.ntotal}")
        
        return storage, info
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        # 清理临时目录
        if code_source.startswith(("http://", "https://", "git@")):
            os.system(f"rm -rf {code_path}")

if __name__ == "__main__":
    code_source = "https://github.com/Significant-Gravitas/AutoGPT.git"
    storage, info = main(code_source=code_source)
    print("\nAnalysis Summary:")
    print(f"{'='*50}")
    for file in info.files_processed:
        print(f"✓ {file}")