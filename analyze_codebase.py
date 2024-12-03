from src.parser.code_parser import CodeParser
from src.vectorizer.code_vectorizer import CodeVectorizer, VectorizeConfig
from src.storage.code_storage import CodeStorage
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodebaseInfo:
    """存储代码库分析结果的数据类"""
    file_count: int
    function_count: int
    class_count: int
    files_processed: List[str]

def get_storage_path(code_path: str) -> str:
    """为源代码文件或目录生成对应的存储路径"""
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    path = Path(code_path)
    if path.is_file():
        storage_name = f"{path.stem}_{hash(path.absolute())}"
    else:
        storage_name = f"{path.name}_{hash(str(path.absolute()))}"
    
    return str(storage_dir / storage_name)

def collect_python_files(base_path: str) -> List[Tuple[str, str]]:
    """收集Python文件，返回(文件名, 绝对路径)的列表"""
    python_files = []
    base_path = Path(base_path).resolve()
    
    if base_path.is_file():
        if base_path.suffix == '.py':
            python_files.append((base_path.name, str(base_path)))
    else:
        for py_file in base_path.rglob('*.py'):
            if not any(part.startswith(('.', '__', 'venv', 'env')) 
                      for part in py_file.parts):
                python_files.append((py_file.name, str(py_file)))
    
    return python_files

def analyze_codebase(code_path: str) -> Tuple[CodeStorage, CodebaseInfo]:
    """分析代码库，支持单个文件或整个目录"""
    storage_path = get_storage_path(code_path)
    
    # 初始化组件
    parser = CodeParser()
    vectorizer = CodeVectorizer(VectorizeConfig(
        model_name='/root/autodl-tmp/rag/multilingual-e5-large-instruct',
        max_length=512
    ))
    storage = CodeStorage(vector_dim=1024)
    
    # 如果存储文件已存在，尝试加载
    if os.path.exists(storage_path):
        logger.info(f"Loading existing storage from {storage_path}")
        try:
            storage.load(storage_path)
            codebase_info = CodebaseInfo(
                file_count=0,
                function_count=storage.index.ntotal,
                class_count=0,
                files_processed=[]
            )
            return storage, codebase_info
        except Exception as e:
            logger.warning(f"Failed to load storage: {e}")
            logger.info("Creating new storage...")
    
    # 收集需要处理的Python文件，同时保持相对路径信息
    python_files = collect_python_files(code_path)
    if not python_files:
        raise ValueError(f"No Python files found in {code_path}")
    
    # 用于统计的变量
    total_functions = 0
    total_classes = 0
    processed_files = []
    
    # 处理每个Python文件
    base_path = Path(code_path).resolve()
    for rel_path, abs_path in python_files:
        logger.info(f"Processing file: {abs_path}")
        # try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print(len(code))
        
        # 使用相对路径作为文件标识符
        functions, call_graph = parser.extract_function_info(code, abs_path)
        
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
        processed_files.append(rel_path)
        logger.info(f"Successfully processed {len(functions)} functions in {rel_path}")
            
        # except Exception as e:
        #     logger.error(f"Error processing {rel_path}: {e}")
        #     continue
    
    # 保存storage
    logger.info(f"Saving storage to {storage_path}")
    storage.save(storage_path)
    
    # 创建分析结果
    codebase_info = CodebaseInfo(
        file_count=len(processed_files),
        function_count=total_functions,
        class_count=total_classes,
        files_processed=processed_files
    )
    
    return storage, codebase_info

def main(code_path: str):
    """主函数"""
    try:
        storage, info = analyze_codebase(code_path)
        logger.info("Analysis completed successfully!")
        logger.info(f"Files processed: {info.file_count}")
        logger.info(f"Total functions: {info.function_count}")
        logger.info(f"Total classes: {info.class_count}")
        logger.info(f"Total functions indexed: {storage.index.ntotal}")
        
        return storage, info
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_codebase.py <path_to_code>")
        sys.exit(1)
    
    code_path = sys.argv[1]
    storage, info = main(code_path)
    print("\nAnalysis Summary:")
    print(f"{'='*50}")
    for file in info.files_processed:
        print(f"✓ {file}")