from src.parser.code_parser import CodeParser
from src.vectorizer.code_vectorizer import CodeVectorizer, VectorizeConfig
from src.storage.code_storage import CodeStorage
import os
from pathlib import Path

def get_storage_path(code_path: str) -> str:
    """为源代码文件生成对应的存储路径"""
    # 获取项目根目录下的storage文件夹
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    # 生成存储文件名（使用源文件的hash以避免文件名冲突）
    code_path = Path(code_path)
    storage_name = f"{code_path.stem}_{hash(code_path.absolute())}"
    
    return str(storage_dir / storage_name)

def analyze_codebase(code_path: str):
    # 获取存储路径
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
        print(f"Loading existing storage from {storage_path}")
        try:
            storage.load(storage_path)
            return storage
        except Exception as e:
            print(f"Failed to load storage: {e}")
            print("Creating new storage...")
    
    # 读取并解析代码
    with open(code_path, 'r') as f:
        code = f.read()
    
    # 提取函数信息
    functions, call_graph = parser.extract_function_info(code, code_path)
    
    # 处理每个函数
    for func_id, func_info in functions.items():
        vector = vectorizer.vectorize(func_info)
        storage.add_function(func_id, vector, func_info, call_graph.get(func_id, []))
    
    # 保存storage
    print(f"Saving storage to {storage_path}")
    storage.save(storage_path)
    
    return storage

if __name__ == "__main__":
    # 测试代码分析
    storage = analyze_codebase("examples/example.py")
    print("Storage created successfully.")
    print(f"Total functions indexed: {storage.index.ntotal}")