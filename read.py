from src.storage.code_storage import CodeStorage
from src.vectorizer.code_vectorizer import CodeVectorizer, VectorizeConfig
from pathlib import Path
import json

def load_storage(storage_path: str):
    """加载存储的向量数据库"""
    storage = CodeStorage(vector_dim=1024)
    storage.load(storage_path)
    return storage

def init_vectorizer():
    """初始化向量化模型"""
    return CodeVectorizer(VectorizeConfig(
        model_name='/root/autodl-tmp/rag/multilingual-e5-large-instruct',
        max_length=512
    ))

def search_similar_functions(storage: CodeStorage, vectorizer: CodeVectorizer, query_code: str, k: int = 5):
    """搜索语义相似的函数"""
    # 构造一个临时的FunctionInfo对象用于生成查询向量
    query_info = {
        'name': 'query',
        'code': query_code,
        'docstring': None,
        'params': [],
        'return_type': None,
        'start_line': 0,
        'end_line': 0
    }
    query_vector = vectorizer.vectorize(query_info)
    
    # 搜索相似函数
    results = storage.search_similar(query_vector, k)
    return results

def get_function_calls(storage: CodeStorage, function_id: str):
    """获取函数的调用关系"""
    callees = storage.call_graph.get(function_id, [])
    if not callees:
        print(f"Call graph for {function_id}: {storage.call_graph}")  # 调试信息
        return []
        
    caller_info = storage.functions[function_id]
    print(f"\nFunction {caller_info.name} calls:")
    
    called_functions = []
    for callee_id in callees:
        if callee_id in storage.functions:
            called_functions.append(storage.functions[callee_id])
    
    return called_functions

def print_function_info(func_info, indent=""):
    """美化输出函数信息"""
    print("\n" + "="*50)
    print(f"{indent}Function Name: {func_info.name}")
    print(f"{indent}Parameters: {', '.join(func_info.params)}")
    if func_info.docstring:
        print(f"{indent}Docstring: {func_info.docstring}")
    print(f"\n{indent}Code:")
    print(f"{indent}{func_info.code}")
    print("="*50)

def main():
    # 加载storage
    storage_dir = Path("storage")
    
    # 列出所有可用的存储
    available_storages = list(storage_dir.glob("*"))
    if not available_storages:
        print("No stored codebases found.")
        return
        
    print("Available codebases:")
    for i, storage_path in enumerate(available_storages):
        print(f"{i+1}. {storage_path.name}")
    
    # 选择要查询的codebase
    choice = int(input("\nSelect a codebase (enter number): ")) - 1
    storage_path = available_storages[choice]
    
    # 加载storage和vectorizer
    storage = load_storage(str(storage_path))
    vectorizer = init_vectorizer()
    
    while True:
        print("\nQuery options:")
        print("1. Search similar functions")
        print("2. Show function calls")
        print("3. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == "1":
            query_code = input("\nEnter the function code or description to search:\n")
            results = search_similar_functions(storage, vectorizer, query_code)
            
            print("\nFound similar functions:")
            for i, result in enumerate(results):
                print(f"\n{i+1}. Similarity score: {1/(1+result['distance']):.3f}")
                print_function_info(result['info'])
                
        elif choice == "2":
            # 先展示所有可用的函数
            print("\nAvailable functions:")
            for i, (func_id, func_info) in enumerate(storage.functions.items()):
                print(f"{i+1}. {func_info.name}")
                
            func_idx = int(input("\nSelect a function (enter number): ")) - 1
            func_id = list(storage.functions.keys())[func_idx]
            
            called_functions = get_function_calls(storage, func_id)
            if called_functions:
                print("\nCalled functions:")
                for func in called_functions:
                    print_function_info(func)
            else:
                print("\nThis function doesn't call any other functions.")
                
        elif choice == "3":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()