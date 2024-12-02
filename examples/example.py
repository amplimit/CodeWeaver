# 完整的使用示例
from src.parser.code_parser import CodeParser
from src.vectorizer.code_vectorizer import CodeVectorizer, VectorizeConfig
from src.storage.code_storage import CodeStorage

def analyze_codebase(code_path: str):
    # 初始化组件
    parser = CodeParser(language_dir='languages')
    vectorizer = CodeVectorizer(VectorizeConfig(
        model_name='intfloat/multilingual-e5-large-instruct',
        max_length=512
    ))
    storage = CodeStorage(vector_dim=512)
    
    # 读取并解析代码
    with open(code_path, 'r') as f:
        code = f.read()
    
    # 提取函数信息
    functions, call_graph = parser.extract_function_info(code, code_path)
    
    # 处理每个函数
    for func_id, func_info in functions.items():
        vector = vectorizer.vectorize(func_info)
        storage.add_function(func_id, vector, func_info, call_graph.get(func_id, []))
    
    return storage

# 使用示例
if __name__ == "__main__":
    # 分析代码库
    storage = analyze_codebase("example.py")
    
    # 查找相似函数
    query_code = "def hello(): print('world')"
    query_info = {
        'code': query_code,
        'name': 'query',
        'params': []
    }
    query_vector = vectorizer.vectorize(query_info)
    similar = storage.search_similar(query_vector, k=3)
    
    # 打印结果
    for result in similar:
        print(f"Found similar function: {result['info']['name']}")
        print(f"Distance: {result['distance']}")
        print(f"Code:\n{result['info']['code']}\n")