"""
File: CodeWeaver/parser/code_parser.py
Description: Unified interface for all language-specific parsers.
"""

# 重新导出FunctionInfo和ImportInfo类
from .code_parser_py import FunctionInfo, ImportInfo, CodeParser as PyCodeParser
from .code_parser_cpp import CodeParserCpp
from .code_parser_java import CodeParserJava

# 工厂函数，根据文件类型返回合适的解析器
def get_parser_for_file(file_path: str):
    """
    Returns the appropriate parser based on file extension.
    
    Args:
        file_path: Path to the source file
        
    Returns:
        An instance of the appropriate code parser
        
    Raises:
        ValueError: If the file type is not supported
    """
    if file_path.endswith('.py'):
        return PyCodeParser()
    elif file_path.endswith(('.cpp', '.cc', '.h', '.hpp', '.c')):
        return CodeParserCpp()
    elif file_path.endswith('.java'):
        return CodeParserJava()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# 统一的接口类，可以处理所有语言
class CodeParser:
    """
    Unified parser interface that delegates to language-specific parsers.
    """
    
    def __init__(self):
        """Initialize the unified code parser."""
        self.py_parser = PyCodeParser()
        self.cpp_parser = CodeParserCpp()
        self.java_parser = CodeParserJava()
    
    def extract_function_info(self, code_str: str, file_path: str):
        """
        Extract function information from code.
        
        Args:
            code_str: Source code string
            file_path: Path to the source file
            
        Returns:
            A tuple of (functions_dict, call_graph)
        """
        parser = get_parser_for_file(file_path)
        return parser.extract_function_info(code_str, file_path)