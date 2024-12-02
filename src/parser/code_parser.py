"""
File: src/parser/code_parser.py
Description: Code parser using Tree-sitter for extracting function information and call relationships.
"""

from tree_sitter import Language, Parser
import tree_sitter_python
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class FunctionInfo:
    """Contains extracted function information."""
    name: str
    code: str
    docstring: Optional[str]
    params: List[str]
    return_type: Optional[str]
    start_line: int
    end_line: int

class CodeParser:
    def __init__(self):
        """Initialize the code parser."""
        try:
            self.parser = Parser()
            # 直接使用预编译的Python语言包
            PY_LANGUAGE = Language(tree_sitter_python.language())
            self.parser.set_language(PY_LANGUAGE)
            
            # 创建用于查找函数定义和调用的查询
            self.query = PY_LANGUAGE.query("""
                (function_definition
                  name: (identifier) @function.def
                  parameters: (parameters) @function.params
                  body: (block) @function.body) @function.whole
                  
                (call
                  function: (identifier) @function.call)
            """)
            
        except Exception as e:
            logging.error(f"Failed to initialize parser: {e}")
            raise
            
    def extract_function_info(
        self, 
        code_str: str, 
        file_path: str
    ) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        """
        Extract function information and call relationships from code.
        
        Args:
            code_str: Source code string
            file_path: Path to source file
            
        Returns:
            Tuple containing:
            - Dictionary of function IDs to function information
            - Dictionary of caller IDs to lists of callee IDs
        """
        try:
            # 将代码转换为bytes
            code_bytes = bytes(code_str, 'utf8')
            tree = self.parser.parse(code_bytes)
            
            functions = {}
            call_graph = {}
            
            # 使用查询来查找所有函数定义和调用
            captures = self.query.captures(tree.root_node)
            current_function = None
            
            for node, capture_name in captures:
                if capture_name == "function.whole":
                    # 提取函数信息
                    func_node = node
                    name_node = func_node.child_by_field_name('name')
                    params_node = func_node.child_by_field_name('parameters')
                    body_node = func_node.child_by_field_name('body')
                    
                    # 生成函数ID
                    func_name = code_str[name_node.start_byte:name_node.end_byte]
                    func_id = f"{file_path}::{func_name}"
                    
                    # 提取参数
                    params = []
                    if params_node:
                        for param in params_node.named_children:
                            params.append(code_str[param.start_byte:param.end_byte])
                    
                    # 提取docstring
                    docstring = None
                    if body_node and body_node.named_children:
                        first_child = body_node.named_children[0]
                        if first_child.type == 'string':
                            docstring = code_str[first_child.start_byte:first_child.end_byte]
                    
                    functions[func_id] = FunctionInfo(
                        name=func_name,
                        code=code_str[node.start_byte:node.end_byte],
                        docstring=docstring,
                        params=params,
                        return_type=None,  # Could be extended to extract return type
                        start_line=node.start_point[0],
                        end_line=node.end_point[0]
                    )
                    
                    current_function = func_id
                    call_graph[func_id] = []
                    
                elif capture_name == "function.call" and current_function:
                    # 记录函数调用
                    called_func = code_str[node.start_byte:node.end_byte]
                    called_func_id = f"{file_path}::{called_func}"
                    if current_function in call_graph:
                        call_graph[current_function].append(called_func_id)
            
            return functions, call_graph
            
        except Exception as e:
            logging.error(f"Failed to parse code: {e}")
            raise
            
    def _process_node(
        self,
        cursor,
        code_str: str,
        file_path: str,
        functions: Dict[str, FunctionInfo],
        call_graph: Dict[str, List[str]]
    ) -> None:
        """Process an AST node recursively."""
        node = cursor.node
        
        if node.type == 'function_definition':
            # Extract function information
            func_info = self._extract_function_info(node, code_str)
            
            # Generate unique ID
            signature = FunctionSignature(
                name=func_info.name,
                params=func_info.params,
                return_type=func_info.return_type
            )
            func_id = IdGenerator.generate_function_id(file_path, signature)
            
            # Store function info
            functions[func_id] = func_info
            
            # Analyze function calls
            call_graph[func_id] = self._analyze_calls(node)
        
        # Continue traversal
        if cursor.goto_first_child():
            self._process_node(cursor, code_str, file_path, functions, call_graph)
            while cursor.goto_next_sibling():
                self._process_node(cursor, code_str, file_path, functions, call_graph)
            cursor.goto_parent()
