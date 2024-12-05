from tree_sitter import Language, Parser
import tree_sitter_cpp
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from code_parser_py import FunctionInfo, ImportInfo

class CodeParserCpp:
    """
    C++ code parser using tree-sitter library to analyze C++ source code.
    Extracts information about functions, methods, imports and their relationships.
    
    This parser can handle:
    - Global functions
    - Class methods (public only)
    - Namespace functions
    - Import statements
    - Function calls
    """

    def __init__(self):
        """
        Initialize the C++ parser with tree-sitter.
        Raises exception if parser initialization fails.
        """
        try:
            self.parser = Parser()
            CPP_LANGUAGE = Language(tree_sitter_cpp.language())
            self.parser.language = CPP_LANGUAGE
        except Exception as e:
            raise Exception(f"Failed to initialize C++ parser: {e}")

    def _traverse_tree(self, tree, source_code: bytes):
        """
        Traverse the AST tree in a depth-first manner.
        
        Args:
            tree: The AST tree to traverse
            source_code: Original source code in bytes
            
        Yields:
            Each node in the AST tree
        """
        cursor = tree.walk()
        visited_children = False
        
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    def _process_nested_namespaces(self, node, source_code: bytes, current_namespace=None):
        """
        递归处理嵌套的命名空间
        
        Args:
            node: 当前AST节点
            source_code: 源代码字节
            current_namespace: 当前累积的命名空间路径
        
        Returns:
            包含命名空间内所有函数的元组(functions, calls)
        """
        functions = {}
        calls = {}
        
        if node.type == "namespace_definition":
            ns_name = node.child_by_field_name("name")
            ns_text = self._get_node_text(ns_name, source_code) if ns_name else None
            
            # 构建完整的命名空间路径
            if ns_text:
                current_namespace = f"{current_namespace}::{ns_text}" if current_namespace else ns_text
                
            body = node.child_by_field_name("body")
            if body:
                # 递归处理命名空间内的所有节点
                for child in body.children:
                    child_funcs, child_calls = self._process_nested_namespaces(child, source_code, current_namespace)
                    functions.update(child_funcs)
                    calls.update(child_calls)
                    
        elif node.type == "function_definition":
            name, info, function_calls = self._analyze_function(node, source_code, None, current_namespace)
            if name and info:
                functions[name] = info
                calls[name] = function_calls
                
        return functions, calls

    def _extract_imports(self, tree, source_code: bytes) -> List[ImportInfo]:
        """
        Extract import statements from C++ code.
        
        Args:
            tree: The AST tree
            source_code: Original source code in bytes
            
        Returns:
            List of ImportInfo objects containing import information
            
        Handles:
            - #include statements
            - using declarations
        """
        imports = []
        for node in self._traverse_tree(tree, source_code):
            if node.type == "preproc_include":
                path_node = node.child_by_field_name("path")
                if path_node:
                    path = self._get_node_text(path_node, source_code).strip('"<>')
                    imports.append(ImportInfo(module=path, names=[], is_from_import=False))
                    
            elif node.type == "using_declaration":
                # 处理 using std::string 这样的语句
                name = self._get_node_text(node, source_code)
                if "::" in name:
                    module, item = name.rsplit("::", 1)
                    imports.append(ImportInfo(module=module, names=[item], is_from_import=True))
                else:
                    imports.append(ImportInfo(module=name, names=[], is_from_import=True))
                    
            elif node.type == "using_namespace_definition":
                # 处理 using namespace xxx 语句
                namespace = self._get_node_text(node.children[-1], source_code)
                imports.append(ImportInfo(module=namespace, names=[], is_from_import=True))
                
            elif node.type == "namespace_alias_definition":
                # 处理命名空间别名
                alias = self._get_node_text(node.child_by_field_name("name"), source_code)
                value = self._get_node_text(node.child_by_field_name("value"), source_code)
                imports.append(ImportInfo(module=value, names=[alias], is_from_import=True))
                
        return imports

    def _get_node_text(self, node, source_code: bytes) -> str:
        """
        Extract text content from a node in the AST.
        
        Args:
            node: The AST node
            source_code: Original source code in bytes
            
        Returns:
            The text content of the node
        """
        try:
            return source_code[node.start_byte:node.end_byte].decode('utf8')
        except Exception as e:
            raise Exception(f"Failed to get node text: {e}")

    def extract_function_info(self, code_str: str, file_path: str) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        """
        提取代码中的所有函数信息
        
        Args:
            code_str: C++源代码字符串
            file_path: 源文件路径
            
        Returns:
            (函数信息字典, 函数调用字典)
        
        Raises:
            Exception: 当解析失败时
        """
        try:
            source_code = code_str.encode('utf8')
            tree = self.parser.parse(source_code)
            if tree.root_node.has_error:
                raise Exception("Failed to parse the code: syntax error detected")
                
            functions = {}
            calls = {}
            
            def process_node(node, current_namespace=None):
                if node.type == "namespace_definition":
                    # 处理命名空间
                    ns_funcs, ns_calls = self._process_nested_namespaces(node, source_code)
                    for name, info in ns_funcs.items():
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                    for name, func_calls in ns_calls.items():
                        full_name = f"{file_path}::{name}"
                        calls[full_name] = list(func_calls)
                        
                elif node.type == "class_specifier":
                    # 处理类
                    class_name = node.child_by_field_name("name")
                    class_name = self._get_node_text(class_name, source_code) if class_name else None
                    
                    if class_name:
                        body = node.child_by_field_name("body")
                        if body:
                            in_public = False
                            for child in body.children:
                                if child.type == "access_specifier":
                                    access_text = self._get_node_text(child, source_code)
                                    in_public = access_text.strip(':') == "public"
                                    continue
                                    
                                if child.type == "function_definition" and (in_public or child.type == "constructor"):
                                    name, info, function_calls = self._analyze_function(
                                        child, source_code, class_name, current_namespace)
                                    if name and info:
                                        full_name = f"{file_path}::{name}"
                                        functions[full_name] = info
                                        calls[full_name] = list(function_calls)
                                        
                elif node.type == "template_declaration":
                    # 处理模板
                    declaration = node.child_by_field_name("declaration")
                    if declaration and declaration.type == "function_definition":
                        name, info, function_calls = self._analyze_function(
                            declaration, source_code, None, current_namespace)
                        if name and info:
                            full_name = f"{file_path}::{name}"
                            functions[full_name] = info
                            calls[full_name] = list(function_calls)
                            
                elif node.type == "function_definition":
                    # 处理普通函数
                    name, info, function_calls = self._analyze_function(
                        node, source_code, None, current_namespace)
                    if name and info:
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                        calls[full_name] = list(function_calls)

            # 处理所有顶层节点
            for node in tree.root_node.children:
                process_node(node)
                
            return functions, calls
            
        except Exception as e:
            raise Exception(f"Failed to parse C++ code: {str(e)}")

    def _analyze_function(self, node, source_code: bytes, class_name: Optional[str] = None,
                  namespace: Optional[str] = None) -> Tuple[str, FunctionInfo, Set[str]]:
        """
        Analyze a function node to extract its information.
        
        Args:
            node: The function node from AST
            source_code: Original source code in bytes
            class_name: Name of the containing class (if any)
            namespace: Name of the containing namespace (if any)
            
        Returns:
            Tuple containing:
            - Function name
            - FunctionInfo object
            - Set of function calls made by this function
        """
        if node.type != "function_definition":
            return None, None, set()

        try:
            # 获取函数声明器
            declarator = node.child_by_field_name("declarator")
            if not declarator:
                return None, None, set()

            # 获取函数名和参数
            func_name = None
            func_type = None
            params = []
            is_template = False
            is_constructor = False
            is_destructor = False
            
            # 检查是否是模板函数
            for child in node.children:
                if child.type == "template_declaration":
                    is_template = True
                    break
                    
            # 检查函数类型和名称
            def analyze_declarator(node):
                nonlocal func_name, func_type, is_constructor, is_destructor
                
                if node.type == "function_declarator":
                    declarator = node.child_by_field_name("declarator")
                    if declarator:
                        # 检查构造函数/析构函数
                        if class_name:
                            node_text = self._get_node_text(declarator, source_code)
                            if node_text == class_name:
                                is_constructor = True
                            elif node_text == f"~{class_name}":
                                is_destructor = True
                                
                        analyze_declarator(declarator)
                        
                        # 获取参数
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type == "parameter_declaration":
                                    params.append(self._get_node_text(param, source_code))
                                    
                elif node.type in ["identifier", "field_identifier"]:
                    func_name = self._get_node_text(node, source_code)
                    
                elif node.type == "destructor_name":
                    func_name = self._get_node_text(node, source_code)
                    is_destructor = True
                    
                elif node.type == "qualified_identifier":
                    for child in node.children:
                        if child.type in ["identifier", "field_identifier"]:
                            func_name = self._get_node_text(child, source_code)
                            break

            analyze_declarator(declarator)
            
            if not func_name and class_name:
                # 尝试从类名推断构造函数名
                func_name = class_name
                is_constructor = True

            # 构建完整的函数名
            if func_name:
                if class_name:
                    func_name = f"{class_name}::{func_name}"
                if namespace:
                    func_name = f"{namespace}::{func_name}"

            # 分析函数体中的调用
            body = node.child_by_field_name("body")
            calls = set()
            if body:
                for child in self._traverse_tree(body, source_code):
                    if child.type == "call_expression":
                        func_node = child.child_by_field_name("function")
                        if func_node:
                            call_name = self._get_node_text(func_node, source_code)
                            calls.add(call_name)

            # 创建函数信息对象
            func_info = FunctionInfo(
                name=func_name,
                code=self._get_node_text(node, source_code),
                docstring=None,
                params=params,
                return_type=func_type,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                class_name=class_name
            )

            return func_name, func_info, calls

        except Exception as e:
            raise Exception(f"Error analyzing function: {str(e)}")

    def _find_function_calls(self, func_node, source_code: bytes) -> Set[str]:
        """
        Find all function calls within a function body.
        
        Args:
            func_node: The function body node
            source_code: Original source code in bytes
            
        Returns:
            Set of function names that are called
        """
        calls = set()
        try:
            for node in self._traverse_tree(func_node, source_code):
                if node.type == "call_expression":
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        call_name = self._get_node_text(func_node, source_code)
                        calls.add(call_name)
        except Exception as e:
            raise Exception(f"Error in finding function calls: {e}")
        return calls