from tree_sitter import Language, Parser
import tree_sitter_cpp
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from pathlib import Path
from code_parser_py import FunctionInfo, ImportInfo

class CodeParserCpp:
    def __init__(self):
        try:
            self.parser = Parser()
            CPP_LANGUAGE = Language(tree_sitter_cpp.language())
            self.parser.language = CPP_LANGUAGE
        except Exception as e:
            logging.error(f"Failed to initialize C++ parser: {e}")
            raise

    def _get_node_text(self, node, source_code: bytes) -> str:
        return source_code[node.start_byte:node.end_byte].decode('utf8')

    def _traverse_tree(self, tree, source_code: bytes):
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

    def _extract_imports(self, tree, source_code: bytes) -> List[ImportInfo]:
        imports = []
        for node in self._traverse_tree(tree, source_code):
            if node.type == "preproc_include":
                path_node = node.child_by_field_name("path")
                if path_node:
                    path = self._get_node_text(path_node, source_code).strip('"<>')
                    imports.append(ImportInfo(
                        module=path,
                        names=[],
                        is_from_import=False
                    ))
            elif node.type == "using_declaration":
                name = self._get_node_text(node, source_code)
                imports.append(ImportInfo(
                    module=name.split("::")[0],
                    names=[name.split("::")[-1]],
                    is_from_import=True
                ))
        return imports

    def _find_function_calls(self, func_node, source_code: bytes) -> Set[str]:
        calls = set()
        for node in self._traverse_tree(func_node, source_code):
            if node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    calls.add(self._get_node_text(func_node, source_code))
        return calls

    def _analyze_function(self, node, source_code: bytes, class_name: Optional[str] = None,
                     namespace: Optional[str] = None) -> Tuple[str, FunctionInfo, Set[str]]:
        if node.type != "function_definition":
            return None, None, set()

        # 获取函数声明器
        declarator = node.child_by_field_name("declarator")
        if not declarator:
            return None, None, set()

        # 提取函数名和参数
        func_name = None
        params = []
        cursor = declarator.walk()

        def extract_name_and_params():
            nonlocal func_name
            node = cursor.node

            if node.type == "identifier" and not func_name:
                # 确保这不是参数的标识符
                parent = node.parent
                while parent:
                    if parent.type == "parameter_declaration":
                        break
                    parent = parent.parent
                if not parent:
                    func_name = self._get_node_text(node, source_code)

            elif node.type == "parameter_declaration":
                params.append(self._get_node_text(node, source_code))

            if cursor.goto_first_child():
                extract_name_and_params()
                cursor.goto_parent()

            if cursor.goto_next_sibling():
                extract_name_and_params()

        extract_name_and_params()

        if not func_name:
            return None, None, set()

        # 构建完整的函数名
        full_name = func_name
        if class_name:
            full_name = f"{class_name}::{full_name}"
        if namespace:
            full_name = f"{namespace}::{full_name}"

        # 提取函数体中的调用
        body = node.child_by_field_name("body")
        calls = self._find_function_calls(body, source_code) if body else set()

        func_info = FunctionInfo(
            name=full_name,
            code=self._get_node_text(node, source_code),
            docstring=None,
            params=params,
            return_type=None,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            class_name=class_name
        )

        return full_name, func_info, calls

    def extract_function_info(self, code_str: str, file_path: str) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        try:
            source_code = code_str.encode('utf8')
            tree = self.parser.parse(source_code)
            
            functions = {}
            calls = {}
            
            # 只处理顶层节点
            cursor = tree.walk()
            
            def process_top_level_node(node, namespace=None):
                if node.type == "namespace_definition":
                    ns_name = node.child_by_field_name("name")
                    current_namespace = self._get_node_text(ns_name, source_code) if ns_name else None
                    
                    # 处理命名空间内的所有顶层节点
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            process_top_level_node(child, current_namespace)
                            
                elif node.type == "class_specifier":
                    class_name = node.child_by_field_name("name")
                    class_name = self._get_node_text(class_name, source_code) if class_name else None
                    
                    body = node.child_by_field_name("body")
                    if body and len(body.children) > 0:
                        field_list = body.children[0]
                        in_public = False
                        
                        for child in field_list.children:
                            if child.type == "access_specifier":
                                in_public = self._get_node_text(child, source_code) == "public"
                                continue
                                
                            if in_public and child.type == "function_definition":
                                name, info, function_calls = self._analyze_function(child, source_code, class_name, namespace)
                                if name and info:
                                    full_name = f"{file_path}::{name}"
                                    functions[full_name] = info 
                                    calls[full_name] = list(function_calls)
                                elif field.type == "access_specifier":
                                    next_node = field.next_sibling
                                    while next_node and next_node.type != "access_specifier":
                                        if next_node.type == "function_definition":
                                            name, info, function_calls = self._analyze_function(next_node, source_code, class_name, namespace)
                                            if name and info:
                                                full_name = f"{file_path}::{name}"
                                                functions[full_name] = info
                                                calls[full_name] = list(function_calls)
                                        next_node = next_node.next_sibling
                                    
                elif node.type == "function_definition":
                    name, info, function_calls = self._analyze_function(node, source_code, None, namespace)
                    if name and info:
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                        calls[full_name] = list(function_calls)
            
            # 处理所有顶层节点
            for node in tree.root_node.children:
                process_top_level_node(node)
                
            return functions, calls
            
        except Exception as e:
            logging.error(f"Failed to parse C++ code: {str(e)}")
            raise