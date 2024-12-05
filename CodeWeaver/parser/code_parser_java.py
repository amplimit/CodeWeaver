from tree_sitter import Language, Parser
import tree_sitter_java
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from code_parser_py import FunctionInfo, ImportInfo

class CodeParserJava:
    def __init__(self):
        try:
            self.parser = Parser()
            JAVA_LANGUAGE = Language(tree_sitter_java.language())
            self.parser.language = JAVA_LANGUAGE
        except Exception as e:
            raise

    def _get_node_text(self, node, source_code: bytes) -> str:
        return source_code[node.start_byte:node.end_byte].decode('utf8')

    def _traverse_tree(self, node, source_code: bytes):
        cursor = node.walk()
        
        def visit():
            node = cursor.node
            yield node
            
            if cursor.goto_first_child():
                yield from visit()
                cursor.goto_parent()
            
            if cursor.goto_next_sibling():
                yield from visit()
                
        yield from visit()

    def _extract_imports(self, tree, source_code: bytes) -> Tuple[str, List[ImportInfo]]:
        imports = []
        package_name = None
        
        for node in self._traverse_tree(tree.root_node, source_code):
            if node.type == 'package_declaration':
                scoped_name = None
                # 遍历包声明的所有子节点寻找标识符
                for child in self._traverse_tree(node, source_code):
                    if child.type == 'scoped_identifier':
                        scoped_name = self._get_node_text(child, source_code)
                        break
                if scoped_name:
                    package_name = scoped_name
            elif node.type == 'import_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    import_path = self._get_node_text(name_node, source_code)
                    parts = import_path.split('.')
                    module = '.'.join(parts[:-1])
                    name = parts[-1]
                    imports.append(ImportInfo(
                        module=module,
                        names=[name],
                        is_from_import=True
                    ))
        
        return package_name, imports

    def _get_full_method_chain(self, node, source_code: bytes) -> str:
        parts = []
        current = node
        
        while current:
            if current.type == 'identifier':
                parts.append(self._get_node_text(current, source_code))
                break
            elif current.type == 'method_invocation':
                name = current.child_by_field_name('name')
                if name:
                    parts.append(self._get_node_text(name, source_code))
                current = current.child_by_field_name('object')
            elif current.type == 'field_access':
                field = current.child_by_field_name('field')
                if field:
                    parts.append(self._get_node_text(field, source_code))
                current = current.child_by_field_name('object')
            else:
                break
                
        return '.'.join(reversed(parts)) if parts else None

    def _find_function_calls(self, method_node, source_code: bytes) -> Set[str]:
        calls = set()
        
        if not method_node:
            return calls
            
        for node in self._traverse_tree(method_node, source_code):
            if node.type == 'method_invocation':
                full_chain = self._get_full_method_chain(node, source_code)
                if full_chain:
                    calls.add(full_chain)
                else:
                    name = node.child_by_field_name('name')
                    if name:
                        calls.add(self._get_node_text(name, source_code))
                        
        return calls

    def _get_docstring(self, node, source_code: bytes) -> Optional[str]:
        current = node.prev_sibling
        while current:
            text = self._get_node_text(current, source_code)
            if current.type == 'line_comment':
                if text.strip().startswith('//'):
                    break
            elif current.type == 'block_comment':
                if text.strip().startswith('/**'):
                    return text
            current = current.prev_sibling
        return None

    def _analyze_method(
        self, 
        node, 
        source_code: bytes, 
        class_name: Optional[str] = None,
        imports: List[ImportInfo] = None
    ) -> Tuple[str, FunctionInfo, Set[str]]:
        is_constructor = node.type == 'constructor_declaration'
        
        name_node = node.child_by_field_name('name')
        if not name_node and not is_constructor:
            return None, None, set()
            
        if is_constructor:
            method_name = class_name.split('.')[-1] if class_name else "constructor"
        else:
            method_name = self._get_node_text(name_node, source_code)
        
        params = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for child in self._traverse_tree(params_node, source_code):
                if child.type == 'formal_parameter':
                    name = child.child_by_field_name('name')
                    if name:
                        params.append(self._get_node_text(name, source_code))
        
        return_type = None
        if not is_constructor:
            type_node = node.child_by_field_name('type')
            if type_node:
                return_type = self._get_node_text(type_node, source_code)
                # 去除泛型部分,只保留基本类型
                if '<' in return_type:
                    return_type = return_type.split('<')[0]
        
        docstring = self._get_docstring(node, source_code)
        
        body = node.child_by_field_name('body')
        calls = self._find_function_calls(body, source_code) if body else set()
        
        method_info = FunctionInfo(
            name=method_name,
            code=self._get_node_text(node, source_code),
            docstring=docstring,
            params=params,
            return_type=return_type,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            class_name=class_name
        )
        
        return method_name, method_info, calls

    def extract_function_info(
        self, 
        code_str: str, 
        file_path: str
    ) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        try:
            source_code = code_str.encode('utf8')
            tree = self.parser.parse(source_code)
            
            package_name, imports = self._extract_imports(tree, source_code)
            functions = {}
            calls = {}
            
            def get_full_class_name(class_name: str, scope: Optional[str] = None) -> str:
                result = class_name
                if package_name:
                    result = f"{package_name}.{class_name}"
                elif scope:
                    result = f"{scope}.{class_name}"
                return result
            
            def process_class(node, parent_scope=None):
                class_name = None
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    base_name = self._get_node_text(class_name_node, source_code)
                    class_name = get_full_class_name(base_name, parent_scope)
                
                body = node.child_by_field_name('body')
                if body:
                    for child in self._traverse_tree(body, source_code):
                        if child.type == 'class_declaration':
                            process_class(child, class_name)
                        elif child.type in ('method_declaration', 'constructor_declaration'):
                            name, info, method_calls = self._analyze_method(
                                child, source_code, class_name, imports)
                            if name and info:
                                qualified_name = f"{file_path}::{class_name}.{name}"
                                functions[qualified_name] = info
                                calls[qualified_name] = list(method_calls)
            
            for node in tree.root_node.children:
                if node.type == 'class_declaration':
                    process_class(node)
                elif node.type == 'method_declaration':
                    name, info, method_calls = self._analyze_method(node, source_code, None, imports)
                    if name and info:
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                        calls[full_name] = list(method_calls)
            
            return functions, calls
            
        except Exception as e:
            raise