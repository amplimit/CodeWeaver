from tree_sitter import Language, Parser
import tree_sitter_python
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from pathlib import Path

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
    class_name: Optional[str] = None  # 所属类名
    source: Optional[str] = None      # 导入来源，格式如 "from xxx import yyy"

@dataclass
class ImportInfo:
    """存储导入信息"""
    module: str
    names: List[str]
    aliases: Dict[str, str] = None  # 存储名称到别名的映射
    is_from_import: bool = False    # 区分是否是from import

class CodeParser:
    def __init__(self):
        """Initialize the code parser."""
        try:
            self.parser = Parser()
            PY_LANGUAGE = Language(tree_sitter_python.language())
            self.parser.language = PY_LANGUAGE
            
            # 更新查询以处理更多导入情况
            self.import_query = PY_LANGUAGE.query("""
                (import_statement
                    name: (dotted_name) @import_path)
                (import_from_statement
                    module_name: (dotted_name)? @from_path
                    name: (dotted_name) @import_name)
                (aliased_import
                    name: (dotted_name) @import_name
                    alias: (identifier) @alias)
                (import_statement
                    name: (aliased_import
                        name: (dotted_name) @import_path
                        alias: (identifier) @alias))
            """)
            
        except Exception as e:
            logging.error(f"Failed to initialize parser: {e}")
            raise

    def _get_node_text(self, node, source_code: bytes) -> str:
        """获取节点的原始文本"""
        return source_code[node.start_byte:node.end_byte].decode('utf8')

    def _extract_imports(self, tree, source_code: bytes) -> List[ImportInfo]:
       """提取所有的导入信息"""
       imports = []
       current_import = None
       
       # 获取所有匹配
       query_matches = self.import_query.captures(tree.root_node)
       
       # 转换格式,从字典转为列表
       captures = []
       for type_name, nodes in query_matches.items():
           for node in nodes:
               captures.append((node, type_name))
               
       # 如果没有匹配到任何import语句，直接返回空列表
       if not captures:
           return []
       
       i = 0
       while i < len(captures):
           node, type_name = captures[i]
           text = self._get_node_text(node, source_code)
           
           if type_name == 'from_path':
               current_import = ImportInfo(
                   module=text, 
                   names=[], 
                   aliases={},
                   is_from_import=True
               )
           elif type_name == 'import_path':
               current_import = ImportInfo(
                   module=text,
                   names=[text],
                   aliases={},
                   is_from_import=False
               )
           elif type_name == 'import_name':
               if current_import and current_import.is_from_import:
                   current_import.names.append(text)
                   # 检查下一个捕获是否是别名
                   if i + 1 < len(captures) and captures[i + 1][1] == 'alias':
                       alias_node = captures[i + 1][0]
                       alias = self._get_node_text(alias_node, source_code)
                       current_import.aliases[text] = alias
                       i += 1
           elif type_name == 'alias' and current_import and not current_import.is_from_import:
               original_name = current_import.names[-1]
               current_import.aliases[original_name] = text
           
           # 完成一个导入语句的处理
           if current_import and (
               type_name in ('import_path', 'alias') or 
               (type_name == 'import_name' and current_import.is_from_import)
           ):
               imports.append(current_import)
               current_import = None
           
           i += 1
       
       return imports

    def _find_function_calls(self, func_node, source_code: bytes) -> Set[str]:
        """查找函数体中的所有函数调用"""
        calls = set()
        cursor = func_node.walk()
        
        def visit_node():
            node = cursor.node
            
            if node.type == 'call':
                func_node = node.child_by_field_name('function')
                if func_node:
                    # 处理各种类型的函数调用
                    if func_node.type == 'identifier':
                        calls.add(self._get_node_text(func_node, source_code))
                    elif func_node.type == 'attribute':
                        call_chain = []
                        current = func_node
                        while current:
                            if current.type == 'identifier':
                                call_chain.append(self._get_node_text(current, source_code))
                                break
                            elif current.type == 'attribute':
                                attr_node = current.child_by_field_name('attribute')
                                if attr_node:
                                    call_chain.append(self._get_node_text(attr_node, source_code))
                                current = current.child_by_field_name('object')
                            else:
                                break
                        if call_chain:
                            calls.add('.'.join(reversed(call_chain)))
            
            if cursor.goto_first_child():
                visit_node()
                cursor.goto_parent()
            
            if cursor.goto_next_sibling():
                visit_node()
        
        # 从函数体开始分析
        body_node = func_node.child_by_field_name('body')
        if body_node:
            cursor.reset(body_node)
            visit_node()
            
        return calls

    def _analyze_function(
        self, 
        node, 
        source_code: bytes, 
        class_name: Optional[str] = None,
        imports: List[ImportInfo] = None
    ) -> Tuple[str, FunctionInfo, Set[str]]:
        """分析函数节点的详细信息"""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None, None, set()
        
        func_name = self._get_node_text(name_node, source_code)
        
        # 获取参数
        params = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type != ',':
                    param_text = self._get_node_text(param, source_code)
                    if not (class_name and param_text == 'self'):
                        params.append(param_text)
        
        # 获取docstring
        docstring = None
        body_node = node.child_by_field_name('body')
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == 'string':
                    docstring = self._get_node_text(expr, source_code)
        
        # 获取函数调用
        calls = self._find_function_calls(node, source_code)
        
        # 检查是否是导入的函数
        source = None
        if imports:
            for imp in imports:
                if func_name in imp.names:
                    if imp.is_from_import:
                        source = f"from {imp.module} import {func_name}"
                    else:
                        source = f"import {imp.module}"
                    break
        
        func_info = FunctionInfo(
            name=func_name,
            code=self._get_node_text(node, source_code),
            docstring=docstring,
            params=params,
            return_type=None,  # TODO: 添加返回类型解析
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            class_name=class_name,
            source=source
        )
        
        return func_name, func_info, calls

    def _find_functions_in_node(
        self, 
        node, 
        source_code: bytes,
        imports: List[ImportInfo],
        class_name: Optional[str] = None
    ) -> Tuple[Dict[str, FunctionInfo], Dict[str, Set[str]]]:
        """在给定节点中递归查找所有函数"""
        functions = {}
        calls_dict = {}
        cursor = node.walk()
        
        def visit_node():
            node = cursor.node
            
            if node.type == 'class_definition':
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    current_class = self._get_node_text(class_name_node, source_code)
                    body_node = node.child_by_field_name('body')
                    if body_node:
                        class_funcs, class_calls = self._find_functions_in_node(
                            body_node, source_code, imports, current_class)
                        functions.update(class_funcs)
                        calls_dict.update(class_calls)
            
            elif node.type == 'function_definition':
                func_name, func_info, func_calls = self._analyze_function(
                    node, source_code, class_name, imports)
                if func_name and func_info:
                    if class_name:
                        full_name = f"{class_name}.{func_name}"
                    else:
                        full_name = func_name
                    functions[full_name] = func_info
                    calls_dict[full_name] = func_calls
            
            if cursor.goto_first_child():
                visit_node()
                cursor.goto_parent()
                
            if cursor.goto_next_sibling():
                visit_node()
        
        visit_node()
        return functions, calls_dict

    def extract_function_info(
        self, 
        code_str: str, 
        file_path: str
    ) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        """提取函数信息和调用图"""
        try:
            source_code = code_str.encode('utf8')
            tree = self.parser.parse(source_code)
            
            # 获取导入信息
            imports = self._extract_imports(tree, source_code)
            
            # 构建导入映射
            import_map = {}  # 存储函数名到模块的映射
            for imp in imports:
                if imp.names:
                    for name in imp.names:
                        actual_name = name
                        if imp.aliases and name in imp.aliases:
                            actual_name = imp.aliases[name]
                        import_map[actual_name] = (imp.module, imp.is_from_import)
            
            # 获取所有函数及其调用
            functions_dict, raw_calls = self._find_functions_in_node(
                tree.root_node, source_code, imports)
            
            # 构建最终的函数字典和调用图
            functions = {}
            call_graph = {}
            
            for func_name, func_info in functions_dict.items():
                func_id = f"{file_path}::{func_name}"
                functions[func_id] = func_info
                
                # 处理函数调用
                if func_name in raw_calls:
                    processed_calls = []
                    for call in raw_calls[func_name]:
                        parts = call.split('.')
                        base_name = parts[0]
                        
                        if base_name in import_map:
                            # 导入的函数调用
                            module, is_from = import_map[base_name]
                            if is_from:
                                processed_calls.append(f"{module}::{call}")
                            else:
                                if len(parts) > 1:
                                    processed_calls.append(f"{module}::{'.'.join(parts[1:])}")
                                else:
                                    processed_calls.append(f"{module}::{base_name}")
                        else:
                            # 本地函数调用
                            processed_calls.append(f"{file_path}::{call}")
                    
                    call_graph[func_id] = processed_calls
            
            return functions, call_graph
            
        except Exception as e:
            logging.error(f"Failed to parse code: {str(e)}")
            raise