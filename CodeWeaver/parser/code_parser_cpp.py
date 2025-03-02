from tree_sitter import Language, Parser
import tree_sitter_cpp
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path
from .code_parser_py import FunctionInfo, ImportInfo

class CodeParserCpp:
    """
    C++ code parser using tree-sitter library to analyze C++ source code.
    Extracts information about functions, methods, imports and their relationships.
    
    This parser can handle:
    - Global functions
    - Class methods (public/private)
    - Namespace functions (including nested)
    - Import statements and aliases
    - Function calls
    - Template functions
    - Constructors and destructors
    - Function overloading
    - Virtual functions
    """

    def __init__(self):
        """Initialize the C++ parser with tree-sitter."""
        try:
            self.parser = Parser()
            CPP_LANGUAGE = Language(tree_sitter_cpp.language())
            self.parser.language = CPP_LANGUAGE
        except Exception as e:
            raise Exception(f"Failed to initialize C++ parser: {e}")

    def _get_node_text(self, node, source_code: bytes) -> str:
        """Extract text content from an AST node."""
        try:
            return source_code[node.start_byte:node.end_byte].decode('utf8')
        except Exception as e:
            raise Exception(f"Failed to get node text: {e}")

    def _traverse_tree(self, tree, source_code: bytes):
        """Traverse the AST tree in a depth-first manner."""
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
        """Extract all import statements and using declarations."""
        print("\nProcessing imports:")

        imports = []
        namespace_aliases = {}  # Track namespace aliases
        
        for node in self._traverse_tree(tree, source_code):
            if node.type == "preproc_include":
                # Handle #include statements
                path_node = node.child_by_field_name("path")
                if path_node:
                    path = self._get_node_text(path_node, source_code).strip('"<>')
                    imports.append(ImportInfo(module=path, names=[], is_from_import=False))
                    print(f"Found include: {path}")
                    
            elif node.type == "using_directive":
                # Handle using namespace xxx statements
                namespace = self._get_node_text(node.child_by_field_name("name"), source_code)
                imports.append(ImportInfo(module=namespace, names=[], is_from_import=True))
                print(f"Found using namespace: {namespace}")
                    
            elif node.type == "using_declaration":
                # Handle using std::string statements
                name = self._get_node_text(node, source_code)
                if "::" in name:
                    module, item = name.rsplit("::", 1)
                    imports.append(ImportInfo(module=module, names=[item], is_from_import=True))
                    print(f"Found using declaration: {module}::{item}")
                else:
                    imports.append(ImportInfo(module=name, names=[], is_from_import=True))
                    print(f"Found using declaration: {name}")
                    
            elif node.type == "namespace_alias_definition":
                # Handle namespace alias definitions
                name = self._get_node_text(node.child_by_field_name("name"), source_code)
                value = self._get_node_text(node.child_by_field_name("value"), source_code)
                namespace_aliases[name] = value
                
            elif node.type == "declaration":
                # Handle using alias declarations
                if len(node.children) >= 3:
                    alias = None
                    value = None
                    
                    for child in node.children:
                        if child.type == "type_identifier":
                            alias = self._get_node_text(child, source_code)
                        elif child.type == "qualified_identifier":
                            value = self._get_node_text(child, source_code)
                            
                    if alias and value:
                        imports.append(ImportInfo(module=value, names=[alias], is_from_import=True))
                        
        return imports

    def _process_nested_namespaces(self, node, source_code: bytes, current_namespace=None):
        """Recursively process nested namespaces."""
        functions = {}
        calls = {}
        
        if node.type == "namespace_definition":
            ns_name = node.child_by_field_name("name")
            ns_text = self._get_node_text(ns_name, source_code) if ns_name else None
            
            if ns_text:
                current_namespace = f"{current_namespace}::{ns_text}" if current_namespace else ns_text
                
            body = node.child_by_field_name("body")
            if body:
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

    def _find_function_calls(self, func_node, source_code: bytes) -> List[str]:
        """Find all function calls within a function body."""
        calls = []  # Using list to preserve multiple calls to same function
        try:
            for node in self._traverse_tree(func_node, source_code):
                if node.type == "call_expression":
                    function_node = node.child_by_field_name("function")
                    if function_node:
                        call_name = self._get_node_text(function_node, source_code)
                        calls.append(call_name)
        except Exception as e:
            raise Exception(f"Error in finding function calls: {e}")
        return calls

    def _get_function_signature(self, node, source_code: bytes) -> str:
        """Get full function signature including parameters."""
        try:
            declarator = node.child_by_field_name("declarator")
            if not declarator:
                return None
                
            func_name = None
            params = []
            
            # Get function name
            for child in self._traverse_tree(declarator, source_code):
                if child.type in ["identifier", "field_identifier", "destructor_name", "operator_name"]:
                    name = self._get_node_text(child, source_code)
                    if not any(p for p in params if name in p):  # Avoid capturing param names
                        func_name = name
                        break
                elif child.type == "parameter_declaration":
                    params.append(self._get_node_text(child, source_code))
                    
            # Special handling for constructors/destructors
            if not func_name:
                for child in node.children:
                    if child.type == "function_declarator":
                        for subchild in child.children:
                            if subchild.type in ["identifier", "field_identifier"]:
                                func_name = self._get_node_text(subchild, source_code)
                                break
                        
            if func_name:
                return f"{func_name}{'(' + ', '.join(params) + ')'}"
            return None
            
        except Exception as e:
            print(f"Error getting function signature: {e}")
            return None

    def _is_pure_virtual(self, node, source_code: bytes) -> bool:
        """Check if a function is pure virtual (= 0)."""
        # Look for = 0 pattern in function body
        for i in range(len(node.children) - 1):
            curr = node.children[i]
            next = node.children[i + 1]
            
            if curr.type == "=" and next.type == "number_literal":
                if self._get_node_text(next, source_code).strip() == "0":
                    print(f"Found pure virtual function")
                    return True
                    
        # Also check function declarator
        declarator = node.child_by_field_name("declarator")
        if declarator:
            for child in declarator.children:
                if child.type == "= 0":
                    print(f"Found pure virtual function (declarator)")
                    return True
                    
        return False

    def _analyze_function(self, node, source_code: bytes, class_name: Optional[str] = None,
                         namespace: Optional[str] = None) -> Tuple[str, FunctionInfo, List[str]]:
        """Analyze a function node to extract its information."""
        if node.type != "function_definition":
            return None, None, []

        try:
            declarator = node.child_by_field_name("declarator")
            if not declarator:
                return None, None, []

            # Get basic function info
            func_name = None
            params = []
            is_constructor = False
            is_destructor = False
            is_const = False
            is_virtual = False
            is_pure_virtual = self._is_pure_virtual(node, source_code)
            
            # Get function signature and name
            signature = self._get_function_signature(node, source_code)
            if signature:
                func_name = signature.split('(')[0]
                
                # Check for constructor/destructor
                if class_name:
                    if func_name == class_name:
                        is_constructor = True
                        print(f"Found constructor: {class_name}")
                    elif func_name == f"~{class_name}":
                        is_destructor = True
                        print(f"Found destructor: {class_name}")
                    else:
                        print(f"Function name: {func_name}, Class name: {class_name}")

            # Check function specifiers
            for child in node.children:
                if child.type == "virtual":
                    is_virtual = True
                elif child.type == "const_qualifier":
                    is_const = True
                print(f"Function modifiers - virtual: {is_virtual}, const: {is_const}, pure virtual: {is_pure_virtual}")

            # Get parameters
            parameters = declarator.child_by_field_name("parameters")
            if parameters:
                for param in parameters.children:
                    if param.type == "parameter_declaration":
                        params.append(self._get_node_text(param, source_code))

            # Build full name
            if func_name:
                if class_name:
                    func_name = f"{class_name}::{func_name}"
                if namespace:
                    func_name = f"{namespace}::{func_name}"

            # Get function calls
            body = node.child_by_field_name("body")
            calls = self._find_function_calls(body, source_code) if body else []

            # Create function info
            func_info = FunctionInfo(
                name=func_name,
                code=self._get_node_text(node, source_code),
                docstring=None,
                params=params,
                return_type=None,  # Could be enhanced to extract return type
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                class_name=class_name
            )

            return func_name, func_info, calls

        except Exception as e:
            raise Exception(f"Error analyzing function: {str(e)}")

    def extract_function_info(self, code_str: str, file_path: str) -> Tuple[Dict[str, FunctionInfo], Dict[str, List[str]]]:
        """Extract information about all functions in the code."""
        try:
            source_code = code_str.encode('utf8')
            tree = self.parser.parse(source_code)
            if tree.root_node.has_error:
                raise Exception("Failed to parse the code: syntax error detected")
                
            functions = {}
            calls = {}
            
            def process_node(node, current_namespace=None):
                """Process a single AST node."""
                if node.type == "namespace_definition":
                    # Handle namespaces
                    ns_funcs, ns_calls = self._process_nested_namespaces(node, source_code)
                    for name, info in ns_funcs.items():
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                    for name, func_calls in ns_calls.items():
                        full_name = f"{file_path}::{name}"
                        calls[full_name] = func_calls
                        
                elif node.type == "class_specifier":
                    # Handle classes
                    class_name = node.child_by_field_name("name")
                    class_name = self._get_node_text(class_name, source_code) if class_name else None
                    
                    if class_name:
                        body = node.child_by_field_name("body")
                        if body:
                            in_public = True  # Default public for struct
                            class_type = node.child_by_field_name("type")
                            if class_type and self._get_node_text(class_type, source_code) == "class":
                                in_public = False  # Default private for class
                                
                            for child in body.children:
                                if child.type == "access_specifier":
                                    access_text = self._get_node_text(child, source_code)
                                    in_public = access_text.strip(':') == "public"
                                    continue
                                    
                                if child.type == "function_definition":
                                    # Include all public methods and constructor/destructor
                                    name, info, function_calls = self._analyze_function(
                                        child, source_code, class_name, current_namespace)
                                    if name and info:
                                        is_special = name == class_name or name == f"~{class_name}"
                                        if in_public or is_special:
                                            full_name = f"{file_path}::{name}"
                                            functions[full_name] = info
                                            calls[full_name] = function_calls
                                        
                elif node.type == "template_declaration":
                    print("\nProcessing template declaration:")
                    # Handle template functions
                    declaration = node.child_by_field_name("declaration")
                    if declaration and declaration.type == "function_definition":
                        name, info, function_calls = self._analyze_function(
                            declaration, source_code, None, current_namespace)
                        if name and info:
                            # Store template parameters
                            template_params = node.child_by_field_name("parameters")
                            if template_params:
                                info.template_params = [
                                    self._get_node_text(param, source_code)
                                    for param in template_params.children
                                    if param.type == "type_parameter"
                                ]
                            # 使用完整签名作为key
                            signature = self._get_function_signature(declaration, source_code)
                            if signature:
                                full_name = f"{file_path}::{signature}"
                                functions[full_name] = info
                                print(f"Added template function: {full_name}")
                            calls[full_name] = function_calls
                            
                elif node.type == "function_definition":
                    # Handle regular functions
                    name, info, function_calls = self._analyze_function(
                        node, source_code, None, current_namespace)
                    if name and info:
                        full_name = f"{file_path}::{name}"
                        functions[full_name] = info
                        calls[full_name] = function_calls

            # Process all top-level nodes
            for node in tree.root_node.children:
                process_node(node)
                
            return functions, calls
            
        except Exception as e:
            raise Exception(f"Failed to parse C++ code: {str(e)}")