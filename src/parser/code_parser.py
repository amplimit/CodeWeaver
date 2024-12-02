"""
File: src/parser/code_parser.py
Description: Code parser using Tree-sitter for extracting function information and call relationships.
"""

from tree_sitter import Language, Parser
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import os

from ..utils.id_generator import IdGenerator, FunctionSignature

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
    def __init__(self, language_dir: str):
        """
        Initialize the code parser.
        
        Args:
            language_dir: Directory containing tree-sitter language files
        """
        try:
            self.parser = Parser()
            
            # Load the pre-compiled language file
            language_path = os.path.join(language_dir, 'tree-sitter-python/python.so')
            if not os.path.exists(language_path):
                raise FileNotFoundError(
                    f"Language file not found at {language_path}. "
                    "Please compile the language grammar first."
                )
                
            # Create Language object
            PY_LANGUAGE = Language(language_path, 'python')
            
            # Set language for parser
            self.parser.set_language(PY_LANGUAGE)
            
        except Exception as e:
            logging.error(f"Failed to initialize parser: {e}")
            raise
            
    def _extract_function_info(self, node, code_str: str) -> FunctionInfo:
        """Extract information from a function node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        
        # Get function name
        name_node = node.child_by_field_name('name')
        func_name = code_str[name_node.start_byte:name_node.end_byte] if name_node else ""
        
        # Get parameters
        params = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.named_children:
                params.append(code_str[param.start_byte:param.end_byte])
        
        # Get docstring if exists
        docstring = None
        body_node = node.child_by_field_name('body')
        if body_node and body_node.named_children:
            first_child = body_node.named_children[0]
            if first_child.type == 'string':
                docstring = code_str[first_child.start_byte:first_child.end_byte]
        
        return FunctionInfo(
            name=func_name,
            code=code_str[start_byte:end_byte],
            docstring=docstring,
            params=params,
            return_type=None,  # Can be extended to extract return type
            start_line=node.start_point[0],
            end_line=node.end_point[0]
        )
    
            
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
            tree = self.parser.parse(bytes(code_str, 'utf8'))
            
            functions: Dict[str, FunctionInfo] = {}
            call_graph: Dict[str, List[str]] = {}
            
            # Process AST
            cursor = tree.walk()
            self._process_node(cursor, code_str, file_path, functions, call_graph)
            
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