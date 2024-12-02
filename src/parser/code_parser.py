"""
File: src/parser/code_parser.py
Description: Code parser using Tree-sitter for extracting function information and call relationships.
"""

from tree_sitter import Language, Parser
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

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
            # Load language grammar
            Language.build_library(
                # Generate .so file
                Path(language_dir) / 'build/my-languages.so',
                # Include language files
                [Path(language_dir) / 'vendor/tree-sitter-python']
            )
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