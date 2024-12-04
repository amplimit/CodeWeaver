"""
File: src/utils/id_generator.py
Description: Utility functions for generating unique identifiers for code functions.
"""

import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FunctionSignature:
    """Function signature information."""
    name: str
    params: List[str]
    return_type: Optional[str] = None
    
class IdGenerator:
    """Generates unique IDs for functions based on their signature and location."""
    
    @staticmethod
    def generate_function_id(
        file_path: str, 
        signature: FunctionSignature
    ) -> str:
        """
        Generate a unique function ID.
        
        Args:
            file_path: Source file path
            signature: Function signature information
            
        Returns:
            str: Unique function identifier
        """
        # Combine all signature components
        sig_str = f"{signature.name}::{','.join(signature.params)}"
        if signature.return_type:
            sig_str += f"::{signature.return_type}"
            
        # Create unique ID string
        id_string = f"{file_path}::{sig_str}"
        
        # Generate hash
        return hashlib.sha256(id_string.encode()).hexdigest()