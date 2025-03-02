"""
File: src/vectorizer/code_vectorizer.py
Description: Code vectorization using pre-trained language models.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional
import numpy as np
import logging
from dataclasses import dataclass
from ..parser.code_parser_py import FunctionInfo  # 导入 FunctionInfo 类型

@dataclass
class VectorizeConfig:
    """Configuration for code vectorization."""
    model_name: str = 'intfloat/multilingual-e5-large-instruct'
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None

class CodeVectorizer:
    def __init__(self, config: VectorizeConfig):
        """
        Initialize the code vectorizer.
        
        Args:
            config: Vectorization configuration
        """
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            
            # Setup device
            self.device = (torch.device(config.device) if config.device
                         else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"Failed to initialize vectorizer: {e}")
            raise
            
    def vectorize(self, function_info: FunctionInfo) -> np.ndarray:
        """
        Generate vector representation for code.
        
        Args:
            function_info: FunctionInfo object containing function information
            
        Returns:
            numpy.ndarray: Vector representation with shape (1, vector_dim)
        """
        try:
            # Prepare input text
            input_text = self._prepare_input(function_info)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
            # 确保返回的是 (1, dim) 形状的数组
            vector = embeddings.cpu().numpy()
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)
            return vector
            
        except Exception as e:
            logging.error(f"Failed to vectorize code: {e}")
            raise
            
    def _prepare_input(self, function_info: FunctionInfo) -> str:
        """Prepare input text for the model."""
        return f"""
        Function: {function_info.name}
        Parameters: {', '.join(function_info.params)}
        Docstring: {function_info.docstring or ''}
        Code:
        {function_info.code}
        """.strip()