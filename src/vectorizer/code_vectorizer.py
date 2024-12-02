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
            
    def vectorize(self, code_info: Dict[str, Any]) -> np.ndarray:
        """
        Generate vector representation for code.
        
        Args:
            code_info: Dictionary containing function information
            
        Returns:
            numpy.ndarray: Vector representation
        """
        try:
            # Prepare input text
            input_text = self._prepare_input(code_info)
            
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
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Failed to vectorize code: {e}")
            raise
            
    def _prepare_input(self, code_info: Dict[str, Any]) -> str:
        """Prepare input text for the model."""
        return f"""
        Function: {code_info['name']}
        Parameters: {', '.join(code_info['params'])}
        Docstring: {code_info.get('docstring', '')}
        Code:
        {code_info['code']}
        """.strip()