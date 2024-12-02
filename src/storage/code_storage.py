"""
File: src/storage/code_storage.py
Description: Storage management for code vectors and call relationships.
"""

import faiss
import numpy as np
from typing import Dict, List, Any, Optional
import pickle
from pathlib import Path
import logging

class CodeStorage:
    def __init__(self, vector_dim: int = 512):
        """
        Initialize code storage.
        
        Args:
            vector_dim: Dimension of code vectors
        """
        try:
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(vector_dim)
            
            # Initialize mappings
            self.id_to_index: Dict[str, int] = {}
            self.index_to_id: Dict[int, str] = {}
            self.call_graph: Dict[str, List[str]] = {}
            self.functions: Dict[str, Dict[str, Any]] = {}
            
        except Exception as e:
            logging.error(f"Failed to initialize storage: {e}")
            raise
            
    def add_function(
        self,
        func_id: str,
        vector: np.ndarray,
        func_info: Dict[str, Any],
        callees: List[str]
    ) -> None:
        """
        Add a function to storage.
        
        Args:
            func_id: Unique function identifier
            vector: Function vector representation
            func_info: Function information
            callees: List of called function IDs
        """
        try:
            # Add to FAISS index
            index = self.index.ntotal
            self.index.add(np.array([vector]))
            
            # Update mappings
            self.id_to_index[func_id] = index
            self.index_to_id[index] = func_id
            self.call_graph[func_id] = callees
            self.functions[func_id] = func_info
            
        except Exception as e:
            logging.error(f"Failed to add function: {e}")
            raise
            
    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar functions.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of similar functions with distances
        """
        try:
            D, I = self.index.search(np.array([query_vector]), k)
            
            results = []
            for idx, dist in zip(I[0], D[0]):
                func_id = self.index_to_id.get(idx)
                if func_id:
                    results.append({
                        'function_id': func_id,
                        'distance': float(dist),
                        'info': self.functions[func_id],
                        'callees': self.call_graph[func_id]
                    })
            return results
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise
            
    def save(self, path: str) -> None:
        """
        Save storage state to disk.
        
        Args:
            path: Directory path for storage
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(path / 'vectors.index'))
            
            # Save metadata
            with open(path / 'metadata.pkl', 'wb') as f:
                pickle.dump({
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'call_graph': self.call_graph,
                    'functions': self.functions
                }, f)
                
        except Exception as e:
            logging.error(f"Failed to save storage: {e}")
            raise