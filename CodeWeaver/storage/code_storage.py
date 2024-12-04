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
    def __init__(self, vector_dim: int = 1024):  # multilingual-e5-large-instruct 默认维度是1024
        """
        Initialize code storage.
        
        Args:
            vector_dim: Dimension of code vectors
        """
        try:
            # 初始化FAISS索引 - 使用最基础的IndexFlatL2
            self.vector_dim = vector_dim
            self.index = faiss.IndexFlatL2(vector_dim)
            
            # 初始化映射
            self.id_to_index: Dict[str, int] = {}
            self.index_to_id: Dict[int, str] = {}
            self.call_graph: Dict[str, List[str]] = {}
            self.functions: Dict[str, Any] = {}
            
        except Exception as e:
            logging.error(f"Failed to initialize storage: {e}")
            raise
            
    def add_function(
        self,
        func_id: str,
        vector: np.ndarray,
        func_info: Any,
        callees: List[str]
    ) -> None:
        """
        Add a function to storage.
        
        Args:
            func_id: Unique function identifier
            vector: Function vector representation (shape: [1, vector_dim])
            func_info: Function information
            callees: List of called function IDs
        """
        try:
            # 验证向量维度
            if vector.shape != (1, self.vector_dim):
                raise ValueError(
                    f"Vector dimension mismatch. Expected (1, {self.vector_dim}), "
                    f"got {vector.shape}"
                )
            
            # 添加到FAISS索引
            index = self.index.ntotal
            self.index.add(vector)
            
            # 更新映射
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
            query_vector: Query vector with shape (1, vector_dim)
            k: Number of results to return
            
        Returns:
            List of similar functions with distances
        """
        try:
            # 验证查询向量维度
            if query_vector.shape != (1, self.vector_dim):
                raise ValueError(
                    f"Query vector dimension mismatch. Expected (1, {self.vector_dim}), "
                    f"got {query_vector.shape}"
                )
            
            # 搜索相似向量
            D, I = self.index.search(query_vector, k)
            
            # 构造结果
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
            
            # 保存FAISS索引
            faiss.write_index(self.index, str(path / 'vectors.index'))
            
            # 保存元数据
            with open(path / 'metadata.pkl', 'wb') as f:
                pickle.dump({
                    'vector_dim': self.vector_dim,
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'call_graph': self.call_graph,
                    'functions': self.functions
                }, f)
                
        except Exception as e:
            logging.error(f"Failed to save storage: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load storage state from disk.
        
        Args:
            path: Directory path for storage
        """
        try:
            path = Path(path)
            
            # 加载FAISS索引
            self.index = faiss.read_index(str(path / 'vectors.index'))
            
            # 加载元数据
            with open(path / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.vector_dim = metadata['vector_dim']
                self.id_to_index = metadata['id_to_index']
                self.index_to_id = metadata['index_to_id']
                self.call_graph = metadata['call_graph']
                self.functions = metadata['functions']
                
        except Exception as e:
            logging.error(f"Failed to load storage: {e}")
            raise