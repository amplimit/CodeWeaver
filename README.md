# CodeWeaver

CodeWeaver is an intelligent code analysis and understanding tool that uses advanced natural language processing and vector similarity search to help developers navigate and comprehend large codebases.

## 🌟 Features

- **Semantic Code Analysis**: Analyzes Python codebases to understand function relationships, class hierarchies, and code semantics
- **Vector-Based Code Search**: Find similar code snippets and functions using semantic similarity
- **Call Graph Generation**: Visualize function call relationships within your codebase
- **Incremental Analysis**: Save and load analysis results to avoid reprocessing unchanged code
- **AI-Powered Code Understanding**: Integrate with LLMs for intelligent code comprehension and Q&A

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git (for cloning repositories)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/StevenChen16/CodeWeaver.git
cd CodeWeaver
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage

1. **Analyze a Codebase**:
```python
from analyze_codebase import analyze_codebase

# Analyze local directory or Git repository
storage, info = analyze_codebase("path/to/your/code")
```

2. **Query Similar Functions**:
```python
from query import search_similar_functions, init_vectorizer

# Initialize components
vectorizer = init_vectorizer()

# Search for similar functions
results = search_similar_functions(storage, vectorizer, "your query or code snippet")
```

3. **Use AI Assistant**:
```python
from AI import CodeUnderstandingAI

# Initialize AI assistant
ai = CodeUnderstandingAI()

# Analyze project and start chat interface
ai.analyze_project("path/to/your/code")
ai.chat()
```

#### Advanced Features

- **Call Graph Analysis**: Explore function call relationships
- **Code Similarity Search**: Find semantically similar code snippets
- **Interactive AI Chat**: Ask questions about your codebase

## 🏗️ Project Structure

```
CodeWeaver/
├── CodeWeaver/           # Core package
│   ├── parser/          # Code parsing modules
│   ├── storage/         # Vector storage and retrieval
│   ├── utils/          # Utility functions
│   └── vectorizer/     # Code vectorization
├── examples/           # Usage examples
├── grammars/          # Tree-sitter grammars
└── AI.py              # AI assistant interface
```

## 🛠️ Core Components

### Parser (`CodeWeaver.parser`)
Handles code parsing and extraction of function information, class hierarchies, and call graphs.

### Vectorizer (`CodeWeaver.vectorizer`)
Converts code into semantic vectors using advanced language models.

### Storage (`CodeWeaver.storage`)
Manages vector storage and retrieval using FAISS for efficient similarity search.

### AI Assistant (`AI.py`)
Provides an interactive interface for code understanding and querying using LLMs.

## 📝 Example

```python
from CodeWeaver.parser.code_parser import CodeParser
from CodeWeaver.vectorizer.code_vectorizer import CodeVectorizer
from CodeWeaver.storage.code_storage import CodeStorage

# Initialize components
parser = CodeParser()
vectorizer = CodeVectorizer()
storage = CodeStorage(vector_dim=1024)

# Process code and store vectors
functions, call_graph = parser.extract_function_info(code)
for func_id, func_info in functions.items():
    vector = vectorizer.vectorize(func_info)
    storage.add_function(func_id, vector, func_info, call_graph.get(func_id, []))

# Search for similar functions
similar_functions = storage.search_similar(query_vector, k=5)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Contact

- GitHub: [@StevenChen16](https://github.com/StevenChen16)
- Email: [i@stevenchen.site](mailto:i@stevenchen.site)
- Project Link: [https://github.com/StevenChen16/CodeWeaver](https://github.com/StevenChen16/CodeWeaver)

## 🙏 Acknowledgments

- Thanks to all contributors who have helped with the project
- Built with [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for reliable code parsing
- Powered by [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search