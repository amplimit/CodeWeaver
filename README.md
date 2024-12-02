# CodeWeaver

**CodeWeaver** is a versatile tool for analyzing, vectorizing, and storing code structures to enable semantic search and dependency exploration. Its modular design ensures flexibility, making it easy to extend or adapt for various use cases.

---

## 🚀 Features

1. **Code Parsing**:
   - Extracts functions, class methods, docstrings, parameters, and return types.
   - Analyzes function-to-function call relationships.
   - Generates unique identifiers for each function using robust signature-based techniques.

2. **Semantic Vectorization**:
   - Embeds code blocks into high-dimensional vector spaces using models like `multilingual-e5-large-instruct`.
   - Captures function semantics across multiple programming languages.

3. **Efficient Storage and Query**:
   - **FAISS** for high-performance semantic search.
   - Adjacency lists for graph-based dependency traversal.
   - Both systems interact seamlessly via function IDs.

4. **Modular Architecture**:
   - Clear separation of concerns with reusable components in parsing, vectorization, and storage.

---

## 📂 Project Structure

```plaintext
│  README.md           # Project documentation
│  requirements.txt    # Python dependencies
│
├─examples             # Example scripts for using CodeWeaver
│      example.py
│
├─src
│  ├─parser            # Code parsing and analysis
│  │      code_parser.py
│  │
│  ├─storage           # Storage management (FAISS and adjacency list)
│  │      code_storage.py
│  │
│  ├─utils             # Utility functions
│  │      id_generator.py
│  │
│  └─vectorizer        # Semantic vectorization
│          code_vectorizer.py
│
└─test                 # Unit and integration tests
│
└─grammars
   ├─tree-sitter-python
```

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CodeWeaver.git
   cd CodeWeaver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -m unittest discover -s test
   ```

---

## 🛠️ Usage

### Example Workflow

1. **Code Parsing**:
   Extract functions, relationships, and metadata:
   ```bash
   python src/parser/code_parser.py --source-dir /path/to/codebase
   ```

2. **Vectorization**:
   Generate semantic embeddings for parsed functions:
   ```bash
   python src/vectorizer/code_vectorizer.py --model multilingual-e5-large-instruct --output-dir ./embeddings
   ```

3. **Storage**:
   Save vectors in FAISS and relationships in adjacency list:
   ```bash
   python src/storage/code_storage.py --action store --input-dir ./embeddings
   ```

4. **Query**:
   Perform semantic or dependency-based searches:
   ```bash
   python src/storage/code_storage.py --action query --query "find a function to sort an array"
   ```

---

## 📖 Documentation

- [How to Contribute](CONTRIBUTING.md)
- [API Documentation](docs/api.md)
- [Use Cases](docs/use_cases.md)

---

## 🛠️ Roadmap

- [ ] Add support for graph databases (e.g., Neo4j).
- [ ] Implement multi-threaded parsing and vectorization.
- [ ] Extend support for more programming languages (e.g., Rust, Go).

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👥 Contributors

- Your Name ([i@stevenchen.site](mailto:i@stevenchen.site))