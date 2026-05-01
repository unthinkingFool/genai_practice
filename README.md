# 🤖 GenAI Practice Repository

A hands-on learning repository for Generative AI — covering chat models, embedding models, LangChain, LangGraph, RAG pipelines, vector databases, and AI agent tools.

---

## 📌 About This Repo

This repo documents my journey learning **Generative AI from the ground up**. Every folder contains working code, experiments, and notes from real practice sessions. The goal is to build a solid foundation in LLMs and AI agent development.

---

## 🗂️ Repository Structure

```
genai-practice/
│
├── 01_chat_models/          # Interacting with LLMs via APIs (OpenAI, Groq, Anthropic, etc.)
│   ├── basic_chat.py
│   ├── streaming.py
│   └── system_prompts.py
│
├── 02_embedding_models/     # Text embeddings, similarity search, cosine distance
│   ├── basic_embeddings.py
│   └── similarity_search.py
│
├── 03_langchain/            # LangChain chains, prompt templates, output parsers
│   ├── prompt_templates.py
│   ├── chains.py
│   ├── output_parsers.py
│   └── memory.py
│
├── 04_langgraph/            # Stateful multi-agent workflows with LangGraph
│   ├── simple_graph.py
│   ├── conditional_edges.py
│   └── multi_agent_pipeline.py
│
├── 05_rag/                  # Retrieval-Augmented Generation pipelines
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── retrieval_chain.py
│   └── rag_pipeline.py
│
├── 06_vector_databases/     # Working with FAISS, Chroma, Pinecone, Qdrant
│   ├── faiss_demo.py
│   ├── chroma_demo.py
│   └── pinecone_demo.py
│
├── 07_agent_tools/          # Building and using tools with LLM agents
│   ├── custom_tools.py
│   ├── search_agent.py
│   └── tool_calling.py
│
├── 08_projects/             # End-to-end mini projects combining everything
│   └── research_pipeline/   # Multi-agent research assistant (LangGraph + Tavily)
│
├── notes/                   # Personal notes and concept summaries
│   ├── concepts.md
│   └── resources.md
│
├── requirements.txt
└── README.md
```

---

## 🧠 Topics Covered

| Area | Topics |
|---|---|
| **Chat Models** | LLM API calls, streaming, system prompts, temperature, multi-turn conversations |
| **Embedding Models** | Text vectorization, cosine similarity, semantic search |
| **LangChain** | Prompt templates, chains (LCEL), memory, output parsers, document loaders |
| **LangGraph** | Stateful graphs, nodes & edges, conditional routing, human-in-the-loop |
| **RAG** | Document ingestion, chunking strategies, retrieval chains, context injection |
| **Vector Databases** | FAISS, Chroma, Pinecone — indexing, querying, metadata filtering |
| **Agent Tools** | Tool calling, custom tools, ReAct agents, web search (Tavily), code execution |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/genai-practice.git
cd genai-practice
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional, for tracing
LANGCHAIN_TRACING_V2=true                 # Optional
```

---

## 🧰 Tech Stack

- **LLMs** — OpenAI GPT, Groq (LLaMA 3), Anthropic Claude
- **Frameworks** — LangChain, LangGraph
- **Embeddings** — OpenAI `text-embedding-3-small`, HuggingFace models
- **Vector DBs** — FAISS, ChromaDB, Pinecone
- **Agent Tools** — Tavily Search, custom Python tools
- **UI** — Streamlit (for interactive demos)
- **Language** — Python 3.10+

---

## 🚀 Featured Projects

### 🔬 Multi-Agent Research Pipeline
> `08_projects/research_pipeline/`

A LangGraph-based multi-agent system that takes a research query, searches the web via Tavily, synthesizes findings, and returns a structured report.

**Stack:** LangGraph · LangChain · Groq (`llama-3.3-70b-versatile`) · Tavily · Streamlit

---

## 📈 Learning Roadmap

- [x] Chat model basics & API usage
- [x] Embeddings & semantic similarity
- [x] LangChain prompt templates & chains
- [x] LangGraph stateful agent graphs
- [x] RAG pipeline with vector databases
- [x] Agent tools (Tavily, custom tools)
- [ ] Fine-tuning open-source models
- [ ] LLMOps & deployment (LangSmith, MLflow)
- [ ] Production-grade agent systems

---

## 📚 Resources

- [LangChain Docs](https://docs.langchain.com)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Groq API](https://console.groq.com)
- [Tavily Search](https://tavily.com)
- [FAISS](https://faiss.ai)
- [ChromaDB](https://docs.trychroma.com)

---

## 🙋 Author

**Swapnil**
Learning GenAI one agent at a time. 🚀

---

> ⭐ If you find this helpful, feel free to star the repo!